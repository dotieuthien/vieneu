"""Continuous batching over the frame graph (design §4).

One worker thread owns the GPU. Each tick it admits pending requests (one HF
prefill for all of them), replays the graph (one frame for every slot) and,
every ``K`` ticks or when a newcomer reaches ``leadin`` frames, decodes the new
frames of every row through ONE streaming codec session.

With ``overlap_codec`` (design §7.1, §7.3) the codec runs on its own thread and
CUDA stream: the worker packs each decode into a ``CodecJob`` (codes already
copied out of the ring on the GPU, rows that finish) and keeps stepping while
the codec thread runs the jobs FIFO. The worker runs on its own stream too, so
the two only meet through the job's CUDA event. Without it, the worker runs each
job itself on the default stream (the original behaviour).

The codec is ``SlotCodec`` (codec.py): a row keeps its scheduler slot in the
codec, rows with the same number of new frames share one forward, and a slot's
codec state is reset by the job that finalises its row. Jobs run FIFO, so that
reset always lands before the first job of the slot's next row.
"""
from __future__ import annotations

import collections
import contextlib
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .codec import SlotCodec, verify
from .frame import StreamFrame, check_sampling
from .profiler import PROF

log = logging.getLogger(__name__)

SAMPLES_PER_FRAME = 3840   # 80 ms @ 48 kHz


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.8
    top_k: int = 25
    top_p: float = 0.95
    repetition_penalty: float = 1.2

    def check(self) -> None:
        check_sampling(self.temperature, self.top_k, self.top_p, self.repetition_penalty)


class Request:
    """One text chunk in flight. Iterate for its audio (float32 @ 48 kHz).

    ``gen_done`` is set when the last frame is generated, before its audio is
    decoded: the cue to submit the next chunk. Closing (client gone) cancels.
    """

    def __init__(self, prompt: torch.Tensor, speaker_emb, cap: int, sampling: Sampling):
        self.prompt = prompt              # (T, n_vq + 1) token rows, on the host
        self.speaker_emb = speaker_emb    # (192,) numpy or None
        self.cap = cap                    # frame budget
        self.sampling = sampling
        self.audio: "queue.Queue" = queue.Queue()   # ndarray | None (end) | exception
        self.gen_done = threading.Event()
        self.cancelled = False
        # Owned by the worker thread.
        self.slot: Optional[int] = None
        self.f0 = 0                           # tick of the row's first frame
        self.decoded_upto = 0                 # tick up to which audio was emitted
        self.finished_at: Optional[int] = None
        self.marks: dict = {}                 # stage -> perf_counter, PROFILE=1 only

    def __iter__(self):
        try:
            while True:
                item = self.audio.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                PROF.mark(self, "consumed")
                yield item
        finally:
            PROF.request_done(self)
            self.close()

    def close(self) -> None:
        self.cancelled = True
        self.gen_done.set()

    def fail(self, err: BaseException) -> None:
        self.gen_done.set()
        self.audio.put(err)


@dataclass
class CodecJob:
    """One codec call, decided by the worker; the codec side only executes it."""
    rows: List[Tuple[Request, torch.Tensor]]   # rows with new frames: (request, codes (n_vq, n))
    done: List[Request]                        # rows that end with this job (audio then None)
    ready: Optional[torch.cuda.Event] = field(default=None)   # codes are on the device

    @property
    def reqs(self) -> List[Request]:
        return [r for r, _ in self.rows] + [r for r in self.done if all(r is not x for x, _ in self.rows)]


class Scheduler:
    def __init__(self, engine, *, max_streams: int = 16, decode_frames: int = 4,
                 leadin_frames: int = 2, max_admit_per_tick: int = 16, overlap_codec: bool = True,
                 codec_queue: int = 1, codec_graphs: bool = True, codec_ring: bool = True,
                 codec_dtype: str = "fp32", verify_codec: bool = False):
        """``engine`` is ``vieneu``'s ``VieNeuTTSv3Turbo`` (model, codec, prompt builder).

        ``codec_queue`` bounds the jobs waiting behind the one being decoded. The
        codec is the slower side, so 1 already keeps it busy; a deeper queue only
        adds latency to every newcomer's first audio. ``codec_graphs`` replays the
        codec as CUDA graphs; ``verify_codec`` first checks ``SlotCodec`` against the
        SDK's streaming decode (``self.codec_check``).
        """
        self.engine = engine
        self.model = engine.model
        self.device = engine.device
        self.B = max_streams
        self.K = decode_frames
        self.leadin = max(1, min(leadin_frames, decode_frames))
        self.max_admit = max_admit_per_tick
        self.frame = StreamFrame(self.model, self.B, int(self.model.config.max_position_embeddings))
        if PROF.enabled:
            PROF.frame_anatomy(self.frame)     # before any other thread touches the GPU
        self.n_vq = self.frame.n_vq
        tok = engine.audio_tokenizer
        self.codec_check = None
        if verify_codec:
            self.codec_check = verify(tok, self.B, self.n_vq, self.K, graphs=codec_graphs,
                                      ring=codec_ring, dtype=codec_dtype)
            log.info("codec check vs the SDK's fp32 streaming decode: %s", self.codec_check)
        self.codec = SlotCodec(tok, self.B, self.n_vq, self.K, graphs=codec_graphs, ring=codec_ring,
                               dtype=codec_dtype)
        self.codec_graphs = self.codec.graphs
        self.tick = 0
        self.slots: List[Optional[Request]] = [None] * self.B
        self._codec_rows: List[Request] = []    # admitted rows whose audio is not finished
        self._pending: "collections.deque[Request]" = collections.deque()
        self._cv = threading.Condition()
        self._running = True
        self.error: Optional[BaseException] = None
        self.overlap_codec = overlap_codec
        self._codec_error: Optional[BaseException] = None
        self._ar_stream = self._codec_stream = self._codec_q = self._codec_thread = None
        if overlap_codec:
            self._ar_stream = torch.cuda.Stream()
            self._codec_stream = torch.cuda.Stream()
            self._codec_q: "queue.Queue[Optional[CodecJob]]" = queue.Queue(maxsize=codec_queue)
            self._codec_thread = threading.Thread(target=self._codec_loop, name="vieneu-lite-codec", daemon=True)
            self._codec_thread.start()
        self._thread = threading.Thread(target=self._run, name="vieneu-lite-scheduler", daemon=True)
        self._thread.start()

    # ── caller side ─────────────────────────────────────────────────────────

    def submit(self, phonemes: str, speaker_emb, ref_codes, max_frames: int,
               sampling: Sampling = Sampling()) -> Request:
        sampling.check()
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        e = self.engine
        with PROF.region("submit.prompt"):
            prompt = e._build_prompt_2d(phonemes, None, ref_codes, e._resolve_style_id())
        if prompt.shape[0] + max_frames + 2 > self.frame.max_len:
            raise ValueError(f"prompt of {prompt.shape[0]} rows + {max_frames} frames "
                             f"does not fit the {self.frame.max_len}-token cache")
        req = Request(prompt, speaker_emb, max_frames, sampling)
        PROF.mark(req, "submit")
        with self._cv:
            if self.error is not None:
                raise RuntimeError("scheduler is down") from self.error
            self._pending.append(req)
            self._cv.notify()
        return req

    @property
    def n_active(self) -> int:
        return sum(r is not None for r in self.slots)

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify()
        self._thread.join(timeout=5)
        if self._codec_thread is not None:
            self._codec_thread.join(timeout=5)

    # ── worker ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        stream_ctx = contextlib.nullcontext()
        if self._ar_stream is not None:
            self._ar_stream.wait_stream(torch.cuda.current_stream())   # state built on the default stream
            stream_ctx = torch.cuda.stream(self._ar_stream)
        try:
            with stream_ctx:
                while True:
                    with self._cv:
                        while (self._running and not self._pending and self.n_active == 0
                               and self._codec_error is None):
                            PROF.unchain("tick")        # the next frame follows an idle wait
                            with PROF.region("worker.idle"):
                                self._cv.wait()
                        if self._codec_error is not None:
                            raise RuntimeError("codec thread failed") from self._codec_error
                        if not self._running:
                            return
                        todo = self._take_pending()
                    if todo:
                        self._admit(todo)
                    if self.n_active:
                        self._step()
        except BaseException as e:   # noqa: BLE001 — everyone waiting must hear about it
            log.exception("scheduler died")
            with self._cv:
                self.error = e
                self._running = False
                dead = [r for r in self.slots if r is not None] + list(self._pending)
                self._pending.clear()
            for r in dead:
                r.fail(e)
        finally:
            if self._codec_q is not None:
                self._codec_q.put(None)       # codec thread finishes queued jobs, then exits

    def _take_pending(self) -> List[Request]:
        """Pop up to ``min(free slots, max_admit)`` live requests (caller holds ``_cv``)."""
        n = min(self.B - self.n_active, self.max_admit)
        todo = []
        while self._pending and len(todo) < n:
            r = self._pending.popleft()
            if r.cancelled:
                r.audio.put(None)
            else:
                PROF.mark(r, "taken")
                todo.append(r)
        return todo

    @torch.no_grad()
    @PROF.timed("admit", gpu=True)
    def _admit(self, todo: List[Request]) -> None:
        # Build each prompt on its own: a bad voice/prompt fails only that request.
        reqs, embeds, spks = [], [], []
        for r in todo:
            try:
                spk = self.engine._resolve_speaker_emb(r.speaker_emb)
                x = self.model._build_inputs_embeds(r.prompt.unsqueeze(0).to(self.device), speaker_emb=spk)[0]
            except Exception as e:   # noqa: BLE001
                r.fail(e)
                continue
            reqs.append(r)
            embeds.append(x)
            spks.append(spk)
        if not reqs:
            return
        try:
            last_h, kv, T = self._prefill(embeds)
        except torch.cuda.OutOfMemoryError as e:
            for r in reqs:
                r.fail(e)
            return

        free = [b for b, r in enumerate(self.slots) if r is None]
        for i, r in enumerate(reqs):
            Ti = embeds[i].shape[0]
            keys = [k[i, :, T - Ti:] for k, _ in kv]       # real tokens only, (n_kv, Ti, hd)
            values = [v[i, :, T - Ti:] for _, v in kv]
            b = free[0]
            s = r.sampling
            try:
                self.frame.admit(b, self.tick, keys, values, last_h[i], spks[i], r.cap,
                                 temperature=s.temperature, top_k=s.top_k, top_p=s.top_p,
                                 repetition_penalty=s.repetition_penalty)
            except ValueError as e:
                r.fail(e)
                continue
            free.pop(0)
            r.slot, r.f0, r.decoded_upto = b, self.tick, self.tick
            self.slots[b] = r
            self._codec_rows.append(r)
            PROF.mark(r, "admitted")

    @PROF.timed("admit.prefill", gpu=True)
    def _prefill(self, embeds: List[torch.Tensor]):
        """One HF forward over left-padded prompts -> ``(last_hidden (n, H), [(k, v)] per layer, T)``."""
        n, T = len(embeds), max(e.shape[0] for e in embeds)
        x = embeds[0].new_zeros(n, T, embeds[0].shape[-1])
        mask = torch.zeros(n, T, device=self.device, dtype=torch.long)
        for i, e in enumerate(embeds):
            x[i, T - e.shape[0]:] = e
            mask[i, T - e.shape[0]:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model.semantic_backbone(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                                           use_cache=True, return_dict=True)
        cache = out.past_key_values
        if hasattr(cache, "layers"):
            kv = [(layer.keys, layer.values) for layer in cache.layers]
        else:
            kv = list(zip(cache.key_cache, cache.value_cache))
        return out.last_hidden_state[:, -1], kv, T

    @PROF.timed("step")
    def _step(self) -> None:
        self.frame.replay()
        self.tick += 1
        finished, length = self.frame.status()      # the one host read per tick
        for b, r in enumerate(self.slots):
            if r is None or r.finished_at is not None:
                continue
            if r.cancelled:
                r.finished_at = self.tick
            elif finished[b]:
                r.finished_at = length[b]
                r.gen_done.set()
        if self.tick % self.K == 0:
            self._decode(self._codec_rows)
        else:
            # Off a K boundary only newcomers that reached their lead-in are decoded:
            # every codec forward costs the same, so the others wait for the boundary.
            due = [r for r in self._codec_rows
                   if r.decoded_upto == r.f0 and self.tick - r.f0 >= self.leadin]
            if due:
                self._decode(due)

    @torch.no_grad()
    @PROF.timed("decode", gpu=True)
    def _decode(self, which: List[Request]) -> None:
        """Pack the new frames of ``which`` (codec rows) into a job, then run or queue it."""
        if not which:
            return
        # Every row has decoded_upto >= tick - K: all rows are decoded every K ticks.
        blk = self.frame.recent_codes(self.tick, self.K)      # (K, B, n_vq), frames [tick-K, tick), a copy
        base = self.tick - self.K
        rows, done = [], []
        for r in which:
            PROF.mark(r, "job")
            hi = self.tick if r.finished_at is None else min(r.finished_at, self.tick)
            if hi > r.decoded_upto:
                rows.append((r, blk[r.decoded_upto - base:hi - base, r.slot].T))   # (n_vq, n)
            r.decoded_upto = hi
            if r.finished_at is not None and hi >= r.finished_at:
                done.append(r)

        # The codes are copied out of the ring, so finished rows give their slot
        # back now, not when their audio comes out of the codec.
        for r in done:
            self.slots[r.slot] = None
        self._codec_rows = [r for r in self._codec_rows if all(r is not d for d in done)]
        job = CodecJob(rows, done)

        if self._codec_q is None:
            self._run_job(job)
            return
        blk.record_stream(self._codec_stream)    # don't let the allocator reuse it under the codec
        job.ready = torch.cuda.Event()
        job.ready.record()                       # on the worker's stream, after the copy
        with PROF.region("decode.backpressure"):
            self._codec_q.put(job)               # blocks while the codec is behind: backpressure

    # ── codec side ──────────────────────────────────────────────────────────

    def _codec_loop(self) -> None:
        """Run jobs FIFO on the codec stream. After a failure, keep draining so the
        worker never blocks on a full queue, failing every request it hands over."""
        with torch.cuda.stream(self._codec_stream):
            while True:
                with PROF.region("codec.idle"):
                    job = self._codec_q.get()
                if job is None:
                    break
                if self._codec_error is not None:
                    for r in job.reqs:
                        r.fail(self._codec_error)
                    continue
                try:
                    self._run_job(job)
                except BaseException as e:   # noqa: BLE001
                    log.exception("codec failed")
                    with self._cv:
                        self._codec_error = e
                        self._cv.notify()
                    for r in job.reqs:
                        r.fail(e)

    @torch.no_grad()
    @PROF.timed("codec")
    def _run_job(self, job: CodecJob) -> None:
        if job.ready is not None:
            torch.cuda.current_stream().wait_event(job.ready)
        groups: Dict[int, List[Tuple[Request, torch.Tensor]]] = {}
        for r, c in job.rows:
            PROF.mark(r, "codec")
            groups.setdefault(c.shape[1], []).append((r, c))
        out = []
        with PROF.region("codec.decode", gpu=True):
            for n, rows in groups.items():                 # one forward per frame count
                audio = self.codec.decode(n, [r.slot for r, _ in rows], torch.stack([c for _, c in rows], 1))
                out.append((rows, audio.float().mean(1)))   # (rows, n * 3840), channels averaged
        with PROF.region("codec.d2h"):
            out = [(rows, audio.cpu().numpy()) for rows, audio in out]
        self.codec.reset([r.slot for r in job.done])
        for rows, audio in out:
            for k, (r, _) in enumerate(rows):
                if not r.cancelled:
                    PROF.mark(r, "audio")
                    r.audio.put(np.array(audio[k], dtype=np.float32))
        for r in job.done:
            r.audio.put(None)
