"""Streaming MOSS codec over the scheduler's slots, one forward per job (design §7.4).

The SDK's streaming ``batch_decode`` goes through ``MossAudioTokenizerDecodeSession.step``,
which runs the decoder ONE frame at a time and reads every slot's audio length
with ``.item()``: ~900 small ops plus a host sync per slot per frame. A 4-frame
call was ~180 ms of CPU for ~54 ms of GPU work on an RTX 3060 (profiler, c=1),
i.e. the codec alone cost RTF ~0.57.

The decoder is causal with a finite attention context per layer and its streaming
attention takes a chunk of any length (``_forward_streaming_sdpa``), so N frames
in one forward give the audio of N one-frame forwards (``verify`` checks it
against the SDK path). Here:

* rows with the same number of new frames share ONE ``_decode_frame`` call; the
  exec mask keeps every other slot's state untouched;
* codec slot == scheduler slot. Jobs run FIFO, so the job that finalises a slot
  (and resets it) always runs before the first job of the slot's next row;
* with ``graphs=True`` each forward length 1..K is a CUDA graph over all slots,
  exec-mask copies included: a job is a few copies + one replay per length;
* with ``ring=True`` attention keeps K/V in a ring of ``context + K`` positions
  per layer (``_ring_sdpa``): a forward writes only its new positions, where the
  SDK concatenates the whole cache, slices it, ``where``s it and copies it back
  (~5 passes over ~400 MB for 16 slots in fp32);
* ``dtype="bf16"`` runs the decoder transformers under bf16 autocast.

The SDK's own ``use_cuda_graph`` mode returned wrong audio because ``step``
kept views into the graph's static output across frames; outputs are gathered
into fresh tensors here.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _ring_sdpa(self, x: torch.Tensor, state) -> torch.Tensor:
    """``MossAudioTokenizerMultiheadAttention._forward_streaming_sdpa`` over a ring.

    Index ``i`` of the ring holds the newest position ``p <= last`` with
    ``p % C == i``. A capacity ``C >= context + T`` keeps every key a query can
    see; a slot outside the exec mask still writes its chunk, but only over
    positions at least ``context`` older than anything it will attend to, and its
    offset does not move, so its next real chunk overwrites them.
    """
    B, T, _ = x.shape
    q, k, v = self._project_qkv(x)                                   # (B, H, T, D)
    if self.rope is not None:
        q, k = self.rope(q, k, state.offset, time_before_heads=False)
    C = self._ring_capacity
    ring = getattr(state, "_ring", None)
    if ring is None or ring.dtype != k.dtype:
        ring = state._ring = k.new_zeros(2, B, self.num_heads, C, k.shape[-1])
    pos_q = state.offset.view(-1, 1) + torch.arange(T, device=x.device)          # (B, T)
    at = torch.remainder(pos_q, C).view(B, 1, T, 1).expand(-1, self.num_heads, -1, k.shape[-1])
    ring[0].scatter_(2, at, k)
    ring[1].scatter_(2, at, v)
    last = pos_q[:, -1:]
    pos_k = last - torch.remainder(last - torch.arange(C, device=x.device), C)   # (B, C)
    delta = pos_q[:, :, None] - pos_k[:, None, :]
    bias = (pos_k >= 0)[:, None, :] & (delta >= 0) & (delta < self.context)
    out = F.scaled_dot_product_attention(q, ring[0], ring[1], bias[:, None], dropout_p=0.0)
    state.offset[:] = torch.where(state.exec_mask, state.offset + T, state.offset)
    return out.transpose(1, 2).reshape(B, T, self.embed_dim)


class SlotCodec:
    def __init__(self, model, B: int, n_vq: int, max_frames: int, *, graphs: bool = True,
                 ring: bool = True, dtype: str = "fp32"):
        self.model, self.B, self.n_vq, self.max_frames = model, B, n_vq, max_frames
        self.device = next(model.parameters()).device
        mod = sys.modules[type(model).__module__]
        streaming_module = mod.StreamingModule
        if getattr(model, "_active_decode_session", None) is not None:
            raise RuntimeError("close the SDK's streaming decode session first")
        self._old_dtype = model.compute_dtype_name
        model.set_compute_dtype(dtype)
        self._patched = []
        if ring:
            up = 1                     # decoder tokens per code frame at this point
            for d in model.decoder:
                if isinstance(d, mod.MossAudioTokenizerPatchedPretransform):
                    up *= d.patch_size
                    continue
                for m in d.modules():
                    if isinstance(m, mod.MossAudioTokenizerMultiheadAttention):
                        m._ring_capacity = m.context + up * max_frames
                        m._forward_streaming_sdpa = types.MethodType(_ring_sdpa, m)
                        self._patched.append(m)
        self._stack = contextlib.ExitStack()
        for m in model.decoder:
            if isinstance(m, streaming_module):
                self._stack.enter_context(m.streaming(batch_size=B))
        self.states = [m._streaming_state for d in model.decoder for m in d.modules()
                       if isinstance(m, streaming_module) and m._streaming_state is not None]
        self._mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.graphs = graphs and self.device.type == "cuda"
        if self.graphs:
            # Own capture stream: cuBLAS keeps a workspace per stream and a graph bakes
            # in the capture stream's. torch.cuda.graph's default capture stream is
            # shared with the frame graph, and the two graphs replay concurrently (AR
            # and codec streams): one shared split-K workspace hung the GPU.
            self._capture_stream = torch.cuda.Stream()
            pool = torch.cuda.graph_pool_handle()
            for n in range(1, max_frames + 1):
                self._graphs[n] = self._capture(n, pool)

    # ── forward ─────────────────────────────────────────────────────────────

    def _forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``codes`` (n_vq, B, n) for every slot; only slots in ``self._mask`` advance."""
        for st in self.states:
            st.exec_mask.copy_(self._mask)
        return self.model._decode_frame(codes, lengths).audio          # (B, channels, n * 3840)

    @torch.no_grad()
    def _capture(self, n: int, pool):
        codes = torch.zeros(self.n_vq, self.B, n, dtype=torch.long, device=self.device)
        lengths = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._mask.zero_()                 # warm-up runs every kernel but advances no slot
        s = self._capture_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._forward(codes, lengths)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=s):
            audio = self._forward(codes, lengths)
        return graph, codes, lengths, audio

    @torch.no_grad()
    def decode(self, n: int, slots: Sequence[int], codes: torch.Tensor) -> torch.Tensor:
        """Next ``n`` frames of ``slots``: ``codes`` (n_vq, len(slots), n) -> audio
        (len(slots), channels, n * 3840), a fresh tensor on the device."""
        if not 1 <= n <= self.max_frames:
            raise ValueError(f"{n} frames per row, the codec takes 1..{self.max_frames}")
        idx = torch.tensor(list(slots), dtype=torch.long).to(self.device, non_blocking=True)
        self._mask.zero_()
        self._mask.index_fill_(0, idx, True)
        if self.graphs:
            graph, buf, lengths, audio = self._graphs[n]
            buf.index_copy_(1, idx, codes)
            lengths.copy_(self._mask.long() * n)
            graph.replay()
        else:
            buf = codes.new_zeros(self.n_vq, self.B, n)
            buf.index_copy_(1, idx, codes)
            audio = self._forward(buf, self._mask.long() * n)
        return audio.index_select(0, idx)

    def reset(self, slots: Sequence[int]) -> None:
        """Clear the streaming state of ``slots`` (a row left them)."""
        if not slots:
            return
        mask = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        mask[list(slots)] = True
        for st in self.states:
            st.reset(mask)

    def close(self) -> None:
        self._graphs.clear()
        self._stack.close()
        for m in self._patched:              # back to the SDK's attention
            del m._forward_streaming_sdpa
        self._patched = []
        self.model.set_compute_dtype(self._old_dtype)


# ── equivalence check ───────────────────────────────────────────────────────

@torch.no_grad()
def verify(model, B: int, n_vq: int, max_frames: int, *, graphs: bool, ring: bool = True,
           dtype: str = "fp32", seed: int = 0) -> dict:
    """Decode the same random codes through the SDK's streaming path (fp32, one
    frame per forward) and through ``SlotCodec`` (whole chunks, rows joining and
    leaving mid-stream); returns the worst per-row max-abs diff and correlation.
    Rows run ~13 s, past every layer's attention context (<= 10 s), so the ring
    wraps."""
    g = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    m = max_frames
    # Row 0 streams from the start and leaves after call 39; row 1 joins at call 1
    # with a lead-in, then mixes full and short chunks (two forwards per call).
    plan = []
    for i in range(48):
        step = [(0, 2 if i == 0 else m)] if i < 40 else []
        if i >= 1:
            step.append((1, 2 if i == 1 else (m if i % 3 else max(1, m - 1))))
        plan.append(step)
    total = [sum(n for step in plan for r, n in step if r == row) for row in (0, 1)]
    codes = [torch.randint(0, 1024, (n_vq, t), generator=g).to(dev) for t in total]

    # SDK reference: rows are positions in the session's list.
    ref = [[], []]
    pos = [0, 0]
    alive: List[int] = []
    model._reset_batch_decode_streaming_state()
    for i, step in enumerate(plan):
        for r, _ in step:
            if r not in alive:
                alive.append(r)
        want = dict(step)
        rows = []
        for r in alive:
            n = want.get(r, 0)
            rows.append(codes[r][:, pos[r]:pos[r] + n] if n else torch.zeros(n_vq, 1, dtype=torch.long, device=dev))
        leaving = [j for j, r in enumerate(alive) if i + 1 < len(plan) and r not in dict(plan[i + 1])]
        out = model.batch_decode(rows, num_quantizers=n_vq, streaming=True, max_batch_size=B,
                                 reset_stream=(i == 0), finalize_indices=leaving or None)
        for j, r in enumerate(alive):
            n = want.get(r, 0)
            if n:
                ref[r].append(out.audio[j, :, :n * 3840].float())
                pos[r] += n
        alive = [r for j, r in enumerate(alive) if j not in leaving]
    model._reset_batch_decode_streaming_state()

    # SlotCodec: row r lives in slot 3 + r (any slot works).
    codec = SlotCodec(model, B, n_vq, max_frames, graphs=graphs, ring=ring, dtype=dtype)
    got = [[], []]
    pos = [0, 0]
    try:
        for i, step in enumerate(plan):
            by_n: Dict[int, List[int]] = {}
            for r, n in step:
                by_n.setdefault(n, []).append(r)
            for n, rs in by_n.items():
                c = torch.stack([codes[r][:, pos[r]:pos[r] + n] for r in rs], dim=1)
                audio = codec.decode(n, [3 + r for r in rs], c)
                for k, r in enumerate(rs):
                    got[r].append(audio[k].float())
                    pos[r] += n
            nxt = dict(plan[i + 1]) if i + 1 < len(plan) else {}
            codec.reset([3 + r for r, _ in step if r not in nxt])
    finally:
        codec.close()

    worst_diff, worst_corr = 0.0, 1.0
    for r in (0, 1):
        a, b = torch.cat(ref[r], -1).flatten(), torch.cat(got[r], -1).flatten()
        worst_diff = max(worst_diff, (a - b).abs().max().item())
        worst_corr = min(worst_corr, torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    return {"graphs": graphs, "ring": ring, "dtype": dtype, "max_abs_diff": worst_diff, "min_corr": worst_corr}
