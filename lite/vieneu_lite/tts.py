"""Text -> streamed 48 kHz audio on top of the scheduler (design §4.5).

The ``vieneu`` package still loads the model, codec, tokenizer, preset voices,
phonemizer and watermarker, and provides the chunk/pause helpers. Only the
serving path is rewritten here.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np
from vieneu import Vieneu
from vieneu_utils.core_utils import gaps_to_silence, max_expected_frames, pause_pad_samples
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

from .profiler import PROF
from .scheduler import Request, Sampling, Scheduler

log = logging.getLogger(__name__)

SAMPLE_RATE = 48_000


class LiteTTS:
    def __init__(self, *, max_streams: int = 16, decode_frames: int = 4, leadin_frames: int = 2,
                 overlap_codec: bool = True, codec_graphs: bool = True, codec_ring: bool = True,
                 codec_dtype: str = "fp32", verify_codec: bool = False,
                 watermark: bool = True, max_chars: int = 256, max_new_frames: int = 300):
        self.sdk = Vieneu(mode="v3turbo", backend="pytorch", device="cuda")
        self.sched = Scheduler(self.sdk.engine, max_streams=max_streams, decode_frames=decode_frames,
                               leadin_frames=leadin_frames, overlap_codec=overlap_codec,
                               codec_graphs=codec_graphs, codec_ring=codec_ring,
                               codec_dtype=codec_dtype, verify_codec=verify_codec)
        self.watermarker = self.sdk.watermarker if watermark else None
        if watermark and self.watermarker is None:
            log.warning("watermark requested but resemble-perth is not installed; audio is NOT watermarked")
        self.max_chars = max_chars
        self.max_new_frames = max_new_frames

    def voices(self) -> List[str]:
        return [name for _, name in self.sdk.list_preset_voices()]

    def has_voice(self, name: str) -> bool:
        return self.sdk.resolve_voice_name(name) is not None

    def warmup(self) -> None:
        """First kernels / cuDNN plans / codec session are slow; pay that at startup."""
        for _ in self.stream("Xin chào."):
            pass

    def stream(self, text: str, voice: Optional[str] = None,
               sampling: Sampling = Sampling()) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks (~320 ms each) for ``text``. Closing the
        iterator cancels whatever is generating or queued."""
        sampling.check()
        preset = self.sdk.get_preset_voice(voice)          # ValueError if unknown
        spk, ref_codes = preset["speaker_emb"], preset.get("codes")
        with PROF.region("text.split"):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=self.max_chars)
        pauses = gaps_to_silence(gaps)

        def submit(i: int) -> Request:
            with PROF.region("text.phonemize"):
                ph = phonemize_text_with_emotions(chunks[i])
            cap = min(self.max_new_frames, max_expected_frames(ph))   # stop babbling past a missed EOS
            return self.sched.submit(ph, spk, ref_codes, cap, sampling)

        if not chunks:
            return
        cur: Optional[Request] = submit(0)
        nxt: Optional[Request] = None
        prev: Optional[np.ndarray] = None    # last audio sent, to size the pause
        try:
            for i in range(len(chunks)):
                has_next = i + 1 < len(chunks)
                first = True
                for wav in cur:
                    # Hand the next chunk to the scheduler as soon as this one's last
                    # frame exists: one slot per listener, the join costs one prefill.
                    if has_next and nxt is None and cur.gen_done.is_set():
                        nxt = submit(i + 1)
                    if len(wav) == 0:
                        continue
                    if first and prev is not None:
                        pad = pause_pad_samples(prev, wav, SAMPLE_RATE, pauses[i - 1])
                        if pad > 0:
                            yield np.zeros(pad, dtype=np.float32)
                    first = False
                    prev = wav
                    yield self._watermark(wav)
                if has_next:
                    cur, nxt = (nxt or submit(i + 1)), None
        finally:
            for r in (cur, nxt):
                if r is not None:
                    r.close()

    @PROF.timed("watermark")
    def _watermark(self, wav: np.ndarray) -> np.ndarray:
        if self.watermarker is None:
            return wav
        return self.watermarker.apply_watermark(wav, sample_rate=SAMPLE_RATE)

    def close(self) -> None:
        self.sched.close()
