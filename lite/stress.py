"""Stability + quality under a sustained load (run on the GPU host via POST /stress).

1. Alone: each text ``--alone`` times, one request at a time.
2. Load: ``--concurrency`` clients loop requests for ``--duration`` seconds,
   staggered so rows keep joining and leaving the batch. Every request is timed;
   a random sample of their audio is kept.
3. Quality: an ASR model (PhoWhisper) transcribes the alone and the load samples
   after the load, on the idle GPU; CER against the input text. A slot leak or a
   bad codec reset garbles a row's audio, so it shows as a load CER well above
   the alone CER. (Waveforms cannot be compared directly: the bf16 AR model can
   pick different codes when its prompt is prefilled in another batch.)

Writes ``results/stress-C.jsonl`` (one row per request), ``results/stress-C.json``
(summary) and WAVs under ``results/stress-C/``.
"""
import argparse
import json
import random
import re
import threading
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import requests

from benchmark import server_url

SR = 48_000
TEXTS = [
    "Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.",
    "Hôm nay trời Hà Nội se lạnh, gió mùa đông bắc tràn về từ sáng sớm.",
    "Công nghệ chuyển văn bản thành giọng nói đang thay đổi cách chúng ta nghe tin tức, đọc sách "
    "và học ngoại ngữ. Một mô hình tốt cần đọc tự nhiên, đúng dấu và ngắt nghỉ hợp lý.",
]


def speak(base: str, text: str):
    body = {"model": "vieneu-v3-turbo", "input": text, "response_format": "pcm"}
    t0 = time.perf_counter()
    first, buf = None, bytearray()
    with requests.post(f"{base}/v1/audio/speech", json=body, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        for chunk in r.iter_content(chunk_size=None):
            if chunk and first is None:
                first = time.perf_counter() - t0
            buf += chunk
    return first, time.perf_counter() - t0, bytes(buf)


def save(path: Path, b: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def cer(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * len(xs) + 0.5)) - 1))] if xs else None


def transcribe(clips, model_id: str):
    """``clips``: list of int16 PCM bytes @ 48 kHz -> texts."""
    import torch
    import torchaudio.functional as AF
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model_id, device=0, torch_dtype=torch.float16)
    xs = [AF.resample(torch.from_numpy(np.frombuffer(b, dtype="<i2").astype(np.float32) / 32767), SR, 16_000).numpy()
          for b in clips]
    out = asr([{"raw": x, "sampling_rate": 16_000} for x in xs], batch_size=16)
    return [o["text"] for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=180)
    ap.add_argument("--alone", type=int, default=5)
    ap.add_argument("--sample", type=int, default=150, help="load outputs kept for ASR")
    ap.add_argument("--asr", default="vinai/PhoWhisper-small")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = server_url(a.base)
    out = Path(__file__).resolve().parent / "results"
    wavs = out / f"stress-{a.concurrency}"
    wavs.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    alone = []                                     # (text index, pcm)
    for i, text in enumerate(TEXTS):
        for k in range(a.alone):
            _, _, b = speak(base, text)
            alone.append((i, b))
            if k == 0:
                save(wavs / f"alone-{i}.wav", b)
    print(f"alone: {len(alone)} clips", flush=True)

    rows, kept, lock = [], [], threading.Lock()
    seen = [0]
    deadline = time.time() + a.duration

    def client(c):
        time.sleep(rng.random() * 2)               # staggered joins
        k = c
        while time.time() < deadline:
            i = k % len(TEXTS)
            k += 1
            row = {"client": c, "text": i, "t": time.time()}
            try:
                first, total, b = speak(base, TEXTS[i])
                x = np.frombuffer(b, dtype="<i2")
                audio = len(x) / SR
                row.update(ok=True, ttfa_s=first, total_s=total, audio_s=audio, rtf=total / audio,
                           peak=int(np.abs(x).max()) if len(x) else 0)
                with lock:                         # reservoir sample for ASR
                    seen[0] += 1
                    if len(kept) < a.sample:
                        kept.append((i, b, len(rows)))
                    elif rng.random() < a.sample / seen[0]:
                        kept[rng.randrange(a.sample)] = (i, b, len(rows))
            except Exception as e:   # noqa: BLE001
                row.update(ok=False, error=str(e))
            with lock:
                rows.append(row)

    ths = [threading.Thread(target=client, args=(c,)) for c in range(a.concurrency)]
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t0
    print(f"load: {len(rows)} requests in {wall:.0f}s", flush=True)

    texts = transcribe([b for _, b in alone] + [b for _, b, _ in kept], a.asr)
    alone_cer = [cer(TEXTS[i], t) for (i, _), t in zip(alone, texts)]
    load_cer = [cer(TEXTS[i], t) for (i, _, _), t in zip(kept, texts[len(alone):])]
    for (i, b, r), c, t in zip(kept, load_cer, texts[len(alone):]):
        rows[r].update(cer=c, asr=t)
    worst = sorted(range(len(kept)), key=lambda j: -load_cer[j])[:3]
    for rank, j in enumerate(worst):
        save(wavs / f"load-worst{rank}-text{kept[j][0]}.wav", kept[j][1])
    for i in range(len(TEXTS)):
        j = next((j for j, (ti, _, _) in enumerate(kept) if ti == i), None)
        if j is not None:
            save(wavs / f"load-{i}.wav", kept[j][1])

    with (out / f"stress-{a.concurrency}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r.get("ok")]
    alone_len = {i: np.median([len(b) for ti, b in alone if ti == i]) / 2 / SR for i in range(len(TEXTS))}
    summary = {
        "concurrency": a.concurrency, "duration_s": wall, "requests": len(rows), "ok": len(ok),
        "errors": sorted({r["error"] for r in rows if not r.get("ok")})[:5],
        "audio_s_per_s": sum(r["audio_s"] for r in ok) / wall,
        "ttfa_ms": {p: q([r["ttfa_s"] * 1000 for r in ok], p) for p in (0.5, 0.9, 0.99)},
        "rtf": {p: q([r["rtf"] for r in ok], p) for p in (0.5, 0.9, 0.99)},
        "silent_or_clipped": sum(1 for r in ok if r["peak"] < 300 or r["peak"] >= 32767),
        "duration_vs_alone": {p: q([r["audio_s"] / alone_len[r["text"]] for r in ok], p) for p in (0.01, 0.5, 0.99)},
        "asr": a.asr,
        "cer_alone": {p: q(alone_cer, p) for p in (0.5, 0.9, 1.0)},
        "cer_load": {p: q(load_cer, p) for p in (0.5, 0.9, 1.0)},
        "cer_load_over_0.2": sum(c > 0.2 for c in load_cer), "load_sampled": len(load_cer),
        "worst_load": [{"text": kept[j][0], "cer": load_cer[j], "asr": texts[len(alone) + j]} for j in worst],
        "alone_asr": [{"text": i, "cer": c, "asr": t} for (i, _), c, t in zip(alone, alone_cer, texts)][::a.alone],
    }
    (out / f"stress-{a.concurrency}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
