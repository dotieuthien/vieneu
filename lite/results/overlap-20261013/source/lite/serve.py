"""VieNeu-TTS v3 Turbo lite serving: POST /v1/audio/speech -> streamed PCM s16le / WAV, 48 kHz."""
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
import io
import logging
import os
import struct
import socket
import subprocess
import sys
import time
import uuid
import wave
from typing import Literal

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from bench_support import Trace, prepare_gpu, reset_seed

from vieneu_lite.scheduler import Sampling
from vieneu_lite.tts import SAMPLE_RATE, LiteTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vieneu_lite.api")

MODEL_ID = "vieneu-v3-turbo"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    prepare_gpu()
    t = time.perf_counter()
    subprocess.run(["nvidia-smi"], check=True)
    app.state.tts = LiteTTS(
        max_streams=int(os.getenv("MAX_STREAMS", "16")),
        decode_frames=int(os.getenv("DECODE_FRAMES", "4")),
        leadin_frames=int(os.getenv("LEADIN_FRAMES", "2")),
        overlap_codec=os.getenv("OVERLAP_CODEC", "1") != "0",
        watermark=os.getenv("WATERMARK", "1") != "0",
    )
    app.state.benchmark = None
    app.state.tts.warmup()
    log.info("ready in %.1fs", time.perf_counter() - t)
    yield
    job = app.state.benchmark
    if job and job.poll() is None:
        job.terminate()
        job.wait(timeout=10)
    app.state.tts.close()


app = FastAPI(lifespan=lifespan)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")


class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str = Field(min_length=1, max_length=20_000)
    voice: str | None = None
    response_format: Literal["pcm", "wav"] = "pcm"
    sample_rate: Literal[48000] = SAMPLE_RATE
    stream_format: Literal["audio"] = "audio"
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(25, ge=0, le=1024)
    top_p: float = Field(0.95, gt=0, le=1)
    repetition_penalty: float = Field(1.2, gt=0, le=4)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def wav_header(rate: int) -> bytes:
    """WAV header with an unknown (max) length, so players start immediately."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.setnframes(0)
    b = bytearray(buf.getvalue())
    struct.pack_into("<I", b, 4, 0xFFFFFFFF)            # RIFF size
    struct.pack_into("<I", b, len(b) - 4, 0xFFFFFFFF)   # data size
    return bytes(b)


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest, x_request_id: str | None = Header(None)):
    tts: LiteTTS = app.state.tts
    if tts.sched.error is not None:
        raise HTTPException(503, "scheduler is down; restart the process")
    if req.voice and not tts.has_voice(req.voice):
        raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices")
    sampling = Sampling(req.temperature, req.top_k, req.top_p, req.repetition_penalty)
    try:
        sampling.check()
    except ValueError as e:
        raise HTTPException(400, str(e))
    trace = Trace(x_request_id)
    rid = trace.rid

    def body():
        t0 = time.perf_counter()
        first, samples = None, 0
        try:
            if req.response_format == "wav":
                yield wav_header(SAMPLE_RATE)
            for chunk in tts.stream(req.input, req.voice, sampling):
                if first is None:
                    first = time.perf_counter() - t0
                trace.chunk(len(chunk))
                samples += len(chunk)
                yield pcm16(chunk)
        finally:
            trace.finish()
            total, audio_s = time.perf_counter() - t0, samples / SAMPLE_RATE
            log.info("%s ttfa=%s total=%.2fs audio=%.2fs rtf=%s", rid,
                     f"{first * 1000:.0f}ms" if first is not None else "-", total, audio_s,
                     f"{total / audio_s:.2f}" if audio_s else "-")

    media = "audio/wav" if req.response_format == "wav" else "audio/pcm"
    return StreamingResponse(body(), media_type=media,
                             headers={"X-Request-Id": rid, "X-Sample-Rate": str(SAMPLE_RATE)})


@app.get("/v1/voices")
def voices():
    return {"voices": app.state.tts.voices()}


@app.get("/health")
def health():
    tts: LiteTTS = app.state.tts
    s = tts.sched
    if s.error is not None:
        raise HTTPException(503, f"scheduler died: {s.error!r}")
    return {"status": "ok", "backend": "pytorch-lite", "gpu": torch.cuda.get_device_name(),
            "sample_rate": SAMPLE_RATE, "max_streams": s.B, "decode_frames": s.K,
            "leadin_frames": s.leadin, "overlap_codec": s.overlap_codec,
            "active": s.n_active, "pending": s.n_pending,
            "watermark": tts.watermarker is not None, "cuda": torch.version.cuda,
            "versions": {p: version(p) for p in ("vieneu", "torch", "transformers", "sea-g2p")}}


@app.post("/benchmark")
async def benchmark(concurrency: int = Query(8, ge=1, le=32),
                    repeats: int = Query(1, ge=1, le=100), seed: int = 42):
    job = app.state.benchmark
    if job and job.poll() is None:
        raise HTTPException(409, "Benchmark already running")
    name = f"bench-{concurrency}.log"
    base = f"http://127.0.0.1:{int(os.getenv('PORT', '8000'))}"
    command = [sys.executable, "-u", "benchmark.py", "--base", base, "--bench", str(concurrency)]
    if os.getenv("BENCH_MODE") == "1":
        command += ["--repeats", str(repeats), "--seed", str(seed),
                    "--output", str(RESULTS / f"bench-{concurrency}.jsonl")]
    elif repeats != 1:
        raise HTTPException(400, "Repeated benchmark requires BENCH_MODE=1")
    with (RESULTS / name).open("w") as output:
        print(f"benchmark_host={socket.gethostname()} server_pid={os.getpid()} base={base}",
              file=output, flush=True)
        app.state.benchmark = subprocess.Popen(
            command, cwd=RESULTS.parent, stdout=output, stderr=subprocess.STDOUT)
    return {"status": "started", "results": f"/results/{name}"}


@app.post("/benchmark/seed")
def benchmark_seed(seed: int = 42):
    if os.getenv("BENCH_MODE") != "1":
        raise HTTPException(404)
    reset_seed(seed)
    return {"seed": seed}


@app.get("/benchmark")
def benchmark_status():
    job = app.state.benchmark
    return {"started": job is not None, "running": job is not None and job.poll() is None,
            "exit_code": job.poll() if job else None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
