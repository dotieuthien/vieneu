"""VieNeu-TTS v3 Turbo lite serving: POST /v1/audio/speech -> streamed PCM s16le / WAV, 48 kHz."""
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
import faulthandler
import io
import logging
import os
import struct
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from typing import Literal

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from bench_support import Trace, prepare_gpu, reset_seed

from vieneu_lite.profiler import PROF, TraceBusy
from vieneu_lite.scheduler import Sampling
from vieneu_lite.tts import SAMPLE_RATE, LiteTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vieneu_lite.api")

MODEL_ID = "vieneu-v3-turbo"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
# Traces and reports; on VKong point it at the storage volume to pull them back.
PROFILE_DIR = Path(os.getenv("PROFILE_DIR", RESULTS / "profile"))


def _watchdog(sched, stall: float = 30.0) -> None:
    """PROFILE=1: print every thread's stack when rows are active but no frame has
    advanced for ``stall`` seconds (a GPU hang leaves no Python error behind)."""
    last, since = -1, time.monotonic()
    while True:
        time.sleep(5)
        if sched.tick != last or not sched.n_active:
            last, since = sched.tick, time.monotonic()
        elif time.monotonic() - since > stall:
            log.error("no frame for %.0fs with %d active rows; stacks:", time.monotonic() - since, sched.n_active)
            faulthandler.dump_traceback(all_threads=True)
            since = time.monotonic()


@asynccontextmanager
async def lifespan(app):
    prepare_gpu()
    t = time.perf_counter()
    faulthandler.dump_traceback_later(60, repeat=True)   # a hung startup prints every thread's stack
    subprocess.run(["nvidia-smi"], check=True)
    app.state.tts = LiteTTS(
        max_streams=int(os.getenv("MAX_STREAMS", "16")),
        decode_frames=int(os.getenv("DECODE_FRAMES", "4")),
        leadin_frames=int(os.getenv("LEADIN_FRAMES", "2")),
        overlap_codec=os.getenv("OVERLAP_CODEC", "1") != "0",
        codec_graphs=os.getenv("CODEC_GRAPHS", "1") != "0",
        codec_ring=os.getenv("CODEC_RING", "1") != "0",
        codec_dtype=os.getenv("CODEC_DTYPE", "fp32"),
        verify_codec=os.getenv("CODEC_VERIFY", "0") == "1",
        watermark=os.getenv("WATERMARK", "1") != "0",
    )
    app.state.benchmark = None
    app.state.tts.warmup()
    if PROF.enabled:
        anatomy = PROF.anatomy["text"] if PROF.anatomy else "no frame anatomy"
        log.info("PROFILE=1, files in %s\n%s", PROFILE_DIR, anatomy)
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        (PROFILE_DIR / "anatomy.txt").write_text(anatomy + "\n")
        PROF.reset()                     # the window starts after warmup
        threading.Thread(target=_watchdog, args=(app.state.tts.sched,), name="watchdog", daemon=True).start()
    faulthandler.cancel_dump_traceback_later()
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
            "codec_graphs": s.codec_graphs, "codec_check": s.codec_check,
            "active": s.n_active, "pending": s.n_pending,
            "watermark": tts.watermarker is not None, "profile": PROF.enabled, "cuda": torch.version.cuda,
            "versions": {p: version(p) for p in ("vieneu", "torch", "transformers", "sea-g2p")}}


def _profiling():
    if not PROF.enabled:
        raise HTTPException(404, "start the server with PROFILE=1")


@app.get("/profile")
def profile(format: Literal["text", "json"] = "text"):
    """Per-region host/GPU time since the last reset (see vieneu_lite/profiler.py)."""
    _profiling()
    snap = PROF.snapshot()
    return snap if format == "json" else PlainTextResponse(PROF.report(snap) + "\n")


@app.post("/profile/reset")
def profile_reset():
    _profiling()
    PROF.reset()
    return {"status": "reset"}


@app.post("/profile/trace")
def profile_trace(seconds: float = Query(2.0, gt=0, le=10),
                  format: Literal["text", "json"] = "text"):
    """torch.profiler over every thread for ``seconds``. Saves ``trace-*.json.gz`` (Perfetto)
    and ``trace-*.txt`` (this summary + the live report) in PROFILE_DIR."""
    _profiling()
    try:
        s = PROF.trace(seconds, PROFILE_DIR)
    except TraceBusy as e:
        raise HTTPException(409, str(e))
    (PROFILE_DIR / s["file"].replace(".json.gz", ".txt")).write_text(PROF.report() + "\n")
    return s if format == "json" else PlainTextResponse(s["text"] + "\n")


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


@app.post("/stress")
async def stress(concurrency: int = Query(32, ge=1, le=64), duration: float = Query(180, gt=0, le=1800)):
    """BENCH_MODE=1: stress.py on this host (sustained greedy load + quality check)."""
    if os.getenv("BENCH_MODE") != "1":
        raise HTTPException(404)
    job = app.state.benchmark
    if job and job.poll() is None:
        raise HTTPException(409, "Benchmark already running")
    base = f"http://127.0.0.1:{int(os.getenv('PORT', '8000'))}"
    name = f"stress-{concurrency}.log"
    with (RESULTS / name).open("w") as output:
        app.state.benchmark = subprocess.Popen(
            [sys.executable, "-u", "stress.py", "--base", base, "--concurrency", str(concurrency),
             "--duration", str(duration)], cwd=RESULTS.parent, stdout=output, stderr=subprocess.STDOUT)
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
