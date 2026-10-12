"""Serve the original VieNeu SDK: POST /v1/audio/speech -> PCM s16le, 48 kHz."""
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
import os
import socket
import subprocess
import sys
from threading import BoundedSemaphore
from typing import Literal

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from bench_support import Trace, prepare_gpu, reset_seed
from vieneu import Vieneu

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    prepare_gpu()
    app.state.tts = Vieneu(mode="v3turbo", backend="pytorch", device="cuda",
                           max_streams=int(os.getenv("MAX_STREAMS", "16")))
    app.state.benchmark = None
    app.state.slots = BoundedSemaphore(int(os.getenv("MAX_STREAMS", "16")))
    for _ in app.state.tts.infer_stream("Xin chào.", apply_watermark=False):
        pass
    yield
    job = app.state.benchmark
    if job and job.poll() is None:
        job.terminate()
        job.wait(timeout=10)


app = FastAPI(lifespan=lifespan)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")


class Speech(BaseModel):
    input: str = Field(min_length=1, max_length=20000)
    voice: str | None = None
    response_format: Literal["pcm"] = "pcm"
    sample_rate: Literal[48000] = 48000
    stream_format: Literal["audio"] = "audio"


@app.post("/v1/audio/speech")
def speech(req: Speech, x_request_id: str | None = Header(None)):
    if req.voice and app.state.tts.resolve_voice_name(req.voice) is None:
        raise HTTPException(400, "Unknown preset voice")

    trace = Trace(x_request_id)

    def audio():
        try:
            with app.state.slots:
                for chunk in app.state.tts.infer_stream(req.input, voice=req.voice, apply_watermark=False):
                    trace.chunk(len(chunk))
                    yield (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
        finally:
            trace.finish()

    return StreamingResponse(audio(), media_type="audio/pcm", headers={"X-Request-Id": trace.rid})


@app.get("/health")
def health():
    scheduler = app.state.tts._get_stream_scheduler()
    if scheduler._error:
        raise HTTPException(503, str(scheduler._error))
    return {"status": "ok", "backend": "pytorch", "gpu": torch.cuda.get_device_name(), "sample_rate": 48000,
            "max_streams": scheduler.B, "decode_frames": scheduler.K, "leadin_frames": scheduler.leadin,
            "watermark": False, "cuda": torch.version.cuda,
            "versions": {p: version(p) for p in ("vieneu", "torch", "torchaudio", "transformers", "sea-g2p")}}


@app.post("/benchmark")
async def benchmark(concurrency: int = Query(8, ge=1, le=32),
                    repeats: int = Query(1, ge=1, le=100), seed: int = 42):
    job = app.state.benchmark
    if job and job.poll() is None:
        raise HTTPException(409, "Benchmark already running")
    name = f"bench-{concurrency}.log"
    base = f"http://127.0.0.1:{8000}"
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
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
