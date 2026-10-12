"""
Client mẫu cho API streaming chuẩn OpenAI của VieNeu-TTS (apps/openai_speech.py).

    uv run python -m apps.openai_speech                       # bật server (cổng 8000)
    uv run python examples/openai_speech_client.py            # 1 request, ghi out_stream.wav, in TTFA/RTF
    uv run python examples/openai_speech_client.py --bench 8  # 8 client cùng lúc: TTFA/RTF từng luồng
    uv run python examples/openai_speech_client.py --sse      # nhận dạng Server-Sent Events

Chỉ cần `requests`; nếu có `openai` SDK thì `--sdk` dùng đúng client của OpenAI.
"""
import argparse
import hashlib
import os
import socket
import base64
import json
import statistics
import threading
import time
import wave
import uuid
from pathlib import Path

import requests

SR = 48_000


def server_url(base):
    """The server this CLI talks to is the user's own choice (``--base``); only the
    shape is checked: http(s) with a host, no path/query, trailing slash dropped."""
    from urllib.parse import urlsplit
    u = urlsplit(base)
    if u.scheme not in ("http", "https") or not u.hostname or u.path.strip("/") or u.query or u.fragment:
        raise SystemExit(f"--base phải có dạng http://host[:port] hoặc https://host[:port], nhận: {base!r}")
    return f"{u.scheme}://{u.netloc}"


def check_server(base, api_key=None):
    """Fail fast with a readable message when the server is not up (or not ready)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(f"{base}/health", headers=headers, timeout=5)  # NOSONAR — --base là server do người dùng chọn (CLI)
    except requests.ConnectionError:
        raise SystemExit(
            f"Không kết nối được {base} — server chưa chạy?\n"
            f"  Bật ở cửa sổ khác:  uv run python -m apps.openai_speech\n"
            f"  rồi đợi dòng '✅ ready ...' (cổng khác thì thêm --base http://127.0.0.1:<port>)."
        )
    if r.status_code == 401:
        raise SystemExit("Server yêu cầu API key: thêm --api-key <VIENEU_API_KEY>")
    r.raise_for_status()
    h = r.json()
    print(f"server: backend={h.get('backend')} max_streams={h.get('max_streams')} active={h.get('active')}")
    return h


def stream_pcm(base, text, voice=None, sample_rate=SR, api_key=None, request_id=None):
    """POST /v1/audio/speech (pcm) → (ttfa_s, total_s, pcm_bytes). RTF = total / audio."""
    body = {"model": "vieneu-v3-turbo", "input": text, "voice": voice,
            "response_format": "pcm", "sample_rate": sample_rate}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if request_id:
        headers["X-Request-Id"] = request_id
    t0 = time.perf_counter()
    first = None
    buf = bytearray()
    with requests.post(f"{base}/v1/audio/speech", json=body, headers=headers, stream=True, timeout=120) as r:  # NOSONAR — xem server_url()
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        for chunk in r.iter_content(chunk_size=None):
            if not chunk:
                continue
            if first is None:
                first = time.perf_counter() - t0
            buf += chunk
    return first, time.perf_counter() - t0, bytes(buf)


def stream_sse(base, text, voice=None, api_key=None):
    body = {"model": "vieneu-v3-turbo", "input": text, "voice": voice,
            "response_format": "pcm", "stream_format": "sse"}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    first = None
    buf = bytearray()
    with requests.post(f"{base}/v1/audio/speech", json=body, headers=headers, stream=True, timeout=120) as r:  # NOSONAR — xem server_url()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "speech.audio.delta":
                if first is None:
                    first = time.perf_counter() - t0
                buf += base64.b64decode(ev["audio"])
            elif ev["type"] == "speech.audio.done":
                print("done:", ev.get("usage"))
    return first, time.perf_counter() - t0, bytes(buf)


def stream_sdk(base, text, voice=None, api_key="x"):
    from openai import OpenAI   # pip install openai
    client = OpenAI(base_url=f"{base}/v1", api_key=api_key or "x")
    t0 = time.perf_counter()
    first = None
    buf = bytearray()
    with client.audio.speech.with_streaming_response.create(
        model="vieneu-v3-turbo", voice=voice or "Mai Anh", input=text, response_format="pcm",
    ) as r:
        for chunk in r.iter_bytes(4096):
            if first is None:
                first = time.perf_counter() - t0
            buf += chunk
    return first, time.perf_counter() - t0, bytes(buf)


def save_wav(path, pcm, sample_rate=SR):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def report(tag, first, total, pcm, sample_rate=SR):
    audio_s = len(pcm) / 2 / sample_rate
    print(f"{tag}: TTFA {first*1000:.0f} ms | tổng {total:.2f}s | audio {audio_s:.2f}s | RTF {total/audio_s:.2f}")



def repeated_benchmark(a):
    """Keep upstream PCM/timing path; add raw samples, warmup and GPU evidence."""
    from bench_support import gpu_snapshot
    if a.bench < 1 or a.repeats < 1:
        raise SystemExit("bench/repeats must be positive")
    out = Path(a.output)
    snapshots, idle = [], 0
    for _ in range(120):
        snap = gpu_snapshot(); snapshots.append(snap)
        idle = idle + 1 if float(snap['utilization.gpu']) <= 5 else 0
        if idle >= 5:
            break
        time.sleep(1)
    meta = dict(base=a.base, concurrency=a.bench, repeats=a.repeats, seed=a.seed,
                idle_pass=idle >= 5, before=snapshots, warmup_bursts=2,
                hostname=socket.gethostname(), omp_num_threads=os.getenv('OMP_NUM_THREADS'),
                source_sha256={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*Path(__file__).parent.glob('*.py'),
                                         *Path(__file__).parent.glob('vieneu_lite/*.py')]})
    out.with_suffix('.gpu.json').write_text(json.dumps(meta, indent=2))
    if idle < 5:
        raise RuntimeError('GPU not idle before benchmark')
    stop = threading.Event()
    during = []
    def monitor():
        while not stop.is_set():
            try:
                during.append(gpu_snapshot())
            except Exception as e:
                during.append({'error': str(e)})
            stop.wait(1)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with out.open('w') as f:
            for repeat in range(-2, a.repeats):
                seed = a.seed + max(repeat, 0)
                requests.post(f'{a.base}/benchmark/seed', params={'seed': seed}, timeout=30).raise_for_status()
                rows = [None] * a.bench
                barrier = threading.Barrier(a.bench)
                def one(i):
                    rid = uuid.uuid4().hex
                    row = dict(request_id=rid, concurrency=a.bench, repeat=repeat + 1,
                               index=i, seed=seed)
                    try:
                        barrier.wait(timeout=30)
                        first, total, pcm = stream_pcm(a.base, a.text, a.voice, a.sample_rate, a.api_key, rid)
                        audio_s = len(pcm) / 2 / a.sample_rate
                        if first is None or audio_s <= 0:
                            raise RuntimeError('empty audio')
                        row.update(ttfa_s=first, total_s=total, audio_s=audio_s, rtf=total/audio_s, ok=True)
                    except Exception as e:
                        row.update(ok=False, error=str(e))
                    rows[i] = row
                ths = [threading.Thread(target=one, args=(i,)) for i in range(a.bench)]
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
                if repeat >= 0:
                    for row in rows:
                        f.write(json.dumps(row) + '\n')
                    f.flush()
                if not all(row['ok'] for row in rows):
                    raise RuntimeError(rows)
                print(f"{'warmup' if repeat < 0 else 'measured'} c={a.bench} repeat={repeat+1} "
                      f"ok={len(rows)} TTFA={statistics.median(r['ttfa_s'] for r in rows)*1000:.1f}ms", flush=True)
    finally:
        stop.set(); watcher.join(timeout=10)
        meta['during'] = during
        meta['after'] = gpu_snapshot()
        out.with_suffix('.gpu.json').write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--text", default="Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.")
    ap.add_argument("--voice", default=None, help="tên preset (GET /v1/voices); bỏ trống = giọng mặc định")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sample-rate", type=int, default=SR, choices=[48000, 24000, 16000, 8000])
    ap.add_argument("--bench", type=int, default=0, help="N client đồng thời")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", help="per-request JSONL; enables controlled repeated benchmark")
    ap.add_argument("--sse", action="store_true")
    ap.add_argument("--sdk", action="store_true")
    a = ap.parse_args()
    a.base = server_url(a.base)
    check_server(a.base, a.api_key)

    if a.bench and a.output:
        repeated_benchmark(a)
        return
    if a.bench:
        res = [None] * a.bench
        def one(i):
            try:
                res[i] = stream_pcm(a.base, a.text, a.voice, a.sample_rate, a.api_key)
            except Exception as e:   # noqa: BLE001 — một client lỗi không được làm hỏng cả bench
                res[i] = e
        ths = [threading.Thread(target=one, args=(i,)) for i in range(a.bench)]
        t = time.perf_counter()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        wall = time.perf_counter() - t
        ok = [r for r in res if not isinstance(r, Exception)]
        bad = [r for r in res if isinstance(r, Exception)]
        for e in bad[:3]:
            print(f"  lỗi: {e}")
        if not ok:
            raise SystemExit(f"{len(bad)}/{a.bench} client lỗi, không có kết quả.")
        ttfa = [r[0] for r in ok]
        rtf = [r[1] / (len(r[2]) / 2 / a.sample_rate) for r in ok]
        print(f"{a.bench} client ({len(ok)} ok, {len(bad)} lỗi/429): TTFA median {statistics.median(ttfa)*1000:.0f} ms, "
              f"max {max(ttfa)*1000:.0f} ms | RTF max {max(rtf):.2f} (cần < 1) | {wall:.1f}s tổng")
        return
    if a.sse:
        first, total, pcm = stream_sse(a.base, a.text, a.voice, a.api_key)
    elif a.sdk:
        first, total, pcm = stream_sdk(a.base, a.text, a.voice, a.api_key)
    else:
        first, total, pcm = stream_pcm(a.base, a.text, a.voice, a.sample_rate, a.api_key)
    report("1 request", first, total, pcm, a.sample_rate)
    save_wav("out_stream.wav", pcm, a.sample_rate)
    print("→ out_stream.wav")


if __name__ == "__main__":
    main()
