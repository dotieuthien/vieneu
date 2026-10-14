"""Opt-in profiler for the serving path (``PROFILE=1``, read at import).

Live, a few µs per region:

  region     host time on the calling thread (nested regions excluded) and, with
             ``gpu=True``, the device span between two CUDA events on the stream
             that ran it. host >> gpu: CPU/launch-bound; gpu >> host: GPU-bound.
  tick.*     ``tick.period``: one graph replay to the next, i.e. one frame for
             every slot (real time needs <= 80 ms). ``tick.gap``: AR stream time
             between two replays, when the frame graph is not running.
  req.*      per text chunk, the server-side time to first audio split in stages.
  gil.wait   oversleep of a 2 ms sleep loop: how long a thread waits for the GIL.

On demand:

  frame_anatomy  at startup: the frame body run eagerly (``scope``s mark its
                 sub-stages) and as the graph: kernels per frame, which sub-stage
                 owns the GPU time, what the graph saves in launch overhead.
  trace          torch.profiler (CPU ops + CUPTI) over every thread for N seconds:
                 a Perfetto trace, plus per region kernel launches, launch-API
                 time, sync time and kernel time.
"""
from __future__ import annotations

import collections
import contextlib
import functools
import logging
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

ENABLED = os.getenv("PROFILE") == "1"
if ENABLED:
    # Kineto tears CUPTI down after each trace; with a CUDA graph alive the next
    # trace can crash. torch sets this only for inductor graphs, not hand-captured ones.
    os.environ.setdefault("TEARDOWN_CUPTI", "0")

_CUDA = torch.cuda.is_available()
_NULL = contextlib.nullcontext()

# Report order, grouped by the thread that runs them.
ABOUT = {
    "text.split": "split the text into chunks",
    "text.phonemize": "phonemize one chunk",
    "submit.prompt": "tokenize one chunk's prompt",
    "watermark": "watermark one audio chunk",
    "worker.idle": "worker waiting for requests",
    "admit": "newcomers: input embeddings, slicing the prefill KV",
    "admit.prefill": "HF prefill of the newcomers (eager)",
    "admit.load": "copy one prompt's KV into the ring cache (eager)",
    "step": "per-slot bookkeeping after each frame (Python)",
    "step.replay": "launch the frame graph; gpu = the frame",
    "step.status": "wait for the frame, read finished/length (sync)",
    "decode": "copy the new codes out of the ring, build the codec job",
    "decode.backpressure": "worker blocked: codec queue full",
    "tick.period": "one frame for every slot (real time: <= 80 ms)",
    "tick.gap": "AR stream not running the graph between two frames",
    "codec.idle": "codec waiting for a job",
    "codec": "codec job bookkeeping, splitting audio per row",
    "codec.decode": "streaming codec over every row (eager)",
    "codec.d2h": "audio to host (sync)",
    "req.queue": "submit -> taken by the worker",
    "req.admit": "taken -> in a slot (shared prefill)",
    "req.leadin": "in a slot -> first codec job packed",
    "req.codec_queue": "job packed -> codec starts it",
    "req.codec": "codec starts -> first audio out",
    "req.deliver": "first audio out -> read by the HTTP thread",
    "req.first_audio": "submit -> first audio read (no text work)",
    "req.total": "submit -> chunk done",
    "gil.wait": "a woken thread waiting for the GIL (+ ~0.1 ms timer slack)",
}
GROUPS = [
    ("request threads", ("text.", "submit.", "watermark")),
    ("worker", ("worker.", "admit", "step", "decode", "tick.")),
    ("codec", ("codec",)),
    ("per text chunk", ("req.",)),
    ("process", ("gil.",)),
]
STAGES = [("req.queue", "submit", "taken"), ("req.admit", "taken", "admitted"),
          ("req.leadin", "admitted", "job"), ("req.codec_queue", "job", "codec"),
          ("req.codec", "codec", "audio"), ("req.deliver", "audio", "consumed"),
          ("req.first_audio", "submit", "consumed"), ("req.total", "submit", "done")]

LAUNCH_API = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx",
              "cudaMemsetAsync"}
GRAPH_API = {"cudaGraphLaunch"}
# A copy to pageable host memory (``.cpu()``, ``.tolist()``) blocks inside
# cudaMemcpyAsync until the GPU gets there, so copies count as waiting, not launching.
SYNC_API = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy",
            "cudaMemcpyAsync"}


class TraceBusy(RuntimeError):
    pass


def _event() -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


class Series:
    """Count, total and the last ``keep`` samples (ms) of one metric."""

    def __init__(self, keep: int = 4096):
        self.n, self.total = 0, 0.0
        self.recent = collections.deque(maxlen=keep)

    def add(self, ms: float) -> None:
        self.n += 1
        self.total += ms
        self.recent.append(ms)

    def summary(self) -> Optional[dict]:
        if not self.n:
            return None
        xs = sorted(self.recent)
        q = lambda p: xs[max(0, math.ceil(p * len(xs)) - 1)]   # noqa: E731
        return {"n": self.n, "total_ms": self.total, "mean": self.total / self.n,
                "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}


class _Stat:
    def __init__(self):
        self.host, self.gpu, self.incl_ms = Series(), Series(), 0.0


class _Region:
    __slots__ = ("p", "name", "gpu", "chain", "t0", "ev0", "rf", "child")

    def __init__(self, p: "Profiler", name: str, gpu: bool, chain: Optional[str]):
        self.p, self.name, self.gpu, self.chain = p, name, gpu and _CUDA, chain

    def __enter__(self):
        self.rf = torch.profiler.record_function(self.name) if self.p.tracing else None
        if self.rf is not None:
            self.rf.__enter__()
        self.ev0 = _event() if self.gpu else None
        self.child = 0.0
        self.p._stack().append(self)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        t1 = time.perf_counter()
        ev1 = _event() if self.gpu else None
        if self.rf is not None:
            self.rf.__exit__(*exc)
        self.p._close(self, t1, ev1)
        return False


class Profiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.tracing = False                 # regions/scopes also emit record_function
        self.anatomy: Optional[dict] = None
        self.last_trace: Optional[dict] = None
        self._stats = collections.defaultdict(_Stat)
        self._lock = threading.Lock()
        self._resolving = threading.Lock()
        self._trace_lock = threading.Lock()
        self._local = threading.local()
        self._pending = collections.deque()  # (name, start event, end event)
        self._chains = {}                    # chain -> (host t0, end event) of the last region
        self._t0 = time.perf_counter()
        if enabled:
            threading.Thread(target=self._probe_gil, name="profiler-gil", daemon=True).start()

    # ── instrumentation API (all no-ops when disabled) ──────────────────────

    def region(self, name: str, gpu: bool = False, chain: Optional[str] = None):
        """Time a block; ``chain`` also records ``<chain>.period`` / ``<chain>.gap``
        between consecutive regions of that chain."""
        return _Region(self, name, gpu, chain) if self.enabled else _NULL

    def timed(self, name: str, gpu: bool = False, chain: Optional[str] = None):
        """Decorator form of ``region``; returns the function untouched when disabled."""
        def wrap(fn):
            if not self.enabled:
                return fn

            @functools.wraps(fn)
            def inner(*args, **kwargs):
                with _Region(self, name, gpu, chain):
                    return fn(*args, **kwargs)
            return inner
        return wrap

    def scope(self, name: str):
        """Trace-only label, no timing: safe inside the CUDA graph body (only the
        eager anatomy run ever records it)."""
        return torch.profiler.record_function(name) if self.tracing else _NULL

    def scoped(self, name: str):
        def wrap(fn):
            if not self.enabled:
                return fn

            @functools.wraps(fn)
            def inner(*args, **kwargs):
                with self.scope(name):
                    return fn(*args, **kwargs)
            return inner
        return wrap

    def unchain(self, chain: str) -> None:
        """The next region of ``chain`` follows an idle wait: don't count the wait."""
        if self.enabled:
            self._chains.pop(chain, None)

    def mark(self, req, stage: str) -> None:
        """First time ``req`` reaches ``stage`` (see ``STAGES``)."""
        if self.enabled and stage not in req.marks:
            req.marks[stage] = time.perf_counter()

    def request_done(self, req) -> None:
        if not self.enabled:
            return
        self.mark(req, "done")
        m = req.marks
        with self._lock:
            for name, a, b in STAGES:
                if a in m and b in m:
                    self._stats[name].host.add((m[b] - m[a]) * 1e3)

    # ── bookkeeping ─────────────────────────────────────────────────────────

    def _stack(self) -> list:
        s = getattr(self._local, "stack", None)
        if s is None:
            s = self._local.stack = []
        return s

    def _close(self, r: _Region, t1: float, ev1) -> None:
        incl = (t1 - r.t0) * 1e3
        stack = self._stack()
        stack.pop()
        if stack:
            stack[-1].child += incl
        with self._lock:
            s = self._stats[r.name]
            s.host.add(incl - r.child)
            s.incl_ms += incl
            if ev1 is not None:
                self._pending.append((r.name, r.ev0, ev1))
            if r.chain:
                last = self._chains.get(r.chain)
                if last is not None:
                    self._stats[r.chain + ".period"].host.add((r.t0 - last[0]) * 1e3)
                    if last[1] is not None and r.ev0 is not None:
                        self._pending.append((r.chain + ".gap", last[1], r.ev0))
                self._chains[r.chain] = (r.t0, ev1)
        if len(self._pending) > 64:
            self._resolve(wait=False)

    def _resolve(self, wait: bool) -> None:
        """Turn finished CUDA event pairs into gpu samples, oldest first."""
        if not self._resolving.acquire(blocking=wait):
            return
        try:
            while self._pending:
                name, a, b = self._pending[0]
                if wait:
                    b.synchronize()
                elif not b.query():
                    return
                self._pending.popleft()
                ms = a.elapsed_time(b)
                with self._lock:
                    self._stats[name].gpu.add(ms)
        finally:
            self._resolving.release()

    def _probe_gil(self) -> None:
        while True:
            t = time.perf_counter()
            time.sleep(0.002)
            late = (time.perf_counter() - t) * 1e3 - 2.0
            with self._lock:
                self._stats["gil.wait"].host.add(max(0.0, late))

    # ── reading ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Start a new window (anatomy and the last trace are kept)."""
        self._resolve(wait=True)
        with self._lock:
            self._stats.clear()
            self._t0 = time.perf_counter()

    def snapshot(self) -> dict:
        self._resolve(wait=True)
        with self._lock:
            stats = {n: {"host": s.host.summary(), "gpu": s.gpu.summary(), "incl_ms": s.incl_ms}
                     for n, s in self._stats.items()}
            window = time.perf_counter() - self._t0
        return {"window_s": window, "stats": stats, "anatomy": self.anatomy, "trace": self.last_trace}

    def report(self, snap: Optional[dict] = None) -> str:
        snap = snap or self.snapshot()
        win, st = snap["window_s"], snap["stats"]
        out = [f"window {win:.1f} s | host = wall time on its thread, nested regions excluded; "
               f"gpu = CUDA-event span (incl. nested). ms unless noted.", ""]
        tick = (st.get("tick.period") or {}).get("host")
        if tick:
            out += [f"tick p50 {tick['p50']:.1f} ms -> {80 / tick['p50']:.2f}x real time "
                    f"for every active stream ({tick['n']} frames)", ""]
        out.append(f"{'region':<20}{'calls':>7}{'host p50':>10}{'p99':>10}{'total s':>9}{'%win':>6} |"
                   f"{'gpu p50':>10}{'p99':>10}  what")
        order = list(ABOUT) + sorted(n for n in st if n not in ABOUT)
        for title, prefixes in GROUPS + [("other", ("",))]:
            rows = [n for n in order if n in st and n.startswith(prefixes)]
            order = [n for n in order if n not in rows]
            if rows:
                out.append(f"-- {title}")
                out += [_row(n, st[n], win) for n in rows]
        for extra in (snap.get("anatomy"), snap.get("trace")):
            if extra:
                out += ["", extra["text"]]
        return "\n".join(out)

    # ── torch.profiler ──────────────────────────────────────────────────────

    def trace(self, seconds: float, out_dir: Path) -> dict:
        """Record every thread for ``seconds``; save a Perfetto trace and summarise it."""
        if not self._trace_lock.acquire(blocking=False):
            raise TraceBusy("a trace is already running")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            prof = _kineto()
            prof.start()
            t = time.perf_counter()
            self.tracing = True
            try:
                time.sleep(seconds)
            finally:
                self.tracing = False
                wall = time.perf_counter() - t
                prof.stop()
            path = out_dir / f"trace-{time.strftime('%Y%m%d-%H%M%S')}.json.gz"
            prof.export_chrome_trace(str(path))
            s = summarise(prof.events(), wall)
            s["file"] = path.name
            s["text"] = _trace_text(s)
            self.last_trace = s
            return s
        finally:
            self._trace_lock.release()

    @torch.no_grad()
    def frame_anatomy(self, frame, iters: int = 5) -> Optional[dict]:
        """Run the frame body eagerly and as the graph (dirty state; resets the frame).
        A failure is logged, never fatal: profiling must not stop the server."""
        try:
            return self._frame_anatomy(frame, iters)
        except Exception:   # noqa: BLE001
            log.warning("frame anatomy failed", exc_info=True)
            return None

    def _frame_anatomy(self, frame, iters: int) -> dict:
        sync = torch.cuda.synchronize

        def timed(fn):
            host, gpu = [], []
            for _ in range(iters):
                sync()
                a, t = _event(), time.perf_counter()
                fn()
                host.append((time.perf_counter() - t) * 1e3)
                b = _event()
                b.synchronize()
                gpu.append(a.elapsed_time(b))
            return statistics.median(host), statistics.median(gpu)

        try:
            frame._body()
            sync()
            eager_host, eager_gpu = timed(frame._body)
            graph_host, graph_gpu = timed(frame.graph.replay)     # not frame.replay: no tick stats
            prof = _kineto()
            prof.start()
            self.tracing = True
            try:
                with torch.profiler.record_function("frame.eager"):
                    frame._body()
                sync()
                with torch.profiler.record_function("frame.graph"):
                    frame.graph.replay()
                sync()
            finally:
                self.tracing = False
                prof.stop()
            regions = summarise(prof.events(), 0.0)["regions"]
        finally:
            frame.reset()
        a = {"slots": frame.B, "eager_host_ms": eager_host, "eager_gpu_ms": eager_gpu,
             "graph_host_ms": graph_host, "graph_gpu_ms": graph_gpu, "regions": regions}
        a["text"] = _anatomy_text(a)
        self.anatomy = a
        return a


def _kineto():
    from torch._C._profiler import _ExperimentalConfig
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if _CUDA else [])
    # Without profile_all_threads only the calling thread's ops are recorded,
    # and the worker / codec / request threads would be invisible.
    return profile(activities=acts, experimental_config=_ExperimentalConfig(profile_all_threads=True))


def summarise(events, wall_s: float) -> dict:
    """Per user annotation (region/scope), excluding nested annotations: host time,
    launch-API calls and time, blocking syncs, and the GPU ops it launched. Plus the
    GPU timeline: busy share, idle gaps, per stream, top kernels."""
    from torch.autograd import DeviceType
    dev = [e for e in events if e.device_type == DeviceType.CUDA and not e.is_user_annotation]
    # Kernels inside a graph replay are not linked to any CPU op; they share the
    # cudaGraphLaunch call's CUPTI correlation id instead.
    by_corr = collections.defaultdict(list)
    for d in dev:
        by_corr[d.id].append(d)
    anns = [e for e in events if e.device_type == DeviceType.CPU and e.is_user_annotation]
    # cudaGraphLaunch has no CPU-op parent, so torch files it under the OS thread id
    # and it never nests under a region: give it to the shortest region around it.
    graphs = collections.defaultdict(list)
    for g in events:
        if g.device_type == DeviceType.CPU and g.name in GRAPH_API:
            around = [a for a in anns if a.time_range.start <= g.time_range.start
                      and g.time_range.end <= a.time_range.end]
            if around:
                graphs[id(min(around, key=lambda a: a.time_range.elapsed_us()))].append(g)
    regions = {}
    for e in anns:
        r = regions.setdefault(e.name, collections.Counter())
        r["calls"] += 1
        for g in graphs.get(id(e), []):
            r["graphs"] += 1
            r["launch_us"] += g.cpu_time_total
            nodes = by_corr.get(g.id, [])
            if nodes:
                r["ops"] += len(nodes)
                r["kernel_us"] += sum(d.time_range.elapsed_us() for d in nodes)
                r["graph_span_us"] += (max(d.time_range.end for d in nodes)
                                       - min(d.time_range.start for d in nodes))
        host, todo = e.cpu_time_total, [e]
        while todo:
            x = todo.pop()
            r["ops"] += len(x.kernels)
            r["kernel_us"] += sum(k.duration for k in x.kernels)
            for c in x.cpu_children:
                if c.is_user_annotation:       # nested region: its own row
                    host -= c.cpu_time_total
                    continue
                if c.name in LAUNCH_API:
                    r["launches"] += 1
                    r["launch_us"] += c.cpu_time_total
                elif c.name in SYNC_API:
                    r["sync_us"] += c.cpu_time_total
                todo.append(c)
        r["host_us"] += host
    per_call = {}
    for name, r in regions.items():
        n = r["calls"]
        d = {"calls": n, "host_ms": r["host_us"] / n / 1e3, "launches": r["launches"] / n,
             "graphs": r["graphs"] / n, "launch_ms": r["launch_us"] / n / 1e3,
             "sync_ms": r["sync_us"] / n / 1e3, "gpu_ops": r["ops"] / n,
             "kernel_ms": r["kernel_us"] / n / 1e3,
             "us_per_op": r["kernel_us"] / r["ops"] if r["ops"] else 0.0,
             "graph_span_ms": r["graph_span_us"] / n / 1e3}
        d["verdict"] = _verdict(d)
        per_call[name] = d

    busy, gaps = _union([(e.time_range.start, e.time_range.end) for e in dev])
    streams = collections.defaultdict(list)
    names = collections.defaultdict(lambda: [0, 0.0])
    for e in dev:
        streams[e.device_resource_id].append(e)
        k = names[e.name]
        k[0] += 1
        k[1] += e.time_range.elapsed_us()
    per_stream = []
    for sid, evs in sorted(streams.items()):
        sb, _ = _union([(e.time_range.start, e.time_range.end) for e in evs])
        per_stream.append({"stream": sid, "ops": len(evs), "busy_ms": sb / 1e3})
    total_k = sum(v[1] for v in names.values()) or 1.0
    top = sorted(names.items(), key=lambda kv: -kv[1][1])[:15]
    return {"window_s": wall_s, "gpu_busy_ms": busy / 1e3,
            "gaps_over_100us": sum(g > 100 for g in gaps),
            "gap_ms_over_100us": sum(g for g in gaps if g > 100) / 1e3,
            "streams": per_stream, "regions": per_call,
            "top_kernels": [{"name": n, "count": c, "total_ms": t / 1e3, "share": t / total_k,
                             "us_each": t / c} for n, (c, t) in top]}


def _union(intervals):
    """Busy time of a set of intervals and the idle gaps between them (µs)."""
    busy, gaps, cur = 0.0, [], None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur is not None:
                busy += cur[1] - cur[0]
                gaps.append(s - cur[1])
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        busy += cur[1] - cur[0]
    return busy, gaps


def _verdict(d: dict) -> str:
    cpu = d["host_ms"] - d["sync_ms"]       # time the host spent issuing work
    if d["sync_ms"] > 0.5 * d["host_ms"] and d["sync_ms"] > 0.05:
        return "waits on GPU"
    if not d["gpu_ops"]:
        return "CPU only"
    if d["kernel_ms"] < 0.5 * cpu:
        return "launch/CPU-bound"
    if d["kernel_ms"] > 2 * cpu:
        return "GPU-bound"
    return "balanced"


def _row(name: str, s: dict, win: float) -> str:
    h, g = s["host"], s["gpu"]
    calls = (h or g)["n"]
    host = (f"{h['p50']:10.2f}{h['p99']:10.2f}{h['total_ms'] / 1e3:9.2f}{h['total_ms'] / 10 / win:6.1f}"
            if h else " " * 35)
    gpu = f"{g['p50']:10.2f}{g['p99']:10.2f}" if g else " " * 20
    return f"{name:<20}{calls:>7}{host} |{gpu}  {ABOUT.get(name, '')}"


def _region_table(regions: dict, order) -> list:
    out = [f"{'region':<20}{'calls':>6}{'host':>9}{'launch':>8}{'graph':>6}{'launch ms':>10}"
           f"{'sync/copy':>10}{'gpu ops':>8}{'kernel ms':>10}{'us/op':>7}  verdict"]
    for n in order:
        d = regions[n]
        out.append(f"{n:<20}{d['calls']:>6}{d['host_ms']:9.2f}{d['launches']:8.0f}{d['graphs']:6.0f}"
                   f"{d['launch_ms']:10.2f}{d['sync_ms']:10.2f}{d['gpu_ops']:8.0f}{d['kernel_ms']:10.2f}"
                   f"{d['us_per_op']:7.1f}  {d['verdict']}")
    return out


def _trace_text(s: dict) -> str:
    w = s["window_s"] or 1.0
    out = [f"trace {w:.1f} s -> {s['file']} (open in https://ui.perfetto.dev)",
           "CUPTI adds ~1-2 us to every launch while tracing: compare regions, not absolutes.",
           f"GPU busy {s['gpu_busy_ms'] / 10 / w:.1f}% of the window; {s['gaps_over_100us']} idle gaps "
           f"> 0.1 ms, {s['gap_ms_over_100us']:.0f} ms in total"]
    out += [f"  stream {x['stream']:<6} busy {x['busy_ms'] / 10 / w:5.1f}%  {x['ops']} ops" for x in s["streams"]]
    out += ["", "per region, per call, nested regions excluded (ms; launch = kernel/copy launches):"]
    order = [n for n in ABOUT if n in s["regions"]] + sorted(n for n in s["regions"] if n not in ABOUT)
    out += _region_table(s["regions"], order)
    out += ["", "top GPU ops by time:"]
    out += [f"  {k['share'] * 100:5.1f}% {k['count']:>7}x {k['us_each']:8.1f} us  {k['name'][:100]}"
            for k in s["top_kernels"]]
    return "\n".join(out)


def _anatomy_text(a: dict) -> str:
    r = a["regions"]
    subs = ["frame.eager", "frame.acoustic", "frame.sample", "frame.backbone", "frame.attn"]
    eager_ops = sum(r[n]["gpu_ops"] for n in subs if n in r)
    eager_kern = sum(r[n]["kernel_ms"] for n in subs if n in r) or 1.0
    g = r.get("frame.graph", {})
    out = [f"frame anatomy: one frame for all {a['slots']} slots (same cost whatever the occupancy)",
           f"  eager: {eager_ops:.0f} GPU ops, host {a['eager_host_ms']:.1f} ms to launch them "
           f"({a['eager_host_ms'] * 1e3 / max(eager_ops, 1):.0f} us/op), GPU span {a['eager_gpu_ms']:.1f} ms",
           f"  graph: 1 launch, host {a['graph_host_ms']:.2f} ms, GPU {a['graph_gpu_ms']:.1f} ms"
           + (f"; {g['gpu_ops']:.0f} nodes, kernels {g['kernel_ms']:.1f} ms of a "
              f"{g['graph_span_ms']:.1f} ms span -> {g['graph_span_ms'] - g['kernel_ms']:.1f} ms "
              f"of gaps between graph nodes (traced)"
              if g.get("graph_span_ms") else "; kernels inside the graph not attributed"),
           f"  graph saves {a['eager_gpu_ms'] - a['graph_gpu_ms']:.1f} ms per frame",
           "  sub-stages (eager, nested excluded; acoustic = decoder layers + embeddings + EOS head,"
           " backbone = without attention, eager = bookkeeping + input embeds):"]
    for n in subs:
        if n in r:
            d = r[n]
            out.append(f"    {n:<16}{d['gpu_ops']:7.0f} ops {d['kernel_ms']:8.2f} ms kernels "
                       f"({d['kernel_ms'] / eager_kern * 100:4.1f}%) {d['us_per_op']:6.1f} us/op")
    return "\n".join(out)


PROF = Profiler(ENABLED)
