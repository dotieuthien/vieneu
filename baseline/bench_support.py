"""Optional benchmark diagnostics; enabled only with BENCH_MODE=1."""
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time
import uuid

RESULTS = Path(__file__).resolve().parent / 'results'
RESULTS.mkdir(exist_ok=True)
TRACE_LOCK = threading.Lock()
GPU_FIELDS = 'timestamp,uuid,name,driver_version,pstate,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.current.memory'


def gpu_snapshot():
    p = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=' + GPU_FIELDS,
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return dict(zip(GPU_FIELDS.split(','), [x.strip() for x in p.stdout.strip().split(',')]))


def prepare_gpu():
    if os.getenv('BENCH_MODE') != '1':
        return
    observations, idle = [], 0
    for _ in range(120):
        snap = gpu_snapshot(); observations.append(snap)
        idle = idle + 1 if float(snap['utilization.gpu']) <= 5 else 0
        if idle >= 5:
            break
        time.sleep(1)
    record = {'before_load': observations, 'idle_pass': idle >= 5}
    (RESULTS / 'gpu-preflight.json').write_text(json.dumps(record, indent=2))
    if idle < 5:
        raise RuntimeError('GPU did not become idle for 5 consecutive samples; benchmark aborted')
    lock = subprocess.run(['nvidia-smi', '-i', '0', '-lgc', '1500,1500'], capture_output=True, text=True)
    record['clock_lock'] = {'requested_mhz': 1500, 'returncode': lock.returncode,
                            'stdout': lock.stdout, 'stderr': lock.stderr}
    record['after_lock'] = gpu_snapshot()
    (RESULTS / 'gpu-preflight.json').write_text(json.dumps(record, indent=2))
    print('BENCH GPU:', json.dumps(record), flush=True)


def reset_seed(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Trace:
    def __init__(self, request_id=None):
        self.enabled = os.getenv('BENCH_MODE') == '1'
        self.rid = request_id or uuid.uuid4().hex
        self.start = time.perf_counter()
        self.first = None
        self.samples = 0

    def chunk(self, samples):
        if self.first is None:
            self.first = time.perf_counter() - self.start
        self.samples += samples

    def finish(self, **extra):
        if not self.enabled:
            return
        row = dict(request_id=self.rid, server_ttfa_s=self.first,
                   server_total_s=time.perf_counter() - self.start,
                   audio_s=self.samples / 48000, **extra)
        with TRACE_LOCK, (RESULTS / 'server-requests.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
