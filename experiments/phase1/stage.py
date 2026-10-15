"""Rebuild the exact code of a phase 1 experiment, one VKong project folder per variant.

    python3 experiments/phase1/stage.py abba /tmp/p1   # -> /tmp/p1/abba/{sdk,lite}

Code comes from commit 1ef70a3. The ABBA lite ran one step earlier, before the
OVERLAP_CODEC option existed; abba-lite.patch takes that option back out.
Every .py file is then hashed and compared with the sha256 that benchmark.py
recorded on the GPU host during the original run (``source_sha256`` in
``<side>/results/<exp>/block-*/bench-1.gpu.json``), so a match means the
folder is byte-for-byte the code that produced the numbers.

All variants share one App name: ``vkong run`` rents a machine with the first
variant, then ``vkong attach <id> --update -C <variant>`` swaps the code on that
same machine (step by step in PHASE1.md).
"""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REF = "1ef70a3"
APP = "vieneu-phase1"
COMMON = "BENCH_MODE=1 MAX_STREAMS=16"
TAIL = "WATERMARK=0 OMP_NUM_THREADS=4 python -u serve.py"

# variant -> (side in the repo, patch, extra env, block whose bench-1.gpu.json holds the hashes)
EXPERIMENTS = {
    "abba": {
        "sdk": ("baseline", None, "", "baseline/results/abba-20261013/block-1-slots-16"),
        "lite": ("lite", "abba-lite.patch", "", "lite/results/abba-20261013/block-2-slots-16"),
    },
}

VKONG_YAML = """\
type: service
# Same App for every variant: `vkong attach <id> --update -C <variant>` swaps code on one machine.
app: {app}
app_port: 8000

# Phase 1 asked for "RTX 3060"; pin the 12 GB card so reruns compare like with like.
gpu: RTX-3060-12GB
num_gpus: 1
cpu_cores: 4
ram_gb: 16
disk_gb: 40
max_dph: 0.35
verified_only: true

image: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
init_cmd: pip install -r requirements.txt
start: env {env} {tail}
"""
VKONGIGNORE = ".git/\n.venv/\n.vkong/\n__pycache__/\nresults/\n"


def sources(folder: Path) -> dict:
    """The same file set benchmark.py hashes: *.py and vieneu_lite/*.py."""
    files = [*folder.glob("*.py"), *folder.glob("vieneu_lite/*.py")]
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def stage(experiment: str, out: Path) -> bool:
    root = out / experiment
    if root.exists():
        shutil.rmtree(root)
    all_ok = True
    for variant, (side, patch, env, block) in EXPERIMENTS[experiment].items():
        dst = root / variant
        tmp = root / f".{variant}"
        tar = subprocess.run(["git", "-C", str(REPO), "archive", REF, side], check=True, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp)
        if patch:
            subprocess.run(["patch", "-s", "-p1", "-d", str(tmp), "-i", str(HERE / patch)], check=True)
        shutil.copytree(tmp / side, dst, ignore=shutil.ignore_patterns("results", "vkong.yaml", "README.md", ".gitignore"))
        shutil.rmtree(tmp)
        (dst / "vkong.yaml").write_text(VKONG_YAML.format(app=APP, env=" ".join(filter(None, [COMMON, env])), tail=TAIL))
        (dst / ".vkongignore").write_text(VKONGIGNORE)

        want = json.loads((REPO / block / "bench-1.gpu.json").read_text())["source_sha256"]
        got = sources(dst)
        bad = [f"{name}: {'missing' if name not in got else 'not in the original run' if name not in want else 'different'}"
               for name in sorted(set(want) | set(got)) if want.get(name) != got.get(name)]
        all_ok &= not bad
        print(f"{dst}: {len(got)} files, " + ("same as the code that was measured" if not bad else "DIFFERENT"))
        for line in bad:
            print(f"    {line}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=EXPERIMENTS)
    ap.add_argument("out", type=Path, help="parent folder for the staged variants")
    a = ap.parse_args()
    ok = stage(a.experiment, a.out.resolve())
    print("all sources match the original run" if ok else "SOURCES DO NOT MATCH the original run")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
