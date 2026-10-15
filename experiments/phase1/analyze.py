"""Recompute the phase 1 tables in PHASE1.md from the raw per-request files.

    python3 experiments/phase1/analyze.py                       # the committed 2026-10-13 run
    python3 experiments/phase1/analyze.py --abba abba-lan2   # your own rerun (PHASE1.md)

Standard library only, no GPU. Inputs, per side (baseline = SDK, lite):

    <side>/results/<exp>/block-*/bench-<c>.jsonl         one row per timed request (client side)
    <side>/results/<exp>/block-*/server-requests.jsonl   server-side TTFA of the same request IDs

Method, as in phase 1: TTFA and RTF are taken per request, every block of a
variant is pooled, and the median / p90 use linear interpolation at (n-1)*p.
No median of medians. With --check, the committed report.json (ABBA) and
summary.json (overlap) are compared against the recomputed values.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ABBA_LEVELS = [(1, 16), (4, 16), (8, 16), (16, 16), (32, 32)]   # (clients, slots)
OVERLAP_LEVELS = [1, 16]
OVERLAP_VARIANTS = [("baseline", "baseline", "SDK gốc"), ("lite", "off", "Lite, chạy tuần tự"),
                    ("lite", "on", "Lite, codec chạy song song")]


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load(side: str, exp: str, block_glob: str, c: int):
    """Timed requests of every matching block, with server TTFA joined by request ID."""
    rows = []
    for block in sorted((REPO / side / "results" / exp).glob(block_glob)):
        server = {r["request_id"]: r for r in read(block / "server-requests.jsonl")}
        for r in read(block / f"bench-{c}.jsonl"):
            if not r.get("ok"):
                raise SystemExit(f"failed request in {block}/bench-{c}.jsonl: {r}")
            s = server.get(r["request_id"], {})
            rows.append(dict(r, block=block.name, server_ttfa_s=s.get("server_ttfa_s")))
    return rows


def stats(rows):
    ttfa = [r["ttfa_s"] * 1e3 for r in rows]
    rtf = [r["rtf"] for r in rows]
    srv = [r["server_ttfa_s"] * 1e3 for r in rows if r["server_ttfa_s"] is not None]
    return dict(n=len(rows), ttfa50=q(ttfa, .5), ttfa90=q(ttfa, .9), rtf50=q(rtf, .5), rtf90=q(rtf, .9),
                srv50=q(srv, .5) if srv else None, srv90=q(srv, .9) if srv else None, n_srv=len(srv))


def ms(x):
    return "—" if x is None else f"{x:.1f} ms"


def abba(exp: str, check: bool) -> bool:
    print(f"## ABBA: SDK gốc vs lite ({exp})\n")
    sides = {side: {c: stats(load(side, exp, f"block-*-slots-{slots}", c)) for c, slots in ABBA_LEVELS}
             for side in ("baseline", "lite")}
    print("| Người dùng cùng lúc | n mỗi bản | TTFA SDK gốc | TTFA lite | RTF SDK gốc | RTF lite |")
    print("|---:|---:|---:|---:|---:|---:|")
    for c, slots in ABBA_LEVELS:
        b, l = sides["baseline"][c], sides["lite"][c]
        label = f"{c} ({slots} ghế)" if slots != 16 else str(c)
        print(f"| {label} | {b['n']}/{l['n']} | {ms(b['ttfa50'])} | {ms(l['ttfa50'])} | {b['rtf50']:.3f} | {l['rtf50']:.3f} |")

    print("\nTTFA phía server (cùng request, đo trong handler), p50:\n")
    print("| Người dùng | SDK gốc | lite |\n|---:|---:|---:|")
    for c, _ in ABBA_LEVELS:
        print(f"| {c} | {ms(sides['baseline'][c]['srv50'])} | {ms(sides['lite'][c]['srv50'])} |")

    print("\nTTFA p50 theo từng block, theo đúng thứ tự đã chạy:\n")
    blocks = sorted({b.name.split("-slots-")[0] for side in ("baseline", "lite")
                     for b in (REPO / side / "results" / exp).glob("block-*")}, key=lambda s: int(s.split("-")[1]))
    owner = {b: side for side in ("baseline", "lite") for b in blocks
             if any((REPO / side / "results" / exp).glob(f"{b}-slots-*"))}
    print("| Người dùng | " + " | ".join(f"{b} ({'SDK' if owner[b] == 'baseline' else 'lite'})" for b in blocks) + " |")
    print("|---:|" + "---:|" * len(blocks))
    for c, slots in ABBA_LEVELS:
        cells = [ms(stats(load(owner[b], exp, f"{b}-slots-{slots}", c))["ttfa50"]) for b in blocks]
        print(f"| {c} | " + " | ".join(cells) + " |")

    ok = True
    if check:
        print()
        for side in ("baseline", "lite"):
            path = REPO / side / "results" / exp / "report.json"
            if not path.exists():
                print(f"(không có {path.relative_to(REPO)} để đối chiếu)")
                continue
            cases = json.loads(path.read_text())["cases"]
            for c, _ in ABBA_LEVELS:
                got, want = sides[side][c], cases[str(c)]
                pairs = [("n", got["n"], want["n"]), ("client_ttfa_p50_ms", got["ttfa50"], want["client_ttfa_p50_ms"]),
                         ("client_ttfa_p90_ms", got["ttfa90"], want["client_ttfa_p90_ms"]),
                         ("server_ttfa_p50_ms", got["srv50"], want["server_ttfa_p50_ms"]),
                         ("rtf_p50", got["rtf50"], want["rtf_p50"]), ("rtf_p90", got["rtf90"], want["rtf_p90"])]
                bad = [f"{k}: {g} != {w}" for k, g, w in pairs if g is None or abs(g - w) > 1e-6]
                ok &= not bad
                print(f"{path.relative_to(REPO)} c={c}: {'khớp' if not bad else 'LỆCH ' + '; '.join(bad)}")
    return ok


def overlap(exp: str, check: bool) -> bool:
    print(f"\n## Thử cho codec chạy song song ({exp})\n")
    print("| Người dùng cùng lúc | Cấu hình | n | TTFA | RTF |\n|---:|---|---:|---:|---:|")
    got = {}
    for c in OVERLAP_LEVELS:
        for side, name, label in OVERLAP_VARIANTS:
            s = got[name, c] = stats(load(side, exp, f"block-*-{name}", c))
            print(f"| {c} | {label} | {s['n']} | {ms(s['ttfa50'])} | {s['rtf50']:.3f} |")
    ok = True
    if check:
        print()
        for side in ("baseline", "lite"):
            path = REPO / side / "results" / exp / "summary.json"
            if not path.exists():
                print(f"(không có {path.relative_to(REPO)} để đối chiếu)")
                continue
            for g in json.loads(path.read_text())["groups"]:
                s = got[g["name"], g["c"]]
                pairs = [("n", s["n"], g["n"]), ("ttfa p50", s["ttfa50"] / 1e3, g["ttfa_s"]["p50"]),
                         ("ttfa p90", s["ttfa90"] / 1e3, g["ttfa_s"]["p90"]),
                         ("rtf p50", s["rtf50"], g["rtf"]["p50"]), ("rtf p90", s["rtf90"], g["rtf"]["p90"])]
                bad = [f"{k}: {a} != {b}" for k, a, b in pairs if abs(a - b) > 1e-6]
                ok &= not bad
                print(f"{path.relative_to(REPO)} {g['name']} c={g['c']}: {'khớp' if not bad else 'LỆCH ' + '; '.join(bad)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abba", default="abba-20261013", help="folder name under <side>/results/ ('' to skip)")
    ap.add_argument("--overlap", default="", help="folder name under <side>/results/, e.g. overlap-20261013 (not in PHASE1.md)")
    ap.add_argument("--check", action="store_true", help="compare with the committed report.json / summary.json")
    a = ap.parse_args()
    ok = True
    if a.abba:
        ok &= abba(a.abba, a.check)
    if a.overlap:
        ok &= overlap(a.overlap, a.check)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
