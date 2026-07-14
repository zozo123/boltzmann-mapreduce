"""Sandbox-creation sweep on Tensorlake.

Creates default sandboxes under a concurrency cap, runs a trivial command in
each, and measures create and round-trip latency. Unlike the islo backend, this
script does not restore a named shared snapshot; its results must not be labeled
as snapshot-fork measurements.

The Tensorlake SDK is locked as an optional project extra. Export TENSORLAKE_API_KEY
(never committed), then run:

    uv run --locked --extra tensorlake python scripts/tensorlake_fanout.py
    uv run --locked --extra tensorlake python scripts/tensorlake_fanout.py 8 4 16 64
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tensorlake.sandbox import Sandbox

LOG = "runs/tensorlake_fanout.txt"


def create_once(_):
    t0 = time.perf_counter()
    sb = None
    try:
        sb = Sandbox.create()
        t_create = time.perf_counter() - t0
        r = sb.run("sh", args=["-lc", "echo __BMR__ok__BMR__"])
        out = getattr(r, "stdout", "") or ""
        ok = "ok__BMR__" in out or getattr(r, "exit_code", 0) == 0
        return {"ok": ok, "create_s": t_create, "wall_s": time.perf_counter() - t0, "reason": ""}
    except Exception as e:  # noqa
        return {"ok": False, "create_s": 0.0, "wall_s": time.perf_counter() - t0,
                "reason": type(e).__name__}
    finally:
        if sb is not None:
            try:
                sb.terminate()
            except Exception:
                pass


def run_level(k, concurrency):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        res = list(ex.map(create_once, range(k)))
    total = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    cw = np.array([r["create_s"] for r in ok]) if ok else np.array([0.0])
    ww = np.array([r["wall_s"] for r in ok]) if ok else np.array([0.0])
    fails = {}
    for r in res:
        if not r["ok"]:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
    return {"k": k, "ok": len(ok), "rate": len(ok) / k, "total": total,
            "tput": len(ok) / total if total else 0,
            "create_p50": float(np.percentile(cw, 50)), "create_p95": float(np.percentile(cw, 95)),
            "wall_p50": float(np.percentile(ww, 50)), "fails": fails}


def main(argv):
    concurrency = int(argv[0]) if argv else 8
    ladder = [int(x) for x in argv[1:]] or [4, 16, 64, 256, 1024]
    with open(LOG, "a") as fh:
        hdr = (f"\n=== Tensorlake sandbox-create sweep (concurrency={concurrency}, "
               f"create+run+terminate) ===\n"
               f"{'fanout':>7} {'ok':>6} {'succ%':>6} {'total_s':>9} {'sbx/s':>8} "
               f"{'create_p50':>11} {'create_p95':>11} {'wall_p50':>9}\n")
        fh.write(hdr); fh.flush(); print(hdr, end="")
        for k in ladder:
            r = run_level(k, concurrency)
            line = (f"{r['k']:>7} {r['ok']:>6} {100*r['rate']:>5.0f}% {r['total']:>9.1f} "
                    f"{r['tput']:>8.1f} {r['create_p50']:>11.2f} {r['create_p95']:>11.2f} "
                    f"{r['wall_p50']:>9.2f}" + (f"  fails={r['fails']}" if r['fails'] else ""))
            fh.write(line + "\n"); fh.flush(); print(line)
            if r["rate"] < 0.5:
                msg = f"  wall hit at fanout {k} (success {100*r['rate']:.0f}%); stopping."
                fh.write(msg + "\n"); fh.flush(); print(msg); break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
