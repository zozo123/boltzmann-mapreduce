"""Fanout sweep on islo: restore-and-run the named bmr-base OCI snapshot on a
geometric ladder, capped at a fixed concurrency.  The recorded wall time includes
client dispatch, scheduling, snapshot restoration, command execution, capture,
and network round-trip; it is not an isolated restore-latency measurement.  Each
worker runs ``true``.  Capacity failures are retained in the log.

    uv run python scripts/fanout_sweep.py            # ladder 4..1024, 32 concurrent
    uv run python scripts/fanout_sweep.py 32 4 16 64 # concurrency 32, ladder 4 16 64
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SNAPSHOT = "bmr-base"
LOG = "runs/fanout_sweep.txt"
AUTH_HINT = ("login", "unauthor", "not authenticated", "expired", "forbidden")
CAPACITY = ("no capacity", "capacity available", "try again", "quota", "rate limit")


def fork_once(name: str, timeout: int = 600):
    # --delete-after is only a backstop; we delete immediately after capture so the
    # live-sandbox count stays ~= concurrency (lets us reach high TOTAL fanout under
    # the account's concurrent quota).
    cmd = ["islo", "use", name, "--snapshot", SNAPSHOT, "--delete-after", "120", "--", "true"]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["islo", "rm", name, "-f"], capture_output=True, timeout=60)
        return {"ok": False, "wall": time.perf_counter() - t0, "reason": "timeout"}
    wall = time.perf_counter() - t0
    blob = ((p.stdout or "") + (p.stderr or "")).lower()
    reason = ""
    if p.returncode != 0:
        reason = ("capacity" if any(s in blob for s in CAPACITY)
                  else "auth" if any(s in blob for s in AUTH_HINT) else "error")
    subprocess.run(["islo", "rm", name, "-f"], capture_output=True, timeout=60)  # prompt cleanup
    return {"ok": p.returncode == 0, "wall": wall, "reason": reason}


def run_level(k: int, concurrency: int):
    names = [f"bmr-fan-{k}-{i}" for i in range(k)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        res = list(ex.map(fork_once, names))
    total = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    walls = np.array([r["wall"] for r in ok]) if ok else np.array([0.0])
    fails = {}
    for r in res:
        if not r["ok"]:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
    return {
        "k": k, "concurrency": concurrency, "ok": len(ok), "total": k,
        "success_rate": len(ok) / k, "total_wall_s": total,
        "throughput_forks_s": (len(ok) / total) if total > 0 else 0.0,
        "fork_p50_s": float(np.percentile(walls, 50)),
        "fork_p95_s": float(np.percentile(walls, 95)),
        "fork_mean_s": float(walls.mean()), "fails": fails,
    }


def main(argv):
    concurrency = int(argv[0]) if argv else 32
    ladder = [int(x) for x in argv[1:]] or [4, 16, 64, 256, 1024]
    with open(LOG, "a") as fh:
        hdr = (f"\n=== fanout sweep (concurrency={concurrency}, snapshot={SNAPSHOT}, "
               f"fork=`-- true`) ===\n"
               f"{'fanout':>7} {'ok':>6} {'succ%':>6} {'total_s':>9} {'forks/s':>9} "
               f"{'p50_s':>7} {'p95_s':>7}\n")
        fh.write(hdr); fh.flush()
        print(hdr, end="")
        for k in ladder:
            row = run_level(k, concurrency)
            line = (f"{row['k']:>7} {row['ok']:>6} {100*row['success_rate']:>5.1f}% "
                    f"{row['total_wall_s']:>9.1f} {row['throughput_forks_s']:>9.2f} "
                    f"{row['fork_p50_s']:>7.2f} {row['fork_p95_s']:>7.2f}"
                    + (f"  fails={row['fails']}" if row['fails'] else ""))
            fh.write(line + "\n"); fh.flush()
            print(line)
            if row["success_rate"] < 0.5:
                msg = (f"  capacity/error wall hit at fanout {k} "
                       f"(success {100*row['success_rate']:.0f}%); stopping ladder.")
                fh.write(msg + "\n"); fh.flush(); print(msg)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
