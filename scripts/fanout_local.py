"""Local process-pool ensemble scaling to K=1024: validates that the reduce and
its uncertainty scale with fanout, cheaply and without a sandbox quota. This is
the LOCAL backend (OS process forks), distinct from the islo microVM curve.

    uv run python scripts/fanout_local.py
"""
import time
import numpy as np
from bmr.backends import LocalBackend
from bmr.core import reduce_partition


def main():
    rng = np.random.default_rng(0)
    be = LocalBackend()
    lines = ["Local process-pool ensemble scaling (tiny mean shards, true mu=5.0)",
             f"{'fanout K':>9} {'total_s':>9} {'forks/s':>9} {'CI_halfwidth':>13} {'|pool-5|':>9}"]
    for K in [16, 64, 256, 1024]:
        shards = [{"scenario": "mean", "x": rng.normal(5.0, 1.5, 80).tolist(), "shard_id": i}
                  for i in range(K)]
        t0 = time.perf_counter()
        cds = be.map_shards(shards)
        t = time.perf_counter() - t0
        pooled = reduce_partition(cds)
        lo, hi = pooled.ci()[0]
        lines.append(f"{K:>9} {t:>9.2f} {K / t:>9.0f} {(hi - lo) / 2:>13.4f} "
                     f"{abs(pooled.theta[0] - 5.0):>9.4f}")
    out = "\n".join(lines)
    print(out)
    with open("runs/fanout_local.txt", "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    main()
