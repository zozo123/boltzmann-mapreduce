"""Per-fork map worker --- the entrypoint a forked sandbox runs.

Reads one shard (env ``SHARD_B64`` of base64'd JSON, else ``--file``, else stdin),
computes its confidence density, and prints a single sentinel-delimited line:

    __BMR__<json>__BMR__

The sentinel lets the islo backend strip any CLI banner noise around the payload.
"""
from __future__ import annotations

import base64
import json
import os
import sys

from bmr.core import CD, map_mean, map_linreg, map_logistic

SENTINEL = "__BMR__"


def run_one(shard: dict) -> CD:
    """Map a shard dict to a CD. Used by both this CLI and the LocalBackend."""
    backend = os.environ.get("BMR_BACKEND", "local")
    scenario = shard.get("scenario", "mean")
    sid = shard.get("shard_id")
    if scenario == "mean":
        cd = map_mean(shard["x"], shard_id=sid, backend=backend)
    elif scenario == "linreg":
        cd = map_linreg(shard["X"], shard["y"], shard_id=sid, backend=backend)
    elif scenario == "logistic":
        cd = map_logistic(shard["X"], shard["y"], shard_id=sid, backend=backend)
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return CD(
        theta_hat=list(cd.theta_hat),
        info_per_obs=[list(row) for row in cd.info_per_obs],
        n=cd.n,
        meta=dict(cd.meta),
        lineage=list(shard.get("lineage", [])),
        evidence_ids=list(shard.get("evidence_ids", [])),
    )


def _load_shard(argv) -> dict:
    b64 = os.environ.get("SHARD_B64", "").strip()
    if b64:
        return json.loads(base64.b64decode(b64).decode())
    if "--file" in argv:
        path = argv[argv.index("--file") + 1]
        with open(path) as fh:
            return json.load(fh)
    return json.loads(sys.stdin.read())


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        shard = _load_shard(argv)
        cd = run_one(shard)
        sys.stdout.write(f"{SENTINEL}{json.dumps(cd.to_dict())}{SENTINEL}\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:  # surface a structured error
        sys.stderr.write(f"{SENTINEL}{json.dumps({'error': str(exc)})}{SENTINEL}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
