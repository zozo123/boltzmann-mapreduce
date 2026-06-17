"""Execution backends: fork map workers locally, or as islo sandboxes.

LocalBackend forks OS processes (the ensemble of replicas, runnable today).
IsloBackend snapshots a parent sandbox once and forks one sandbox per shard ---
the paper's substrate --- via the real ``islo`` CLI.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from bmr.core import CD
from bmr.worker import SENTINEL, run_one


class IsloUnavailable(RuntimeError):
    """Raised when the islo CLI is missing or not authenticated."""


def _shard_label(shard: dict, i: int) -> str:
    sid = shard.get("shard_id", i)
    return f"bmr-shard-{sid}"


class LocalBackend:
    """Fork one OS process per shard via a process pool."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def map_shards(self, shards: list[dict]) -> list[CD]:
        if not shards:
            return []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(run_one, shards))


class IsloBackend:
    """Snapshot-once, fork-per-shard on islo sandboxes.

    Setup (once); islo.yaml's setup_script installs uv and runs `uv sync`::

        islo use bmr-base --source github://<owner>/<repo> -- true
        islo snapshot save bmr-base --name bmr-base

    Per shard (fork); the entry point is an on-box AI harness when ``harness`` is
    set, else a direct ``uv run`` of the worker::

        islo use bmr-shard-<k> --snapshot bmr-base --workdir <repo> \
            -e SHARD_B64=<base64-json> --delete-after <n> \
            --agent claude --task "<run the worker, relay its output>"
        # or, harness=None:
        islo use bmr-shard-<k> --snapshot bmr-base --workdir <repo> \
            -e SHARD_B64=<base64-json> --delete-after <n> -- uv run python -m bmr.worker
    """

    def __init__(self, base_snapshot: str = "bmr-base", delete_after: int = 180,
                 image: str | None = None, max_inflight: int = 8,
                 workdir: str = "/workspace/boltzmann-mapreduce",
                 harness: str | None = None):
        self.base_snapshot = base_snapshot
        self.delete_after = delete_after
        self.image = image
        self.max_inflight = max_inflight
        self.workdir = workdir
        # When set (e.g. "claude"), each fork's entry point is an on-box AI agent
        # harness (islo --agent ...) rather than a direct interpreter call. The
        # harness runs the worker and echoes its sentinel-delimited output.
        self.harness = harness

    # The task handed to the on-box AI harness. It must run the worker and print
    # its output verbatim so the sentinel payload survives.
    HARNESS_TASK = (
        "Run the shell command `uv run python -m bmr.worker` in the current "
        "directory and print its stdout exactly as produced. It reads the shard "
        "from the SHARD_B64 environment variable and prints a single line wrapped "
        "in __BMR__ markers. Do not edit any files; only run it and relay the line."
    )

    # -- setup helper (documented; run once before forking) --
    def ensure_base(self, repo_source: str) -> list[list[str]]:
        """Return the islo commands that build + snapshot the parent sandbox.

        The repo's islo.yaml setup_script installs uv and runs `uv sync`, so the
        snapshot carries a locked, uv-managed Python environment.
        """
        return [
            ["islo", "use", "bmr-base", "--source", repo_source, "--", "true"],
            ["islo", "snapshot", "save", "bmr-base", "--name", self.base_snapshot],
        ]

    def _check_cli(self) -> None:
        if shutil.which("islo") is None:
            raise IsloUnavailable("islo CLI not found on PATH. Install it, then run: islo login")

    def _run_shard(self, shard: dict, i: int) -> CD:
        b64 = base64.b64encode(json.dumps(shard).encode()).decode()
        name = _shard_label(shard, i)
        cmd = ["islo", "use", name, "--snapshot", self.base_snapshot,
               "--workdir", self.workdir,
               "-e", f"SHARD_B64={b64}", "--delete-after", str(self.delete_after)]
        if self.image:
            cmd += ["--image", self.image]
        if self.harness:
            # Entry point is an on-box AI agent harness, not a direct script.
            cmd += ["--agent", self.harness, "--task", self.HARNESS_TASK]
        else:
            # Fallback for when no harness is needed: run the worker directly via uv.
            cmd += ["--", "uv", "run", "python", "-m", "bmr.worker"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_s = time.perf_counter() - t0
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        low = blob.lower()
        if proc.returncode != 0 and SENTINEL not in blob:
            if any(s in low for s in ("login", "unauthor", "not authenticated", "expired", "forbidden")):
                raise IsloUnavailable(f"islo not authenticated (shard {name}). Run: islo login")
            raise RuntimeError(f"islo shard {name} failed (rc={proc.returncode}): {blob.strip()[:400]}")
        payload = self._extract(blob)
        if payload is None:
            if self.harness:
                raise RuntimeError(
                    f"on-box AI harness '{self.harness}' was launched for shard {name} but "
                    f"returned no one-shot payload (an --agent session is interactive, not a "
                    f"pipe). Run without --harness for the deterministic reduce.")
            raise RuntimeError(f"no BMR payload from shard {name}: {blob.strip()[:400]}")
        if "error" in payload:
            raise RuntimeError(f"worker error in shard {name}: {payload['error']}")
        cd = CD.from_dict(payload)
        # Round-trip-inclusive fork dispatch time (restore + run + capture), not the
        # in-host restore in isolation.
        cd.meta["wall_s"] = round(wall_s, 3)
        cd.meta["sandbox"] = name
        return cd

    @staticmethod
    def _extract(blob: str):
        first = blob.find(SENTINEL)
        if first == -1:
            return None
        second = blob.find(SENTINEL, first + len(SENTINEL))
        if second == -1:
            return None
        return json.loads(blob[first + len(SENTINEL):second])

    def map_shards(self, shards: list[dict]) -> list[CD]:
        if not shards:
            return []
        self._check_cli()
        results: list[CD | None] = [None] * len(shards)
        with ThreadPoolExecutor(max_workers=self.max_inflight) as ex:
            futs = {ex.submit(self._run_shard, s, i): i for i, s in enumerate(shards)}
            for fut in futs:
                i = futs[fut]
                results[i] = fut.result()
        return [r for r in results if r is not None]


def get_backend(name: str, **kw):
    if name == "local":
        return LocalBackend(**{k: v for k, v in kw.items() if k == "max_workers"})
    if name == "islo":
        return IsloBackend(**{k: v for k, v in kw.items()
                              if k in ("base_snapshot", "delete_after", "image",
                                       "max_inflight", "workdir", "harness")})
    raise ValueError(f"unknown backend: {name!r}")
