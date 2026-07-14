"""Run the complete locked repository verification through uv.

    uv run --locked python scripts/verify_e2e.py

Cloud-provider sweeps are intentionally excluded because they require credentials,
quota, and billable external resources. Their SDKs are nevertheless resolved in
``uv.lock`` as optional extras and their modules are compiled below.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "uncertainty-aware-reduction.pdf"
DOCS_PDF = ROOT / "docs" / "uncertainty-aware-reduction.pdf"


def _run(label: str, command: list[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _verify_site_contract() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    required = (
        "arXiv:2607.09689",
        "doi:10.48550/arXiv.2607.09689",
        "https://raw.githubusercontent.com/zozo123/boltzmann-mapreduce/"
        "main/docs/uncertainty-aware-reduction.pdf",
        "uv run --locked python scripts/verify_e2e.py",
        'id="explorer"',
        'id="explorerChart"',
        'data-explorer-preset="falsePrecision"',
        'data-explorer-preset="overlap"',
        "data-worker-enabled",
        "data-auto-sample",
    )
    missing = [value for value in required if value not in html]
    if missing:
        raise SystemExit(f"website release contract is missing: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the complete uv-locked artifact")
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="skip the TeX build when only the Python path is needed",
    )
    args = parser.parse_args(argv)

    _run("tests", [sys.executable, "-m", "pytest", "-q"])
    _run(
        "compile",
        [sys.executable, "-m", "compileall", "-q", "bmr", "scripts", "demo.py"],
    )

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required; run this verifier through uv as documented")
    with tempfile.TemporaryDirectory(prefix="bmr-dist-") as dist:
        _run("package", [uv, "build", "--no-build-isolation", "--out-dir", dist])
        artifacts = {path.suffix for path in Path(dist).iterdir() if path.is_file()}
        if ".whl" not in artifacts or ".gz" not in artifacts:
            raise SystemExit("uv build did not produce both wheel and source distribution")

    with tempfile.TemporaryDirectory(prefix="bmr-e2e-") as tmp:
        output = Path(tmp) / "figs"
        _run(
            "mean demo + outlier stress test",
            [
                sys.executable,
                "demo.py",
                "--scenario",
                "mean",
                "--byzantine",
                "--out",
                str(output),
            ],
        )
        _run(
            "linear-regression demo",
            [sys.executable, "demo.py", "--scenario", "linreg", "--out", str(output)],
        )
        _run(
            "logistic demo",
            [sys.executable, "demo.py", "--scenario", "logistic", "--out", str(output)],
        )
        expected_figures = {"fig_gibbs.png", "fig_cooling.png", "fig_byzantine.png"}
        missing_figures = sorted(name for name in expected_figures if not (output / name).is_file())
        if missing_figures:
            raise SystemExit(f"demo did not produce expected figures: {missing_figures}")

    if not args.skip_paper:
        _run("paper", [sys.executable, "scripts/build_paper.py"])

    if not PDF.is_file() or not DOCS_PDF.is_file() or PDF.read_bytes() != DOCS_PDF.read_bytes():
        raise SystemExit("release PDF copies are missing or differ")
    _verify_site_contract()

    scope = (
        "Python, package, demos, artifacts, website"
        if args.skip_paper
        else "Python, package, demos, paper, artifacts, website"
    )
    print(f"\nE2E PASS: {scope} verified through the uv-locked environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
