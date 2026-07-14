"""Build and publish the revised paper through the repository's uv workflow.

Run from any directory with::

    uv run --locked python scripts/build_paper.py

LaTeX remains a system dependency; Python orchestration and repository dependencies
are provided by the uv-locked environment. The build happens in a temporary tree so
TeX intermediates never dirty the checkout.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PDF_NAME = "uncertainty-aware-reduction.pdf"
SOURCE_DATE_EPOCH = "1783987200"  # 2026-07-14 UTC; stable PDF metadata.


def _require(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(
            f"missing system dependency: {command}. Install a TeX distribution "
            "that provides pdflatex and bibtex, then rerun the uv command."
        )


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def main() -> int:
    for command in ("pdflatex", "bibtex"):
        _require(command)

    with tempfile.TemporaryDirectory(prefix="bmr-paper-") as tmp:
        build = Path(tmp) / "paper"
        build.mkdir()
        for name in ("main_5pp.tex", "body_5pp.tex", "refs.bib"):
            shutil.copy2(PAPER / name, build / name)
        shutil.copytree(PAPER / "figs", build / "figs")

        texmf = Path(tmp) / "texmf"
        texmf.mkdir()
        env = os.environ.copy()
        env.update(
            TEXMFHOME=str(texmf),
            SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH,
            FORCE_SOURCE_DATE="1",
        )
        latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main_5pp.tex"]
        _run(latex, build, env)
        _run(["bibtex", "main_5pp"], build, env)
        _run(latex, build, env)
        _run(latex, build, env)

        log = (build / "main_5pp.log").read_text(encoding="utf-8", errors="replace")
        forbidden = re.compile(r"Overfull|Citation .* undefined|undefined references", re.I)
        match = forbidden.search(log)
        if match:
            raise SystemExit(f"paper log contains a release-blocking warning: {match.group(0)}")

        built_pdf = build / "main_5pp.pdf"
        if not built_pdf.read_bytes().startswith(b"%PDF-") or built_pdf.stat().st_size < 100_000:
            raise SystemExit("paper build did not produce a valid non-trivial PDF")

        destinations = (ROOT / "output" / "pdf" / PDF_NAME, ROOT / "docs" / PDF_NAME)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_pdf, destination)
            print(f"published {destination.relative_to(ROOT)}")

    if destinations[0].read_bytes() != destinations[1].read_bytes():
        raise SystemExit("published paper artifacts differ")
    print(f"paper build verified ({destinations[0].stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
