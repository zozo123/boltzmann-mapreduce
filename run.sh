#!/usr/bin/env bash
# Quickstart with uv. Installs Python + deps into a uv-managed environment and
# runs the tests and demo. The local backend forks OS processes directly
# (no AI harness needed); on islo the entry point is an on-box AI harness
# (see README "Run on islo").
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv sync

echo "== tests =="
uv run pytest -q

echo "== demo: mean + byzantine =="
uv run python demo.py --scenario mean --byzantine

echo "== demo: linreg =="
uv run python demo.py --scenario linreg

echo "Figures in ./figs/. Run on islo with an on-box AI harness:"
echo "  islo login && islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true"
echo "  islo snapshot save bmr-base --name bmr-base"
echo "  uv run python demo.py --backend islo --scenario mean --harness claude"
