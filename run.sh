#!/usr/bin/env bash
# Quickstart with uv. Installs Python + deps into a uv-managed environment and
# runs the tests and demo. The local backend uses a Python process pool; the
# reproducible islo path invokes the deterministic worker directly.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv sync

echo "== tests =="
uv run pytest -q

echo "== demo: mean + outlier stress test =="
uv run python demo.py --scenario mean --byzantine

echo "== demo: linreg =="
uv run python demo.py --scenario linreg

echo "Figures in ./figs/. Run the deterministic worker path on islo:"
echo "  islo login && islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true"
echo "  islo snapshot save bmr-base --name bmr-base"
echo "  uv run python demo.py --backend islo --scenario mean"
