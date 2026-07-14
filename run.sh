#!/usr/bin/env bash
# End-to-end release check with uv. Installs Python + locked dependencies into a
# uv-managed environment, runs the Python paths, rebuilds the paper, and verifies
# the published artifact contract. Set BMR_SKIP_PAPER=1 for the Python-only path.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || { echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv sync --locked

args=()
if [[ "${BMR_SKIP_PAPER:-0}" == "1" ]]; then
  args+=(--skip-paper)
fi
uv run --locked python scripts/verify_e2e.py "${args[@]}"

echo "Run the deterministic worker path on islo:"
echo "  islo login && islo use bmr-base --source github://zozo123/boltzmann-mapreduce -- true"
echo "  islo snapshot save bmr-base --name bmr-base"
echo "  uv run --locked python demo.py --backend islo --scenario mean"
