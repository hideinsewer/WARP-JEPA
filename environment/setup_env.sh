#!/bin/bash
# Builds the exact environment used for every experiment in the paper (Linux x86_64, CUDA 12.8 driver).
# Usage (from the repository root): bash environment/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install uv==0.9.28
uv python install 3.10.19
uv venv .venv --python 3.10.19
# main environment: torch 2.11.0+cu128, gymnasium 1.3.0, pymunk 7.3.0, numpy 2.2.6, ...
uv pip install --python .venv/bin/python --no-deps -r environment/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# overlays kept out of the main venv (installed with --target, put on PYTHONPATH by environment/activate.sh):
#   runtime     : Transformers 5.20.2 stack, needed to unpickle the released ViT encoders
#   runtime_sim : MuJoCo 3.14.0 / dm_control / OGBench 1.2.1 for Cube and AntMaze
uv pip install --python .venv/bin/python --no-deps --target cost_probe/artifacts/runtime -r environment/overlay_runtime.txt
uv pip install --python .venv/bin/python --no-deps --target cost_probe/artifacts/runtime_sim -r environment/overlay_runtime_sim.txt
# profile_cost.py and the trainers record the git commit in their metadata, so the code must be a git repository
if [ ! -d .git ]; then
  git init -q && git add -A && git -c user.name=warp-jepa -c user.email=warp-jepa@localhost commit -qm "code release"
fi
echo "ENV_READY: source environment/activate.sh"
