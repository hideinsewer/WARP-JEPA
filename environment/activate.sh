# source environment/activate.sh   (from the repository root)
R=$(pwd)
source "$R/.venv/bin/activate"
export PYTHONPATH=$R:$R/cost_probe/artifacts/runtime:$R/cost_probe/artifacts/runtime_sim${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HJEPA_HOME=${HJEPA_HOME:-$R/cost_probe/artifacts/data}
