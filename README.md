# WARP-JEPA: World-Model-in-the-Loop Amortized Rollout Planning for Frozen Hierarchical JEPAs

This is the code for *World-model-in-the-loop Amortized Rollout Planning* (WARP-JEPA). It replaces the online
gradient-descent planner of a frozen hierarchical JEPA world model (H-JEPA) with learned K-head proposers at
every level. Each proposal is verified in one batched rollout of the frozen predictor.

- `cost_probe/` contains our code: proposers, training, closed-loop evaluation, timing and statistics.
- `h_jepa/`, `stable_worldmodel/` and `scripts/` are the public [H-JEPA](https://github.com/kevinghst/H-JEPA) code base (MIT license, see `LICENSE`).
  We use it unchanged to train the world models and as the Online GD baseline.

## 1. Environment

All experiments in the paper ran in this environment on single NVIDIA H200 GPUs. It needs Linux x86_64 and an
NVIDIA driver that supports CUDA 12.8.

```bash
bash environment/setup_env.sh      # uv, CPython 3.10.19, pinned packages
source environment/activate.sh     # venv + PYTHONPATH + MUJOCO_GL=egl
```

- `environment/requirements.txt`: the complete frozen package list (`torch==2.11.0+cu128`,
  `torchvision==0.26.0`, `gymnasium==1.3.0`, `pymunk==7.3.0`, `numpy==2.2.6`, ...), installed with `--no-deps`.
- `environment/overlay_runtime.txt`: Transformers 5.20.2 and its stack. The released ViT encoder pickles need it.
- `environment/overlay_runtime_sim.txt`: MuJoCo 3.14.0, dm_control 1.0.47 and OGBench 1.2.1 for Cube and
  AntMaze. MuJoCo versions change episode outcomes.
- MuJoCo EGL rendering needs the system libglvnd libraries (`libEGL.so.1`, `libOpenGL.so.0`; we used Ubuntu 24.04
  `libegl1 libgl1 libglvnd0 libglx0 libopengl0` 1.7.0-1build1). Minimal CUDA containers often lack them.

Unit tests (CPU):

```bash
CUDA_VISIBLE_DEVICES= python -m unittest discover -s cost_probe -p 'test_*.py'
```

## 2. Data, world models and tasks

The datasets, evaluation tasks and world-model configurations come from the public H-JEPA release. Push-T and Cube training data come from the public LeWorldModel release. Place them as follows:

```
cost_probe/artifacts/data/                 # = $HJEPA_HOME
  pusht_expert_{train,val}.h5, cube_single_expert_{train,val}.h5,
  visual_antmaze_medium_*.h5, fourroom_tp35_d1*.h5
  eval_trajs/{pusht,fourroom,ogbench,ant}/...   # the 50 official tasks per environment
  ckpts/<env>/<env>_hjepa_l<L>/seed<S>/       # trained H-JEPA world models
cost_probe/artifacts/models -> data/ckpts      # symlink
h_jepa/assets/eval_trajs -> ../../cost_probe/artifacts/data/eval_trajs   # symlink
```

Train the frozen world models with the official configurations (`<env>` in `pusht fourroom cube ant`, `<L>` in
`2 3 4`, seeds 42/43/44). The run writes `$HJEPA_HOME/ckpts/...` and runs the official end-of-training
planning evaluation:

```bash
(cd h_jepa && python main_hjepa.py --config-name <env>_hjepa_l<L> seed=<S> final_probing_decoding_eval.enabled=false)
```

Held-out Push-T tasks (seeds 1003-1006, 50 tasks each; seed 42 reproduces the official task file):

```bash
python cost_probe/make_tasks.py --seed 1003 --out cost_probe/artifacts/data/eval_trajs/pusht/fresh/goal_offset_75_val_seed1003.pt
```

## 3. Two-level models (L2: MM, TM, TT)

```bash
# frozen level-1/level-2 latents of the training split (and val for seed 42); gated bit-exact against the planner
python cost_probe/build_l1_cache.py --env <env> --split train --model-seed <S> --batched-encode
python cost_probe/build_l1_cache.py --env <env> --split val   --model-seed <S>

# subordinate (level-1) proposers, K=16, lambda_bc=1
python cost_probe/train_decoder.py --env <env> --model-seed <S> --out cost_probe/runs/decoder/<env>_mlp_s<S>
python cost_probe/train_decoder.py --env <env> --model-seed <S> --architecture tf_ar_wm --out cost_probe/runs/decoder/<env>_tf_s<S>

# top-level (level-2) proposers, K=32, lambda_bc=0.05
# goal offsets: Push-T default, FourRoom --offset-range 5 95, Cube --offset-range 5 40, AntMaze --offset-range 5 135
python cost_probe/train_upper.py --env <env> --model-seed <S> --lambda-bc 0.05 [OFFSETS] --out cost_probe/runs/upper/<env>_mlp_s<S>
python cost_probe/train_upper.py --env <env> --model-seed <S> --lambda-bc 0.05 --architecture tf_ar_wm [OFFSETS] --out cost_probe/runs/upper/<env>_tf_s<S>
```

Closed-loop evaluation on the 50 official tasks (`--task-file F` for the held-out Push-T tasks, `--tasks 10` for
the sequential timing runs). Each run writes `report.json`, `metadata.json`, `parity.json` and per-call timings.

```bash
U=cost_probe/runs/upper; D=cost_probe/runs/decoder
FAST="--action-grads-only --upper-warm-start --lower-bypass-iters 10"
python cost_probe/profile_cost.py --env <env> --model-seed <S> --tasks 50 --action-grads-only --out runs/<env>_s<S>_online_gd
python cost_probe/profile_cost.py --env <env> --model-seed <S> --tasks 50 $FAST --upper-proposal $U/<env>_mlp_s<S>/proposal.pt --lower-decoder $D/<env>_mlp_s<S>/decoder.pt --out runs/<env>_s<S>_mm
python cost_probe/profile_cost.py --env <env> --model-seed <S> --tasks 50 $FAST --upper-proposal $U/<env>_tf_s<S>/proposal.pt  --lower-decoder $D/<env>_mlp_s<S>/decoder.pt --out runs/<env>_s<S>_tm
python cost_probe/profile_cost.py --env <env> --model-seed <S> --tasks 50 $FAST --upper-proposal $U/<env>_tf_s<S>/proposal.pt  --lower-decoder $D/<env>_tf_s<S>/decoder.pt  --out runs/<env>_s<S>_tt
```

`--action-grads-only` keeps the released Online GD planner bit-exact and computes only action gradients. Every run
checks this against the released planner and writes the result to `parity.json`. The ablation proposers use
`--architecture tf_parallel` or `tf_ar_action`. `--upper-refine-iters R` adds R gradient steps after the proposals,
and `--upper-samples N` changes the GD candidate count.

## 4. Three- and four-level models (L3/L4: MMM/TMM/TTT, MMMM/TMMM/TTTT)

```bash
python cost_probe/build_l1_cache.py --env <env> --split train --model-seed <S> --levels 3 --batched-encode
python cost_probe/build_l1_cache.py --env <env> --split val   --model-seed <S> --levels 3
for A in mlp tf_ar_wm; do
  python cost_probe/three_level.py train-l1 --env <env> --model-seed <S> --architecture $A --out R/l1_$A
  python cost_probe/three_level.py train-l2 --env <env> --model-seed <S> --architecture $A --out R/l2_$A
  python cost_probe/three_level.py train-l3 --env <env> --model-seed <S> --architecture $A --out R/l3_$A
done
python cost_probe/three_level.py eval --env <env> --model-seed <S> --tasks 50 --action-grads-only --out runs/l3_online_gd
python cost_probe/three_level.py eval --env <env> --model-seed <S> --tasks 50 --action-grads-only --bypass-iters 10 \
  --l3-proposal R/l3_tf_ar_wm/proposal.pt --l2-decoder R/l2_mlp/l2_decoder.pt --l1-decoder R/l1_mlp/decoder.pt --out runs/l3_tmm
```

`four_level.py` has the same interface (`train-l1` ... `train-l4`; `eval` with `--l4-proposal --l3-decoder
--l2-decoder --l1-decoder`) and uses caches built with `--levels 4`.

## Pretrained proposers

The trained proposers of every model in the paper (180 checkpoints, each with its `config.json` and
`train_log.jsonl`) are attached to the [v1.0 release](https://github.com/hideinsewer/WARP-JEPA/releases/tag/v1.0):

| Archive | Contents |
| --- | --- |
| `warp_jepa_proposers_l2.tar` | two-level models: `decoder/` (level 1) and `upper/` (level 2) for Push-T, FourRoom, Cube, AntMaze, seeds 42-44 |
| `warp_jepa_proposers_l3.tar` | three-level models: `l3/<env>_l3_{l1dec,l2dec,l3up}_<arch>_s<S>_*` |
| `warp_jepa_proposers_l4.tar` | four-level models: `l4/<env>_l4_{l1dec,l2dec,l3dec,l4up}_<arch>_s<S>_*` |

```bash
for p in l2 l3 l4; do
  curl -LO https://github.com/hideinsewer/WARP-JEPA/releases/download/v1.0/warp_jepa_proposers_$p.tar
done
curl -LO https://github.com/hideinsewer/WARP-JEPA/releases/download/v1.0/SHA256SUMS && sha256sum -c SHA256SUMS
for p in l2 l3 l4; do tar -xf warp_jepa_proposers_$p.tar -C cost_probe/runs --strip-components=1; done
```

Naming: `*_bc005_*`/`*_bc1_*` = lambda_bc 0.05 (top level) / 1 (subordinate levels); no architecture tag or `mlp` =
residual MLP, `tf_ar_wm` = world-model-in-the-loop Transformer. In the L2 archive, `*_h200t211` are the MLP proposers
(MM), `*_tf_ar_wm_*_h200tf2` the 1-layer Transformers (TM uses `upper/<env>_up_tf_ar_wm_bc005_s<S>_h200tf2` with
`decoder/<env>_l1dec_bc1_s<S>_h200t211`; TT also uses `decoder/<env>_l1dec_tf_ar_wm_bc1_s<S>_h200tf2`), and
`*_h200tf1` the 4-layer Push-T ablation variants (`tf_parallel`, `tf_ar_action`, `tf_ar_wm`).

The proposers were trained on our own H-JEPA world models (official configurations, seeds 42-44, one H200 each).
`CHECKPOINTS_MANIFEST.json` lists the sha256 of every proposer and of the world model it was trained on. They can be
loaded with any H-JEPA checkpoint of the same configuration; the planners then print a warning instead of failing.
Proposers act in the latent space of their world model, so with a different checkpoint (e.g. one you retrain) expect
lower success; retrain the proposers on that checkpoint (sections 3 and 4) to reproduce the paper.

## 5. Analysis and diagnostics

- `compare_runs.py --reference RUN --out FILE RUN...`: paired task-by-task comparison (Wilson intervals, exact
  McNemar test, Newcombe paired 95% interval) and the per-level time breakdown.
- `eval_upper_offline.py`, `eval_decoder_offline.py`: offline terminal cost and action MSE on validation segments
  (mechanism analysis).
- `check_decoder_parity.py`, `check_action_alignment.py`: bit-exact parity gates of proposer scoring against the
  planner cost and of cached actions against frames.
- `kernel_profile.py`: per-step wall time vs GPU busy time of the GD planner.
- `env_manifest.py`: digest of an installed environment, for comparing two machines.

## Citation

```bibtex
@article{li2026warpjepa,
  title   = {{WARP-JEPA}: World-Model-in-the-Loop Amortized Rollout Planning for Frozen Hierarchical Joint-Embedding Predictive Architectures},
  author  = {Li, Junjie and Zhou, Zhaojiacheng and Wang, Kaixiang and Wang, Zihan and Zhao, QingSong and Lou, Jiong and Li, Jie},
  journal = {arXiv preprint},
  year    = {2026}
}
```
