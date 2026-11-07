"""Dump a Push-T eval task set through the official planning_eval dump-only path.

run_planning_eval with cfg.dump_eval_trajs_path samples stratified eval starts from
the dataset (seeded by cfg.seed), writes the task payload and returns without
running any episode. The output file is refused if it already exists.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'h_jepa')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MUJOCO_GL', 'egl')

import torch  # noqa: E402  (after sys.path/env setup, like profile_cost.py)

import planning_eval as evaluation  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--dataset', default='pusht_expert_val')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'{args.out} already exists')
    args.out = args.out.expanduser().resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / 'cost_probe/artifacts'
    cfg = evaluation.load_eval_config('pusht_l2')
    cfg.cache_dir = str(artifacts / 'data')  # HDF5Dataset resolves <cache_dir>/<dataset_name>.h5
    cfg.policy = str(artifacts / 'models/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt')
    cfg.eval.dataset_name = args.dataset
    cfg.eval.num_eval = args.num_eval
    cfg.seed = args.seed
    cfg.load_eval_trajs_path = None  # dumping samples from the dataset, not a saved file
    cfg.dump_eval_trajs_path = str(args.out)
    metrics = evaluation.run_planning_eval(cfg, results_dir=args.out.parent / f'dump_cfg_seed{args.seed}')
    print('METRICS', metrics, flush=True)
    print('TASKS_WRITTEN', args.out, 'sha256', sha256(args.out), flush=True)


if __name__ == '__main__':
    main()
