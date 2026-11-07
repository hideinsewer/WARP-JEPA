"""Verify cache action[t] is the action executed at frame t.

Rolls out the frozen level-1 world model from z1=embed[t] with blocks built from
normalized actions t+o..t+o+9 for o in {-1,0,+1} and compares the predicted
latents against cached embed[t+5] and embed[t+10]. PASS iff offset 0 wins both.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

import decoder as D

VAL = ROOT / 'cost_probe/artifacts/data/cache/pusht_l1_val.h5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=4096)
    args = parser.parse_args()
    parts = D.PlannerParts('cuda')
    with h5py.File(VAL, 'r') as f:
        embed = torch.from_numpy(f['embed'][:]).to('cuda')
        action = torch.from_numpy(f['action'][:]).to('cuda')
        episode = torch.from_numpy(np.asarray(f['episode'][:])).to('cuda')
    t = torch.arange(1, len(embed) - 11, device='cuda')
    ok = (episode[t - 1] == episode[t]) & (episode[t + 11] == episode[t])
    valid = t[ok]
    gen = torch.Generator(device='cuda').manual_seed(0)
    pick = valid[torch.randint(len(valid), (min(args.n, len(valid)),), device='cuda', generator=gen)]
    z1 = embed[pick]
    tgt5, tgt10 = embed[pick + 5], embed[pick + 10]
    mean, std = parts.action_mean, parts.action_std
    results = {}
    for o in (-1, 0, 1):
        raw = action[pick[:, None] + o + torch.arange(10, device='cuda')]
        blocks = ((raw - mean) / std).reshape(-1, 2, 10)
        info = {'pixels': z1.new_zeros(len(pick), 1, 1), 'embed_0': z1[:, None, None]}
        with torch.no_grad():
            pred = parts.l1_cost_model.rollout(info, blocks[:, None])['predicted_embed_0'][:, 0]
        results[o] = {'mse_t5': float(((pred[:, 1] - tgt5) ** 2).mean()),
                      'mse_t10': float(((pred[:, 2] - tgt10) ** 2).mean())}
    passed = (results[0]['mse_t5'] < results[-1]['mse_t5'] and results[0]['mse_t5'] < results[1]['mse_t5'] and
              results[0]['mse_t10'] < results[-1]['mse_t10'] and results[0]['mse_t10'] < results[1]['mse_t10'])
    out = dict(n=len(pick), offsets=results, passed=passed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print('ALIGNMENT', json.dumps(out))


if __name__ == '__main__':
    main()
