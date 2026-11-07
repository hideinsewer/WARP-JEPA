"""Offline evaluation of the ActionDecoder vs the planner's own GD on cached frames."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

import decoder as D

VAL = ROOT / 'cost_probe/artifacts/data/cache/pusht_l1_val.h5'
GD_KS = (1, 2, 5, 10, 30)
REFINE_RS = (1, 2, 5)


def gd_batch(parts, z1, g2, m, n_steps, num_samples, init=None):
    """Released level-1 GradientSolver on a batch of envs; returns best (B,2,10) and best cost."""
    solver = parts.solver.level_solvers[1]
    saved = (solver.n_steps, solver.early_stop_patience, solver.batch_size,
             solver._n_envs, solver.num_samples)
    solver.n_steps, solver.early_stop_patience = n_steps, 0
    solver.batch_size, solver._n_envs, solver.num_samples = 64, z1.shape[0], num_samples
    info = {'pixels': z1.new_zeros(z1.shape[0], 1, 1), 'embed_0': z1[:, None],
            'goal_embed_0': g2[:, None], 'goal_action': m[:, None],
            'goal_embed_0_in_upper_space': True,
            'action_cost_weight': parts.action_cost_weight,
            'cost_last_n': parts.cost_last_n,
            'intermediate_cost_weight': parts.intermediate_cost_weight}
    if init is not None:
        orig = solver.init_action
        solver.init_action = lambda *a, **k: setattr(solver, 'init',
                                                     torch.nn.Parameter(init.clone()))
        try:
            out = solver.solve(info_dict=info, planning_horizon=2, steps_taken=0, eval_budget=None)
        finally:
            solver.init_action = orig
    else:
        out = solver.solve(info_dict=info, planning_horizon=2, steps_taken=0, eval_budget=None)
    solver.n_steps, solver.early_stop_patience, solver.batch_size, solver._n_envs, solver.num_samples = saved
    best = out['actions'].to(parts.device)
    return best, parts.level1_cost(z1, g2, m, best[:, None])[:, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gd-only', action='store_true')
    p.add_argument('--batch', type=int, default=256)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    parts = D.PlannerParts('cuda')
    val = D.CacheData(VAL, 'cuda', holdout_every=None, scaler=parts.action_scaler)
    n = len(val.valid)
    dec = None
    if not args.gd_only:
        ck = torch.load(args.decoder, map_location='cuda', weights_only=False)
        dec = D.ActionDecoder(heads=ck['args'].get('heads', 16), width=ck['args'].get('width', 1024),
                              blocks=ck['args'].get('blocks', 4)).to('cuda')
        dec.load_state_dict(ck['state_dict'])
        dec.eval()
    gen = torch.Generator(device='cuda').manual_seed(0)
    results = {}
    for cond in ('data', 'prior'):
        rows = {f'gd_{k}': [] for k in GD_KS}
        if dec is not None:
            rows.update({'decoder': [], **{f'dec_r{r}': [] for r in REFINE_RS}})
        costs_gd10, bc = [], []
        for start in range(0, n, args.batch):
            idx = val.valid[start:start + args.batch]
            z1 = val.embed[idx]
            z1_seq = val.embed[idx[:, None] + torch.tensor([0, 5, 10], device='cuda')]
            blocks = val.normalized_blocks(idx)
            with torch.no_grad():
                m = (parts.data_macro(z1_seq, blocks) if cond == 'data' else
                     torch.minimum(torch.maximum(
                         parts.prior_mean + parts.prior_std * torch.randn(
                         len(idx), 4, device='cuda', generator=gen), parts.l2_lo), parts.l2_hi))
                z2 = parts.encode_level2(z1)
                g2 = parts.subgoal(z2, m)
            for k in GD_KS:
                best, cost = gd_batch(parts, z1, g2, m, k, 50)
                rows[f'gd_{k}'].append(cost)
                if k == 10:
                    costs_gd10.append(cost)
            if dec is not None:
                with torch.no_grad():
                    out = dec(z1, z2, g2, m)
                    dcost = parts.level1_cost(z1, g2, m, out)
                    rows['decoder'].append(dcost.min(1).values)
                    bc.append(float(D.bc_loss(out, blocks, torch.ones(len(idx), dtype=torch.bool, device='cuda'))))
                for r in REFINE_RS:
                    best, cost = gd_batch(parts, z1, g2, m, r, dec.heads, init=out)
                    rows[f'dec_r{r}'].append(cost)
        for k in rows:
            rows[k] = torch.cat(rows[k])
        ref = torch.cat(costs_gd10)
        results[cond] = {k: dict(mean=float(v.mean()), median=float(v.median()),
                                 frac_le_gd10=float((v <= ref).float().mean()))
                         for k, v in rows.items()}
        if dec is not None:
            results[cond]['decoder']['bc_sse_mean'] = float(np.mean(bc))
    # batch-1 latency: decoder vs GD on 50 single samples
    lat = {}
    gen1 = torch.Generator(device='cuda').manual_seed(0)
    idx = val.valid[:50]
    z1 = val.embed[idx]; z1_seq = val.embed[idx[:, None] + torch.tensor([0, 5, 10], device='cuda')]
    with torch.no_grad():
        m = parts.data_macro(z1_seq, val.normalized_blocks(idx))
        z2 = parts.encode_level2(z1); g2 = parts.subgoal(z2, m)
    for name, fn in ((f'gd_{k}', (lambda k=k: gd_batch(parts, z1[i:i+1], g2[i:i+1], m[i:i+1], k, 50))) for k in GD_KS):
        ts = []
        for i in range(50):
            torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t))
        lat[name] = float(np.median(ts))
    if dec is not None:
        ts = []
        for i in range(50):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                dc = dec(z1[i:i+1], z2[i:i+1], g2[i:i+1], m[i:i+1])
                parts.level1_cost(z1[i:i+1], g2[i:i+1], m[i:i+1], dc)
            torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t))
        lat['decoder'] = float(np.median(ts))
    results['latency_ms_median'] = lat
    results['n_val_samples'] = n
    args.out.joinpath('offline_eval.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:2500])


if __name__ == '__main__':
    main()
