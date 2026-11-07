"""Offline evaluation of UpperProposal checkpoints vs the planner's own level-2 GD.

Problems come from the validation cache (the 21 episodes the evaluation tasks are drawn
from). For each level-2 horizon h in 2..8: --per-h frames t whose run covers 10h frames,
goal z2[t + 10h - 5] (the remaining distance of a planner call on the expert's pace),
expert plan = the data macros of t..t+10h-1. Every method's returned plan is scored with
the planner's level-2 cost. GD runs the released level-2 solver (400 samples, AdamW, action
noise, clip box), 64 problems per batch, model parameters frozen; gd_released keeps the
released budget and early stopping at one problem per batch (the first --released-per-h
problems of each h).
"""

import argparse
import contextlib
import io
import json
import subprocess
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

import decoder as D
import upper_proposal as U

VAL = ROOT / 'cost_probe/artifacts/data/cache/pusht_l1_val.h5'
GD_KS = (10, 30, 90)
REFINE_RS = (1, 2, 5)
LATENCY_HS = (2, 8)
LATENCY_N = 20


def level2_info(z2, g2, action_cost_weight):
    return {'pixels': z2.new_zeros(z2.shape[0], 1, 1), 'embed_0': z2[:, None], 'goal_embed_0': g2[:, None],
            'goal_embed_0_level': 2, 'goal_embed_0_in_upper_space': False,
            'dense_upper_goal_projection': False, 'action_cost_weight': action_cost_weight}


def gd2(parts, z2, g2, h, cost_kwargs, n_steps=None, init=None, batch_size=64):
    """The planner's level-2 GradientSolver on B problems -> best plans (B,h,4), their lean
    costs (B,), iterations per problem, wall seconds per problem. n_steps None keeps the
    released budget and early stopping; init (B,K,h,4) replaces the released initial
    candidates (num_samples becomes K)."""
    s = parts.solver.level_solvers[2]
    saved = (s.n_steps, s.early_stop_patience, s.batch_size, s._n_envs, s.num_samples)
    if n_steps is not None:
        s.n_steps, s.early_stop_patience = n_steps, 0
    s.batch_size, s._n_envs = batch_size, z2.shape[0]
    if init is not None:
        s.num_samples = init.shape[1]
        s.init_action = lambda *a, **k: setattr(s, 'init', torch.nn.Parameter(init.clone()))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = s.solve(info_dict=level2_info(z2, g2, cost_kwargs['action_cost_weight']),
                          planning_horizon=h, steps_taken=0, eval_budget=None)
    finally:
        s.n_steps, s.early_stop_patience, s.batch_size, s._n_envs, s.num_samples = saved
        if init is not None:
            del s.init_action
    record = s.solve_records[-1]
    best = out['actions'].to(parts.device)
    with torch.no_grad():
        cost = U.level2_cost(parts.l2_cost_model, z2, g2, best[:, None], **cost_kwargs)[:, 0]
    return best, cost, record['n_iters'], record['wall_clock_per_env']


def propose(net, parts, z2, g2, h, cost_kwargs):
    """Best of the network's K plans under the lean level-2 cost -> plans (B,K,h,4), best (B,h,4), cost (B,)."""
    with torch.no_grad():
        out = net(z2, g2, torch.full((len(z2),), h, dtype=torch.long, device=z2.device))[:, :, :h]
        best = U.level2_cost(parts.l2_cost_model, z2, g2, out, **cost_kwargs).min(1)
    return out, out[torch.arange(len(z2), device=z2.device), best.indices], best.values


def summarize(costs, plans, expert, dist, references, released_n, prior_std):
    """Per-method statistics; references: name -> costs on the same problems (gd_released:
    on the first released_n)."""
    err = ((plans - expert) / prior_std).pow(2).sum(-1)  # (n,h)
    row = dict(n=len(costs), mean=float(costs.mean()), median=float(costs.median()),
               p90=float(torch.quantile(costs, .9)),
               mean_relative=float((costs / dist.clamp_min(1.0)).mean()),
               bc_per_macro_median=float(err.mean(1).median()),
               bc_first_macro_median=float(err[:, 0].median()))
    for name in ('gd_10', 'gd_30', 'gd_90'):
        row[f'frac_le_{name}'] = float((costs <= references[name]).float().mean())
    row['frac_le_gd_released'] = float((costs[:released_n] <= references['gd_released']).float().mean())
    return row


def sync_ms(fn):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - begin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal', type=Path, action='append', default=[],
                   help='UpperProposal checkpoint (repeatable); labelled by its directory name')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-h', type=int, default=192)
    p.add_argument('--released-per-h', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    parts = D.PlannerParts('cuda')
    kw = U.level2_cost_kwargs(parts)
    data = U.L2Data(VAL, parts, holdout_every=None)
    nets = {}
    for path in args.proposal:
        label = path.resolve().parent.name
        assert label not in nets, f'duplicate checkpoint label {label}'
        nets[label] = (*U.load_upper_proposal(path, 'cuda'), path)
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    results, pooled, saved = {}, {}, {}
    with sdpa_kernel(SDPBackend.MATH):
        for h in range(U.MIN_H, U.MAX_H + 1):
            pool = data.frames_with(data.span * h, 'all')
            t = pool[torch.randperm(len(pool), device='cuda', generator=gen)[:args.per_h]].sort().values
            z2, g2, expert = data.problems(t, h, torch.full_like(t, data.span * h - 5))
            assert bool(torch.isfinite(expert).all()), 'expert plan outside the run'
            n, m = len(t), min(args.released_per_h, len(t))
            dist = (g2 - z2).pow(2).sum(-1)
            costs, plans, extra = {}, {}, {}
            with torch.no_grad():
                plans['prior_mean'] = parts.prior_mean.expand(n, h, U.MACRO_DIM).clone()
                plans['expert'] = expert
                for name in ('prior_mean', 'expert'):
                    costs[name] = U.level2_cost(parts.l2_cost_model, z2, g2, plans[name][:, None], **kw)[:, 0]
            for k in GD_KS:
                plans[f'gd_{k}'], costs[f'gd_{k}'], _, seconds = gd2(parts, z2, g2, h, kw, n_steps=k)
                extra[f'gd_{k}'] = dict(seconds_per_batch_of_64=float(np.mean(seconds)))
            best, cost, iters, seconds = gd2(parts, z2[:m], g2[:m], h, kw, batch_size=1)
            released = dict(cost=cost, plan=best)
            extra['gd_released'] = dict(n=m, mean_iterations=float(np.mean(iters)),
                                        median_seconds=float(np.median(seconds)),
                                        mean=float(cost.mean()), median=float(cost.median()))
            for label, (net, _, _) in nets.items():
                out, plans[label], costs[label] = propose(net, parts, z2, g2, h, kw)
                for r in REFINE_RS:
                    plans[f'{label}_r{r}'], costs[f'{label}_r{r}'], _, _ = gd2(
                        parts, z2, g2, h, kw, n_steps=r, init=out)
            references = {**costs, 'gd_released': released['cost']}
            results[h] = {name: summarize(costs[name], plans[name], expert, dist, references, m,
                                          parts.prior_std)
                          for name in costs}
            for name, info in extra.items():
                results[h].setdefault(name, {}).update(info)
            results[h]['median_start_goal_distance'] = float(dist.median())
            saved[h] = dict(t=t.cpu(), dist=dist.cpu(), released_cost=released['cost'].cpu(),
                            released_iters=iters, **{f'cost/{k}': v.cpu() for k, v in costs.items()},
                            **{f'plan/{k}': v.cpu() for k, v in plans.items()})
            for name in costs:
                pooled.setdefault(name, []).append(costs[name])
            print('UPPER_OFFLINE', h, json.dumps({k: round(v['median'], 3) for k, v in results[h].items()
                                                   if isinstance(v, dict) and 'median' in v}), flush=True)
        pooled = {name: dict(mean=float(torch.cat(rows).mean()), median=float(torch.cat(rows).median()))
                  for name, rows in pooled.items()}
        latency = {}
        for h in LATENCY_HS:
            pool = data.frames_with(data.span * h, 'all')[:LATENCY_N]
            z2, g2, _ = data.problems(pool, h, torch.full_like(pool, data.span * h - 5))
            row = {}
            for label, (net, _, _) in nets.items():
                row[label] = float(np.median([sync_ms(lambda i=i: propose(net, parts, z2[i:i + 1], g2[i:i + 1], h, kw))
                                              for i in range(len(pool))][2:]))
            row['gd_10'] = float(np.median([sync_ms(lambda i=i: gd2(parts, z2[i:i + 1], g2[i:i + 1], h, kw, n_steps=10,
                                                                     batch_size=1))
                                            for i in range(len(pool))][2:]))
            latency[h] = row
    config = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  val_cache=D.relpath(VAL), val_cache_sha256=D.sha256(VAL), per_h=args.per_h,
                  released_per_h=args.released_per_h, seed=args.seed, gd_steps=GD_KS, refine_steps=REFINE_RS,
                  cost_kwargs=kw, gpu=torch.cuda.get_device_name(),
                  proposals={label: dict(path=D.relpath(path), sha256=D.sha256(path), args=ck['args'],
                                         git_commit=ck['git_commit'])
                             for label, (_, ck, path) in nets.items()})
    report = dict(config=config, by_horizon=results, pooled=pooled, latency_ms_median_batch1=latency)
    (args.out / 'offline_eval.json').write_text(json.dumps(report, indent=2, default=str))
    torch.save(saved, args.out / 'per_problem.pt')
    print('UPPER_OFFLINE_DONE', args.out / 'offline_eval.json', flush=True)


if __name__ == '__main__':
    main()
