"""Parity gate for the decoder's lean planner pieces (all checks must pass)."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe'), str(ROOT / 'h_jepa')]

import decoder as D
import planning_eval as evaluation
import profile_cost as pc
from hierarchical_solver import HierarchicalSolver
from stable_worldmodel.solver.gd import GradientSolver

CAPTURED = ROOT / 'cost_probe/runs/pusht_l2_4090_a0_ref/captured_cost.pt'
VAL = ROOT / 'cost_probe/artifacts/data/cache/pusht_l1_val.h5'
TRAIN = ROOT / 'cost_probe/artifacts/data/cache/pusht_l1_train.h5'


def gate_a(parts):
    cap = torch.load(CAPTURED, map_location='cpu', weights_only=False)
    info, actions = cap['info'], cap['actions']
    z1 = info['embed_0'][:, 0, -1].to(parts.device)
    g2 = info['goal_embed_0'][:, 0, 0].to(parts.device)
    m = info['goal_action'][:, 0, 0].to(parts.device)
    cand = actions.to(parts.device)
    lean = parts.level1_cost(z1, g2, m, cand)
    ref = parts.l1_cost_model.get_cost({k: (v.to(parts.device) if torch.is_tensor(v) else v)
                                        for k, v in info.items()}, cand)
    return dict(bit_exact=bool(torch.equal(lean, ref)),
                max_abs_diff=float((lean - ref).abs().max()))


def gate_b(parts):
    cfg = parts.cfg
    timing = pc.Timing()
    captured = {}
    inner = GradientSolver.solve

    def record(solver, info_dict, *a, **k):
        out = inner(solver, info_dict, *a, **k)
        if solver.level == 1:
            for key in ('embed_0', 'goal_embed_0', 'goal_action'):
                captured[key] = info_dict[key].detach().clone()
        else:
            captured['l2_actions'] = out['actions'].detach().clone()
        return out

    import unittest.mock as um
    cfg.eval.num_eval, cfg.eval.chunk_size = 1, 1
    cfg.load_eval_trajs_path = str(ROOT / 'cost_probe/artifacts/data/eval_trajs/pusht/goal_offset_75_val.pt')
    patches = timing.install()
    for p in patches:
        p.start()
    try:
        import tempfile
        evaluation.run_planning_eval(cfg, model=parts.model,
                                     results_dir=Path(tempfile.mkdtemp(prefix='decoder_parity_')))
    finally:
        for p in patches:
            p.stop()
    info, steps, budget, action_space, plan_config = timing.first_plan
    info = {k: (v.to(parts.device) if torch.is_tensor(v) else v) for k, v in info.items()}
    solver = evaluation.build_solver(cfg, parts.model)
    solver.configure(action_space=action_space, n_envs=1, config=plan_config)
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH), \
            um.patch.object(GradientSolver, 'solve', record):
        solver.solve(pc.detached(info), steps, budget)
    z1 = captured['embed_0'].to(parts.device).reshape(1, -1)
    g2_info = captured['goal_embed_0'].to(parts.device).reshape(1, -1)
    m_info = captured['goal_action'].to(parts.device).reshape(1, -1)
    m_l2 = captured['l2_actions'][0, :1].to(parts.device)
    with sdpa_kernel(SDPBackend.MATH):
        z2 = parts.encode_level2(z1)
        # embed_2 from the real hierarchical encode
        enc = solver._encode_all_levels(info, key='pixels')
        d_z2 = float((z2 - enc['embed_2'].reshape(1, -1)).abs().max())
        g2 = parts.subgoal(z2, m_l2)
        d_g2 = float((g2 - g2_info).abs().max())
        d_m = float((m_l2 - m_info).abs().max())
    return dict(bit_exact=(d_z2 == 0.0 and d_g2 == 0.0 and d_m == 0.0),
                max_abs_diff=dict(z2=d_z2, g2=d_g2, m=d_m))


def gate_d(parts):
    data = D.CacheData(TRAIN, parts.device, holdout_every=10**9, holdout=False)
    z1, z1_seq, blocks, _ = data.sample_batch(4096, torch.Generator(device=data.device).manual_seed(0))
    with torch.no_grad():
        m = parts.data_macro(z1_seq, blocks)
    inside = ((m >= parts.l2_lo) & (m <= parts.l2_hi)).float().mean(0)
    return dict(fraction_inside_bounds_per_dim=inside.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    parts = D.PlannerParts('cuda')
    res = {}
    res['a_captured_cost'] = gate_a(parts)
    res['b_planner_capture'] = gate_b(parts)
    res['c_batched_consistency'] = gate_c(parts)
    res['d_data_macro_bounds'] = gate_d(parts)
    res['passed'] = (res['a_captured_cost']['bit_exact'] and res['b_planner_capture']['bit_exact']
                     and res['c_batched_consistency']['passed'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print('PARITY', json.dumps(res, default=float)[:800])


def gate_c(parts):
    data = D.CacheData(VAL, parts.device, holdout_every=10**9, holdout=False)
    gen = torch.Generator(device=data.device).manual_seed(0)
    z1, z1_seq, blocks, _ = data.sample_batch(2, gen)
    cand = torch.randn(2, 4, 2, 10, device=z1.device, generator=gen)
    with torch.no_grad():
        m = parts.data_macro(z1_seq, blocks)
        g2 = parts.subgoal(parts.encode_level2(z1), m)
        both = parts.level1_cost(z1, g2, m, cand)
        single = torch.cat([parts.level1_cost(z1[i:i+1], g2[i:i+1], m[i:i+1], cand[i:i+1])
                            for i in range(2)])
    diff = (both - single).abs().max()
    rel = (diff / single.abs().max()).item()
    return dict(max_abs_diff=float(diff), max_rel_diff=float(rel), passed=rel <= 1e-5)


if __name__ == '__main__':
    main()
