"""Measure unmodified two-level Push-T planning and action-matching overhead."""

import argparse
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'h_jepa')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn.attention import SDPBackend, sdpa_kernel

import planning_eval as evaluation
import decoder
import transformer_heads
import upper_proposal
from hierarchical_solver import HierarchicalSolver, LevelCostModel
from stable_worldmodel.solver.gd import GradientSolver


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def detached(info):
    return {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in info.items()}


MACRO_RULE = 'level-1 solve without goal_embed_0_level (the level above planned)'

OFFICIAL_TASK_FILE = ROOT / 'cost_probe/artifacts/data/eval_trajs/pusht/goal_offset_75_val.pt'
OFFICIAL_TASK_SHA256 = '588e8db4a6be13e41f00b83332906ac91bc6fda8e3e211a06c8a31431b0d0742'
TASK_FILES = {
    'pusht': OFFICIAL_TASK_FILE,
    'fourroom': ROOT / 'cost_probe/artifacts/data/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt',
    'cube': ROOT / 'cost_probe/artifacts/data/eval_trajs/ogbench/goal_offset_20_pickup_val.pt',
    'ant': ROOT / 'cost_probe/artifacts/data/eval_trajs/ant/expert_grid_d3_n50.pt',
}


def has_upper_plan(info_dict):
    """HierarchicalSolver sets goal_embed_0_level only when no level above planned; the world's
    dataset goal_action is present in every solve, so it cannot identify macro-actions."""
    return 'goal_embed_0_level' not in info_dict


class Timing:
    """Synchronize only at solver boundaries; capture microbench inputs in warmup."""

    def __init__(self):
        self.enabled = False
        self.plans = []
        self.levels = []
        self.first_plan = None
        self.low_cost = None

    def install(self):
        original_plan = HierarchicalSolver.solve
        original_level = GradientSolver.solve
        original_cost = LevelCostModel.get_cost
        original_encode = HierarchicalSolver._encode_all_levels
        self.encode_seconds = 0.0

        def plan(solver, info_dict, steps_taken, eval_budget):
            if self.first_plan is None:
                self.first_plan = (detached(info_dict), steps_taken, eval_budget,
                                   solver.level_solvers[1]._action_space,
                                   copy.deepcopy(solver._config))
            if not self.enabled:
                return original_plan(solver, info_dict, steps_taken, eval_budget)
            first = len(self.levels)
            encoding_before = self.encode_seconds
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original_plan(solver, info_dict, steps_taken, eval_budget)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            row = dict(planner_seed=solver.level_solvers[1].torch_gen.initial_seed(),
                       steps_taken=steps_taken, seconds=seconds,
                       encoding_seconds=self.encode_seconds - encoding_before)
            for level in (1, 2):
                row[f'level{level}_seconds'] = sum(
                    x['seconds'] for x in self.levels[first:] if x['level'] == level)
            row['other_seconds'] = seconds - sum(
                row[k] for k in ('encoding_seconds', 'level1_seconds', 'level2_seconds'))
            assert row['other_seconds'] >= -1e-6
            self.plans.append(row)
            return result

        def level(solver, *args, **kwargs):
            if not self.enabled:
                return original_level(solver, *args, **kwargs)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original_level(solver, *args, **kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            info = kwargs['info_dict'] if 'info_dict' in kwargs else args[0]
            self.levels.append({**solver.solve_records[-1],
                                'macro': solver.level == 1 and has_upper_plan(info),
                                'seconds': seconds})
            return result

        def cost(model, info_dict, actions):
            if self.low_cost is None and model.level == 1 and has_upper_plan(info_dict) \
                    and actions.shape[0] == 1:
                self.low_cost = (model, detached(info_dict), actions.detach().clone())
            return original_cost(model, info_dict, actions)

        def encode(solver, *args, **kwargs):
            if not self.enabled:
                return original_encode(solver, *args, **kwargs)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original_encode(solver, *args, **kwargs)
            torch.cuda.synchronize()
            self.encode_seconds += time.perf_counter() - begin
            return result

        self.original_plan = original_plan
        return [patch.object(HierarchicalSolver, 'solve', plan),
                patch.object(GradientSolver, 'solve', level),
                patch.object(LevelCostModel, 'get_cost', cost),
                patch.object(HierarchicalSolver, '_encode_all_levels', encode)]

    def clear(self):
        self.plans.clear()
        self.levels.clear()
        self.encode_seconds = 0.0


def check_timing_parity(cfg, model, timing):
    info, steps, budget, action_space, plan_config = timing.first_plan
    outputs, records, seconds = [], [], []
    for measured in (False, True):
        solver = evaluation.build_solver(cfg, model)
        solver.configure(action_space=action_space, n_envs=1, config=plan_config)
        timing.enabled = measured
        torch.cuda.synchronize()
        begin = time.perf_counter()
        outputs.append(solver.solve(detached(info), steps, budget)['actions'])
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - begin)
        records.append([(level, item['n_iters'])
                        for level, s in solver.level_solvers.items()
                        for item in s.solve_records])
    assert torch.equal(*outputs), 'Timing changed planned actions.'
    assert records[0] == records[1], 'Timing changed early stopping.'
    timing.clear()
    return dict(actions_bit_exact=True, iterations_identical=True,
                plain_seconds=seconds[0], instrumented_seconds=seconds[1],
                relative_timing_difference=seconds[1] / seconds[0] - 1)


def macro_budget(original, iterations, bypass_iterations=None):
    """Fixed level-1 GD budgets: `iterations` for solves with a macro-action,
    `bypass_iterations` for solves without an upper plan; each disables early stopping."""

    def solve(solver, info_dict, *args, **kwargs):
        if solver.level != 1:
            return original(solver, info_dict, *args, **kwargs)
        budget = iterations if has_upper_plan(info_dict) else bypass_iterations
        if budget is None:
            return original(solver, info_dict, *args, **kwargs)
        saved = solver.n_steps, solver.early_stop_patience
        solver.n_steps, solver.early_stop_patience = budget, 0
        try:
            return original(solver, info_dict, *args, **kwargs)
        finally:
            solver.n_steps, solver.early_stop_patience = saved

    return solve


def apply_upper_overrides(cfg, n_steps=None, num_samples=None):
    """Fix the level-2 GD step count and sample count in config, before solvers are built."""
    with open_dict(cfg.solver.solvers.level2) as level2:
        if n_steps is not None:
            level2.n_steps = n_steps
            level2.early_stop_patience = 0
        if num_samples is not None:
            level2.num_samples = num_samples


def decoder_solve(original, dec):
    """Route macro-conditioned level-1 solves to the decoder's best-of-K.

    Level 2, bypass level-1, and any horizon != 2 keep calling the wrapped
    solve. On the first decoder call, the same candidates are also scored
    through the planner's full expanded info; the max |diff| to the lean
    costs is recorded on the wrapper as .check['max_abs_diff'].
    """
    state = {}

    def level(solver, info_dict, *args, **kwargs):
        if solver.level != 1 or not has_upper_plan(info_dict) \
                or kwargs.get('planning_horizon') != 2:
            return original(solver, info_dict, *args, **kwargs)
        t0 = time.time()
        b = info_dict['embed_0'].shape[0]
        dev = solver.device
        z1, g2, m = (info_dict['embed_0'][:, -1:].to(dev), info_dict['goal_embed_0'][:, 0:1].to(dev),
                     info_dict['goal_action'][:, 0:1].to(dev))
        assert z1.shape[:2] == g2.shape[:2] == (b, 1) and z1.shape[2] == g2.shape[2] \
                and m.shape[:2] == (b, 1) and m.shape[2] == 4
        z1, g2, m = z1[:, 0], g2[:, 0], m[:, 0]
        arch = getattr(dec, 'architecture', 'mlp')
        with torch.no_grad():
            z2 = decoder.encode_level2(solver.model.upper_models[0], z1)
            if arch == 'mlp':
                t_head0 = time.perf_counter()
                cand = dec(z1, z2, g2, m)
                head_s, wm_s = time.perf_counter() - t_head0, 0.0
                t_cost0 = time.perf_counter()
                costs = decoder.level1_cost(
                    solver.model, z1, g2, m, cand,
                    action_cost_weight=info_dict['action_cost_weight'],
                    cost_last_n=solver._config.cost_last_n,
                    intermediate_cost_weight=solver._config.intermediate_cost_weight)
                cost_s = time.perf_counter() - t_cost0
            else:
                cand, pred_emb, gen_stats = dec.generate(z1, z2, g2, m, cost_model=solver.model)
                head_s, wm_s = gen_stats['head_seconds'], gen_stats['wm_rollout_seconds']
                t_cost0 = time.perf_counter()
                costs = (transformer_heads.level1_cost_from_rollout(
                    solver.model, z1, g2, m, cand, pred_emb,
                    action_cost_weight=info_dict['action_cost_weight'],
                    cost_last_n=solver._config.cost_last_n,
                    intermediate_cost_weight=solver._config.intermediate_cost_weight)
                    if pred_emb is not None else decoder.level1_cost(
                        solver.model, z1, g2, m, cand,
                        action_cost_weight=info_dict['action_cost_weight'],
                        cost_last_n=solver._config.cost_last_n,
                        intermediate_cost_weight=solver._config.intermediate_cost_weight))
                cost_s = time.perf_counter() - t_cost0
        if 'max_abs_diff' not in state:
            expanded = solver._expand_info_dict_for_samples(
                dict(info_dict), start_idx=0, end_idx=solver.n_envs,
                num_samples=dec.heads)
            expanded['cost_last_n'] = solver._config.cost_last_n
            expanded['intermediate_cost_weight'] = solver._config.intermediate_cost_weight
            with torch.no_grad():
                full = solver.model.get_cost(expanded, cand)
            state['max_abs_diff'] = float((full - costs).abs().max())
        best = cand[torch.arange(b), costs.argmin(dim=1)]
        actions = best.detach().cpu()
        predictions = solver._rollout_final_actions(info_dict, actions)
        elapsed = time.time() - t0
        solver.solve_records.append(dict(level=1, horizon=2, num_samples=dec.heads,
                                         n_iters=[0] * b, wall_clock=elapsed,
                                         wall_clock_per_env=[elapsed] * b, decoder=True,
                                         decoder_architecture=arch,
                                         head_seconds=head_s, wm_rollout_seconds=wm_s,
                                         cost_seconds=cost_s,
                                         decoder_best_cost=costs.min(dim=1).values.tolist()))
        return dict(actions=actions, predictions=predictions)

    level.check = state
    return level


def capture_planning_context(original_encode, context):
    """Remember the HierarchicalSolver and the level-1 embedding of its current observation."""

    def encode(solver, *args, **kwargs):
        result = original_encode(solver, *args, **kwargs)
        if kwargs.get('key', args[1] if len(args) > 1 else 'pixels') == 'pixels':
            context['hsolver'], context['embed_1'] = solver, result['embed_1']
        return result

    return encode


def coupled_costs(dec, context, l2_model, z2, g2, cand, cost_kwargs):
    """Level-2 cost of each candidate plan after the level-1 decoder executes its first macro.

    For a candidate with first macro m and first predicted latent s = P2(z2, m) (the subgoal the
    level-1 solve would receive), the decoder's best plan toward (s, m) under the planner's
    level-1 cost is rolled out by the level-1 model from the current z1; the end latent is
    encoded to level 2 and the candidate's remaining macros are rolled out from there toward g2.
    cand (B,C,h,4) -> remaining costs (B,C), level-1 costs of the decoded plans (B,C).
    """
    hsolver = context['hsolver']
    low_solver = hsolver.level_solvers[1]
    b, c, h = cand.shape[:3]
    flat = lambda x: x.reshape(b * c, *x.shape[2:])
    sub = l2_model.rollout({'pixels': z2.new_zeros(b, c, 1), 'embed_0': z2[:, None, None].expand(b, c, 1, -1)},
                           cand)['predicted_embed_0'][:, :, 1]
    z1 = flat(context['embed_1'][:, -1][:, None].expand(b, c, -1))
    z2c, sub, m = flat(z2[:, None].expand(b, c, -1)), flat(sub), flat(cand[:, :, 0])
    low = (dec(z1, z2c, sub, m, cost_model=low_solver.model)
           if getattr(dec, 'architecture', 'mlp') != 'mlp' else dec(z1, z2c, sub, m))
    low_costs = decoder.level1_cost(
        low_solver.model, z1, sub, m, low,
        action_cost_weight=float(hsolver._level_plan_config(1).action_cost_weight),
        cost_last_n=low_solver._config.cost_last_n,
        intermediate_cost_weight=low_solver._config.intermediate_cost_weight)
    best = low_costs.min(dim=1)
    plan = low[torch.arange(b * c, device=low.device), best.indices]
    end = low_solver.model.rollout({'pixels': z1.new_zeros(b * c, 1, 1), 'embed_0': z1[:, None, None]},
                                   plan[:, None])['predicted_embed_0'][:, 0, -1]
    z2_end = decoder.encode_level2(hsolver.level_models[1], end)
    rest = upper_proposal.level2_cost(l2_model, z2_end, flat(g2[:, None].expand(b, c, -1)),
                                      flat(cand)[:, None, 1:], **cost_kwargs)[:, 0]
    return rest.view(b, c), best.values.view(b, c)


def upper_proposal_solve(original, net, refine_iters=0, couple=None):
    """Route level-2 solves to the proposal network.

    The candidates are the network's K plans plus, when the warm-start wrapper passes
    init_action (the previous plan shifted by one macro-action), that plan as candidate K.
    Without refinement the candidate with the lowest planner level-2 cost is returned; with
    refine_iters R the wrapped solve runs exactly R GD steps, without early stopping, from all
    candidates. With couple = (level-1 decoder, planning context) the candidate with the lowest
    coupled_costs remaining cost is returned instead (no refinement). On the first call the lean
    costs are also computed through the planner's full expanded info; the max |diff| is recorded
    on the wrapper as .check['max_abs_diff'] (with couple, also the max |diff| between z2 and the
    level-2 encoding of the context's z1 as .check['context_z2_max_abs_diff']).
    """
    assert not (refine_iters and couple), 'coupled selection does not refine'
    state = {}

    def solve(solver, info_dict, *args, **kwargs):
        if solver.level != 2:
            return original(solver, info_dict, *args, **kwargs)
        t0 = time.time()
        h = int(kwargs['planning_horizon'])
        b, dev = solver.n_envs, solver.device
        z2, g2 = info_dict['embed_0'][:, -1].to(dev), info_dict['goal_embed_0'][:, -1].to(dev)
        assert z2.shape == g2.shape and z2.shape[0] == b and z2.ndim == 2
        init = kwargs.pop('init_action', None)
        cost_kwargs = dict(action_cost_weight=float(info_dict['action_cost_weight']),
                           cost_last_n=max(1, int(solver._config.cost_last_n)),
                           intermediate_cost_weight=float(solver._config.intermediate_cost_weight))
        arch = getattr(net, 'architecture', 'mlp')
        with torch.no_grad():
            if arch == 'mlp':
                t_head0 = time.perf_counter()
                cand = net(z2, g2, torch.full((b,), h, dtype=torch.long, device=dev))[:, :, :h]
                head_s, wm_s = time.perf_counter() - t_head0, 0.0
                if init is not None:
                    assert init.shape == (b, h, cand.shape[-1]), (init.shape, h)
                    cand = torch.cat([cand, init.to(dev, cand.dtype)[:, None]], dim=1)
                t_cost0 = time.perf_counter()
                costs = upper_proposal.level2_cost(solver.model, z2, g2, cand, **cost_kwargs)
                cost_s = time.perf_counter() - t_cost0
            else:
                if init is not None:
                    assert init.shape == (b, h, 4), (init.shape, h)
                cand, pred_emb, gen_stats = net.generate(
                    z2, g2, torch.full((b,), h, dtype=torch.long, device=dev),
                    cost_model=solver.model,
                    extra_actions=init[:, None] if init is not None else None)
                head_s, wm_s = gen_stats['head_seconds'], gen_stats['wm_rollout_seconds']
                t_cost0 = time.perf_counter()
                costs = (transformer_heads.level2_cost_from_rollout(
                    solver.model, z2, g2, cand, pred_emb, **cost_kwargs)
                    if pred_emb is not None
                    else upper_proposal.level2_cost(solver.model, z2, g2, cand, **cost_kwargs))
                cost_s = time.perf_counter() - t_cost0
        if 'max_abs_diff' not in state:
            expanded = solver._expand_info_dict_for_samples(
                dict(info_dict), start_idx=0, end_idx=b, num_samples=cand.shape[1])
            expanded['cost_last_n'] = cost_kwargs['cost_last_n']
            expanded['intermediate_cost_weight'] = cost_kwargs['intermediate_cost_weight']
            with torch.no_grad():
                full = solver.model.get_cost(expanded, cand)
            state['max_abs_diff'] = float((full - costs).abs().max())
        chosen = costs.argmin(dim=1)
        record = dict(proposal=True, proposal_architecture=arch,
                      proposal_candidates=int(cand.shape[1]),
                      proposal_warm_candidate=init is not None,
                      proposal_chosen=chosen.tolist(),
                      proposal_chosen_warm=[init is not None and c == cand.shape[1] - 1
                                            for c in chosen.tolist()],
                      head_seconds=head_s, wm_rollout_seconds=wm_s, cost_seconds=cost_s,
                      proposal_best_cost=costs.min(dim=1).values.tolist())
        if couple is not None:
            dec, context = couple
            if 'context_z2_max_abs_diff' not in state:
                z2_context = decoder.encode_level2(context['hsolver'].level_models[1],
                                                   context['embed_1'][:, -1].to(dev))
                state['context_z2_max_abs_diff'] = float((z2_context - z2).abs().max())
            with torch.no_grad():
                rest, low_costs = coupled_costs(dec, context, solver.model, z2, g2, cand, cost_kwargs)
            pick = rest.argmin(dim=1)
            rows = torch.arange(b, device=dev)
            record.update(coupled_chosen=pick.tolist(),
                          coupled_differs=(pick != chosen).tolist(),
                          coupled_chosen_warm=[init is not None and c == cand.shape[1] - 1
                                               for c in pick.tolist()],
                          coupled_rest_cost=rest[rows, pick].tolist(),
                          coupled_level2_cost=costs[rows, pick].tolist(),
                          coupled_level1_cost=low_costs[rows, pick].tolist(),
                          proposal_rest_cost=rest[rows, chosen].tolist(),
                          proposal_level1_cost=low_costs[rows, chosen].tolist())
            chosen = pick
        if refine_iters:
            saved = solver.n_steps, solver.early_stop_patience, solver.num_samples
            solver.n_steps, solver.early_stop_patience, solver.num_samples = refine_iters, 0, cand.shape[1]
            solver.init_action = lambda *a, **k: setattr(solver, 'init', torch.nn.Parameter(cand.clone()))
            try:
                result = original(solver, info_dict, *args, **kwargs)
            finally:
                solver.n_steps, solver.early_stop_patience, solver.num_samples = saved
                del solver.init_action
            solver.solve_records[-1].update(record)
            return result
        actions = cand[torch.arange(b, device=dev), chosen].detach().cpu()
        predictions = solver._rollout_final_actions(info_dict, actions)
        elapsed = time.time() - t0
        solver.solve_records.append(dict(level=2, horizon=h, num_samples=int(cand.shape[1]),
                                         n_iters=[0] * b, wall_clock=elapsed,
                                         wall_clock_per_env=[elapsed] * b, **record))
        return dict(actions=actions, predictions=predictions)

    solve.check = state
    return solve


def warm_start_level2(original, replanning_interval):
    """Warm-start level-2 solves: when the previous level-2 solve ran exactly one
    replanning interval earlier with the same batch size, start from its best plan
    shifted by one macro-action (init_action); otherwise keep released behavior."""
    previous = SimpleNamespace(steps_taken=None, actions=None)

    def solve(solver, info_dict, *args, **kwargs):
        if solver.level != 2:
            result = original(solver, info_dict, *args, **kwargs)
            solver.solve_records[-1]['warm_start'] = False
            return result
        steps_taken = kwargs.get('steps_taken')
        warm = (previous.actions is not None and steps_taken is not None
                and previous.steps_taken == steps_taken - replanning_interval
                and previous.actions.shape[0] == solver.n_envs)
        if warm:
            init = previous.actions[:, 1:]
            # Envs whose horizon schedule repeats values (fourroom) leave the
            # shifted plan one macro short of the new planning horizon; pad with
            # the action prior mean and truncate, as GradientSolver.init_action
            # does for init_action shorter/longer than planning_horizon.
            horizon = kwargs.get('planning_horizon')
            if horizon is not None and init.shape[1] != horizon:
                prior_stats = solver._get_action_prior_stats()
                base_mean = (torch.zeros(init.shape[-1], device=init.device)
                             if prior_stats is None else prior_stats[0].to(init.device))
                if init.shape[1] < horizon:
                    pad = base_mean.view(1, 1, -1).expand(
                        init.shape[0], horizon - init.shape[1], -1)
                    init = torch.cat([init, pad.to(init.dtype)], dim=1)
                else:
                    init = init[:, :horizon]
            kwargs['init_action'] = init
        result = original(solver, info_dict, *args, **kwargs)
        previous.steps_taken = steps_taken
        previous.actions = result['actions'].detach().clone()
        solver.solve_records[-1]['warm_start'] = warm
        return result

    return solve


def set_grad_flags(model, flags):
    for parameter, flag in zip(model.parameters(), flags, strict=True):
        parameter.requires_grad_(flag)
        parameter.grad = None


def action_grads_only(original_solve, original_backward):
    """Restrict each solver backward to the optimized actions; parameter flags stay unchanged."""
    state = SimpleNamespace(enabled=True, targets=[])

    def solve(solver, *args, **kwargs):
        if not state.enabled:
            return original_solve(solver, *args, **kwargs)
        optimizer_cls = solver.optimizer_cls

        def optimizer(params, **options):
            params = list(params)
            state.targets[:] = params
            return optimizer_cls(params, **options)

        solver.optimizer_cls = optimizer
        try:
            return original_solve(solver, *args, **kwargs)
        finally:
            solver.optimizer_cls = optimizer_cls
            state.targets.clear()

    def backward(tensor, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if inputs is None and state.targets:
            inputs = list(state.targets)
        return original_backward(tensor, gradient, retain_graph, create_graph, inputs)

    return state, solve, backward


def check_planner_parity(cfg, model, timing, use_variant):
    """Saved inputs under the released planner and the variant; use_variant(flag) returns
    whether a backward should be restricted to the actions."""
    info, steps, budget, action_space, plan_config = timing.first_plan
    cost_model, cost_info, candidates = timing.low_cost
    timing.enabled = False
    costs, gradients, actions, records, populated = [], [], [], [], []
    for variant in (False, True):
        restrict = use_variant(variant)
        for parameter in model.parameters():
            parameter.grad = None
        x = candidates.clone().requires_grad_(True)
        cost = cost_model.get_cost(dict(cost_info), x).sum()
        cost.backward(inputs=[x] if restrict else None)
        costs.append(cost.detach())
        gradients.append(x.grad)
        solver = evaluation.build_solver(cfg, model)
        solver.configure(action_space=action_space, n_envs=1, config=plan_config)
        actions.append(solver.solve(detached(info), steps, budget)['actions'])
        records.append([(level, item['n_iters'])
                        for level, s in solver.level_solvers.items()
                        for item in s.solve_records])
        populated.append(sum(p.numel() for p in model.parameters() if p.grad is not None))
    use_variant(True)
    for parameter in model.parameters():
        parameter.grad = None
    timing.clear()
    return dict(cost_bit_exact=torch.equal(*costs),
                action_gradient_bit_exact=torch.equal(*gradients),
                actions_bit_exact=torch.equal(*actions),
                iterations_identical=records[0] == records[1],
                cost_abs_difference=float((costs[0] - costs[1]).abs()),
                action_gradient_max_abs_difference=float((gradients[0] - gradients[1]).abs().max()),
                action_max_abs_difference=float((actions[0] - actions[1]).abs().max()),
                parameter_elements_with_grad=dict(released=populated[0], variant=populated[1]),
                solve_iterations=records)


def component_benchmark(captured, repeats=7, iterations=10, restrict_backward=False):
    model, info, candidates = captured
    assert model.upper_models[0].temporal_window_size == 1
    assert not info.get('dense_upper_goal_projection', False)
    assert candidates.shape[0] == 1 and candidates.shape[2] == 2
    with torch.no_grad():
        states = model.rollout(dict(info), candidates)['predicted_embed_0'].detach()
    weight = float(info['action_cost_weight'])

    def state_cost(x):
        return model.get_cost({**info, 'action_cost_weight': 0.0}, x).sum()

    def full_cost(x):
        return model.get_cost(dict(info), x).sum()

    def action_cost(x):
        encoded = model._encode_actions_up(states, x, pad=0, dense=False)
        return weight * model._action_cost(encoded, info).sum()

    x = candidates.clone().requires_grad_(True)
    full, state, action = full_cost(x), state_cost(x), action_cost(x)
    torch.testing.assert_close(full, state + action, rtol=1e-5, atol=1e-5)
    grad_full = torch.autograd.grad(full, x)[0]
    grad_split = torch.autograd.grad(state + action, x)[0]
    torch.testing.assert_close(grad_full, grad_split, rtol=1e-4, atol=1e-5)
    costs = dict(full=float(full.detach()), state=float(state.detach()),
                 action=float(action.detach()))

    single_info = {k: v[:, :1] if torch.is_tensor(v) and v.ndim >= 2
                   and v.shape[:2] == candidates.shape[:2] else v
                   for k, v in info.items()}

    def verify_one(x):
        with torch.no_grad():
            return model.get_cost({**single_info, 'action_cost_weight': 0.0}, x[:, :1]).sum()

    cases = {'full_forward_backward': (full_cost, True),
             'state_only_forward_backward': (state_cost, True),
             'action_match_forward_backward': (action_cost, True),
             'full_forward': (full_cost, False),
             'verify_one_forward': (verify_one, False)}
    samples = {name: [] for name in cases}

    def run_case(name, count):
        fn, backward = cases[name]
        for _ in range(count):
            x.grad = None
            loss = fn(x)
            if backward:
                loss.backward(inputs=[x] if restrict_backward else None)

    for name in cases:
        run_case(name, 3)
    for repeat in range(repeats):
        names = list(cases)
        if repeat % 2:
            names.reverse()
        for name in names:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            run_case(name, iterations)
            torch.cuda.synchronize()
            samples[name].append(1000 * (time.perf_counter() - begin) / iterations)
    result = {name: dict(median_ms=float(np.median(values)), samples_ms=values)
              for name, values in samples.items()}
    result['matching_increment_fraction'] = (
        1 - result['state_only_forward_backward']['median_ms']
        / result['full_forward_backward']['median_ms'])
    result.update(candidate_shape=list(candidates.shape), loss_components=costs,
                  loss_and_action_gradient_additivity_pass=True,
                  note='Fixed candidates, no optimizer or early stopping; subtraction is a local microbenchmark.')
    return result


def summarize(timing, evaluation_seconds):
    assert timing.plans and evaluation_seconds > 0
    total = sum(row['seconds'] for row in timing.plans)
    low = sum(row['level1_seconds'] for row in timing.plans)
    high = sum(row['level2_seconds'] for row in timing.plans)
    latencies = np.array([row['seconds'] for row in timing.plans])
    return dict(planning_calls=len(timing.plans), evaluation_seconds=evaluation_seconds,
                planning_seconds=total, lower_seconds=low, upper_seconds=high,
                encoding_seconds=sum(r['encoding_seconds'] for r in timing.plans),
                lower_fraction_planning=low / total,
                lower_fraction_evaluation=low / evaluation_seconds,
                zero_lower_cost_max_speedup=1 / (1 - low / evaluation_seconds),
                median_plan_seconds=float(np.median(latencies)),
                p95_plan_seconds=float(np.quantile(latencies, .95)),
                level_iterations={str(level): [n for r in timing.levels
                                              if r['level'] == level for n in r['n_iters']]
                                  for level in (1, 2)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tasks', type=int, default=5)
    parser.add_argument('--task-file', type=Path, default=None,
                        help='task payload file (default: the env\'s released 50-task file)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze-model-params', action='store_true',
                      help='requires_grad=False on all model parameters; actions keep gradients')
    mode.add_argument('--action-grads-only', action='store_true',
                      help='solver backward passes compute gradients for the actions only')
    parser.add_argument('--lower-macro-iters', type=int,
                        help='exact GD steps, without early stopping, for level-1 solves with a macro-action')
    parser.add_argument('--lower-bypass-iters', type=int,
                        help='exact GD steps, without early stopping, for level-1 solves without an upper plan')
    parser.add_argument('--lower-decoder', type=Path,
                        help='decoder checkpoint: macro-conditioned level-1 solves pick its best-of-K')
    parser.add_argument('--upper-iters', type=int,
                        help='exact GD steps, without early stopping, for every level-2 solve')
    parser.add_argument('--upper-samples', type=int,
                        help='candidate action samples optimized by every level-2 solve')
    parser.add_argument('--upper-warm-start', action='store_true',
                        help='initialize level-2 solves from the previous plan shifted by one macro-action')
    parser.add_argument('--upper-proposal', type=Path,
                        help='UpperProposal checkpoint: level-2 solves pick its best-of-K '
                             '(plus the previous plan shifted by one macro-action with --upper-warm-start)')
    parser.add_argument('--upper-refine-iters', type=int,
                        help='with --upper-proposal: exact GD steps, without early stopping, from the candidates')
    parser.add_argument('--upper-couple', action='store_true',
                        help='with --upper-proposal and --lower-decoder: pick the candidate whose remaining '
                             'plan is cheapest after the decoder executes its first macro-action')
    parser.add_argument('--model-seed', type=int, default=42,
                        help='released checkpoint seed; the planner seed equals it, as in the paper')
    parser.add_argument('--env', default='pusht', choices=tuple(TASK_FILES))
    parser.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
    args = parser.parse_args()
    task_path = (args.task_file.expanduser().resolve() if args.task_file is not None
                 else TASK_FILES[args.env].resolve())
    num_tasks = len(torch.load(task_path, map_location='cpu', weights_only=False)['data'])
    if not 1 <= args.tasks <= num_tasks:
        parser.error(f'--tasks must be in [1, {num_tasks}]')
    for flag in ('lower_macro_iters', 'lower_bypass_iters', 'upper_iters', 'upper_samples',
                 'upper_refine_iters'):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1")
    if args.lower_decoder is not None and args.lower_macro_iters is not None:
        parser.error('--lower-decoder and --lower-macro-iters are mutually exclusive')
    if args.upper_proposal is not None and (args.upper_iters is not None or args.upper_samples is not None):
        parser.error('--upper-proposal excludes --upper-iters and --upper-samples')
    if args.upper_refine_iters is not None and args.upper_proposal is None:
        parser.error('--upper-refine-iters requires --upper-proposal')
    if args.upper_couple and (args.upper_proposal is None or args.lower_decoder is None
                              or args.upper_refine_iters is not None):
        parser.error('--upper-couple requires --upper-proposal and --lower-decoder, without --upper-refine-iters')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    artifacts = ROOT / 'cost_probe/artifacts'
    cfg = evaluation.load_eval_config(f'{args.env}_l2')
    cfg.cache_dir = str(artifacts)
    cfg.policy = str(decoder.checkpoint_path(args.model_seed, args.env, args.family))
    cfg.seed = args.model_seed
    cfg.load_eval_trajs_path = str(task_path)
    task_sha = sha256(task_path)
    if task_path == OFFICIAL_TASK_FILE.resolve():
        assert task_sha == OFFICIAL_TASK_SHA256, 'Task artifact checksum mismatch.'
    task_file_field = decoder.relpath(task_path)
    cfg.eval.num_eval = 1
    cfg.eval.chunk_size = 1
    apply_upper_overrides(cfg, n_steps=args.upper_iters, num_samples=args.upper_samples)
    plan_config = cfg.hierarchical_plan_config
    replanning_interval = int(plan_config.receding_horizon) * int(plan_config.level1.action_block)
    level2_cfg = cfg.solver.solvers.level2
    effective_level2 = dict(n_steps=int(level2_cfg.n_steps),
                            num_samples=int(level2_cfg.get('num_samples', 1)),
                            early_stop_patience=int(level2_cfg.get('early_stop_patience', 5)),
                            min_its=int(level2_cfg.get('min_its', 30)))
    model = evaluation.load_model(cfg)
    grad_flags = [p.requires_grad for p in model.parameters()]
    if args.freeze_model_params:
        model.requires_grad_(False)
    dec_obj = dec_ckpt = None
    if args.lower_decoder is not None:
        dec_obj, dec_ckpt = decoder.load_decoder(args.lower_decoder, torch.device('cuda'))
    up_obj = up_ckpt = None
    if args.upper_proposal is not None:
        up_obj, up_ckpt = upper_proposal.load_upper_proposal(args.upper_proposal, torch.device('cuda'))
    model_sha = sha256(Path(cfg.policy))
    for name, ckpt in (('--lower-decoder', dec_ckpt), ('--upper-proposal', up_ckpt)):
        trained_for = None if ckpt is None else (ckpt.get('model_checkpoint_sha256')
                                                 or ckpt.get('cache_attrs', {}).get('checkpoint_sha256'))
        if trained_for not in (None, model_sha):
            print(f'WARNING: {name} was trained for world model {trained_for[:12]}, loaded {model_sha[:12]}', flush=True)

    def checkpoint_entry(path, ckpt):
        resolved = path.expanduser().resolve()
        return dict(path=decoder.relpath(resolved),
                    sha256=ckpt.get('sha256') or sha256(path),
                    architecture=ckpt.get('args', {}).get('architecture', 'mlp'),
                    trainable_parameters=ckpt.get('trainable_parameters'),
                    args={k: str(v) for k, v in ckpt['args'].items()},
                    git_commit=ckpt['git_commit'])
    metadata = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                           cwd=ROOT, text=True).strip()),
                    python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    dependencies={name: version(name) for name in
                                  ('transformers', 'torchvision', 'gymnasium', 'pymunk', 'numpy')},
                    gpu=torch.cuda.get_device_name(), gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    threads=torch.get_num_threads(), checkpoint_sha256=model_sha,
                    env=args.env, family=args.family,
                    model_seed=args.model_seed, planner_seed=int(cfg.seed),
                    tasks_sha256=task_sha, task_file=task_file_field, tasks=list(range(args.tasks)),
                    parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_parameters_requiring_grad=sum(
                        p.numel() for p, flag in zip(model.parameters(), grad_flags) if flag),
                    parameters_requiring_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    attention_backend='MATH (official evaluation setting)',
                    checkpoint_parameter_grad_flags_preserved=not args.freeze_model_params,
                    freeze_model_params=args.freeze_model_params,
                    action_grads_only=args.action_grads_only,
                    lower_macro_iters=args.lower_macro_iters, macro_rule=MACRO_RULE,
                    lower_bypass_iters=args.lower_bypass_iters,
                    upper_iters=args.upper_iters, upper_samples=args.upper_samples,
                    upper_warm_start=args.upper_warm_start,
                    lower_decoder=(None if dec_ckpt is None else checkpoint_entry(args.lower_decoder, dec_ckpt)),
                    upper_proposal=(None if up_ckpt is None else checkpoint_entry(args.upper_proposal, up_ckpt)),
                    upper_refine_iters=args.upper_refine_iters, upper_couple=args.upper_couple,
                    level2_solver=effective_level2,
                    replanning_interval_env_steps=replanning_interval)
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    print('METADATA', json.dumps(metadata), flush=True)
    patches = []
    if args.lower_macro_iters is not None or args.lower_bypass_iters is not None:
        patches.append(patch.object(GradientSolver, 'solve',
                                    macro_budget(GradientSolver.solve, args.lower_macro_iters,
                                                 args.lower_bypass_iters)))
        patches[-1].start()
    dec_wrapper = None
    if dec_obj is not None:
        dec_wrapper = decoder_solve(GradientSolver.solve, dec_obj)
        patches.append(patch.object(GradientSolver, 'solve', dec_wrapper))
        patches[-1].start()
    up_wrapper = couple = None
    if args.upper_couple:
        context = {}
        patches.append(patch.object(HierarchicalSolver, '_encode_all_levels', capture_planning_context(
            HierarchicalSolver._encode_all_levels, context)))
        patches[-1].start()
        couple = (dec_obj, context)
    if up_obj is not None:
        up_wrapper = upper_proposal_solve(GradientSolver.solve, up_obj, args.upper_refine_iters or 0, couple)
        patches.append(patch.object(GradientSolver, 'solve', up_wrapper))
        patches[-1].start()
    if args.upper_warm_start:
        patches.append(patch.object(GradientSolver, 'solve',
                                    warm_start_level2(GradientSolver.solve, replanning_interval)))
        patches[-1].start()
    use_variant = None
    if args.freeze_model_params:
        def use_variant(variant):
            set_grad_flags(model, [False] * len(grad_flags) if variant else grad_flags)
            return False
    elif args.action_grads_only:
        restriction, solve, backward = action_grads_only(GradientSolver.solve, torch.Tensor.backward)
        for item in (patch.object(GradientSolver, 'solve', solve),
                     patch.object(torch.Tensor, 'backward', backward)):
            item.start()
            patches.append(item)

        def use_variant(variant):
            restriction.enabled = variant
            return variant
    timing = Timing()
    for item in timing.install():
        item.start()
        patches.append(item)
    try:
        print('WARMUP_START', flush=True)
        evaluation.run_planning_eval(cfg, model=model, results_dir=args.out / 'warmup')
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            parity = check_timing_parity(cfg, model, timing)
            if use_variant is not None:
                parity['released_vs_variant'] = check_planner_parity(cfg, model, timing, use_variant)
        if dec_wrapper is not None:
            parity['decoder_cost_vs_planner_info'] = dec_wrapper.check.get('max_abs_diff')
            assert parity['decoder_cost_vs_planner_info'] == 0.0, \
                'Decoder lean cost differs from the planner info cost.'
        if up_wrapper is not None:
            parity['upper_proposal_cost_vs_planner_info'] = up_wrapper.check.get('max_abs_diff')
            assert parity['upper_proposal_cost_vs_planner_info'] == 0.0, \
                'Proposal lean level-2 cost differs from the planner info cost.'
        if couple is not None:
            parity['upper_couple_context_z2_vs_planner'] = up_wrapper.check.get('context_z2_max_abs_diff')
            assert parity['upper_couple_context_z2_vs_planner'] == 0.0, \
                'Coupling context z1 does not encode to the planner level-2 latent.'
        print('PARITY', json.dumps(parity), flush=True)
        (args.out / 'parity.json').write_text(json.dumps(parity, indent=2))
        if use_variant is not None:
            check = parity['released_vs_variant']
            assert all(check[key] for key in ('cost_bit_exact', 'action_gradient_bit_exact',
                                              'actions_bit_exact', 'iterations_identical')), \
                'The variant changed planning.'
            elements = check['parameter_elements_with_grad']
            assert elements['variant'] == 0 < elements['released'], elements
        cfg.eval.num_eval = args.tasks
        timing.enabled = True
        torch.cuda.reset_peak_memory_stats()
        metrics = evaluation.run_planning_eval(cfg, model=model, results_dir=args.out / 'evaluation')
        summary = summarize(timing, metrics['evaluation_time'])
        summary['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
        summary['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2 ** 30
        (args.out / 'planning_summary.json').write_text(json.dumps(summary, indent=2))
        _, cost_info, cost_actions = timing.low_cost
        torch.save({'info': {k: v.detach().cpu() if torch.is_tensor(v) else v
                             for k, v in cost_info.items()},
                    'actions': cost_actions.cpu()}, args.out / 'captured_cost.pt')
        (args.out / 'planning_calls.json').write_text(json.dumps(timing.plans, indent=2))
        (args.out / 'level_calls.json').write_text(json.dumps(timing.levels, indent=2))
        with (args.out / 'planning_calls.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing.plans[0]))
            writer.writeheader()
            writer.writerows(timing.plans)
        print('PLANNING_SUMMARY', json.dumps(summary), flush=True)
        timing.enabled = False
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            components = component_benchmark(timing.low_cost, restrict_backward=args.action_grads_only)
        limitations = ['Five-task pilot by default, one checkpoint and GPU.',
                       'Runtime dependencies differ from the paper; see metadata.',
                       'Zero lower cost is an upper bound, not a measured Decoder speedup.',
                       'Official environment resets retain their original randomness.']
        if args.freeze_model_params:
            limitations.append('Model parameters frozen; parity.json compares the checkpoint gradient flags.')
        if args.action_grads_only:
            limitations.append('Solver backward passes compute action gradients only (parameter flags unchanged); '
                               'parity.json compares them with the released full backward.')
        if args.lower_macro_iters is not None:
            limitations.append(f'Level-1 solves with a macro-action run exactly {args.lower_macro_iters} GD steps '
                               'without early stopping; all other solves keep the released budget.')
        if args.lower_bypass_iters is not None:
            limitations.append(f'Level-1 solves without an upper plan run exactly {args.lower_bypass_iters} GD steps '
                               'without early stopping.')
        if args.upper_iters is not None:
            limitations.append(f'Level-2 solves run exactly {args.upper_iters} GD steps without early stopping.')
        if args.upper_samples is not None:
            limitations.append(f'Level-2 solves optimize {args.upper_samples} candidate action samples.')
        if args.upper_warm_start:
            limitations.append('Level-2 solves start from the previous best plan shifted by one macro-action; '
                               'other candidates keep the released prior-std noise.')
        if args.lower_decoder is not None:
            limitations.append('Level-1 solves with a macro-action use the decoder: best of K by the '
                               "planner's level-1 cost, no GD steps; other solves keep their budgets.")
        if args.upper_proposal is not None:
            limitations.append("Level-2 solves use the proposal network: best of its K plans by the planner's "
                               'level-2 cost' + (', plus the previous plan shifted by one macro-action'
                                                 if args.upper_warm_start else '')
                               + (f', then exactly {args.upper_refine_iters} GD steps from all candidates'
                                  if args.upper_refine_iters else ', no GD steps') + '.')
        if args.upper_couple:
            limitations.append('Level-2 selection is coupled: the candidate whose remaining macros are cheapest '
                               'after the level-1 decoder executes its first macro-action (level-1 model '
                               'rollout) is chosen, not the lowest level-2 cost.')
        report = dict(metadata=metadata, parity=parity, planning=summary, components=components,
                      evaluation_metrics=metrics, limitations=limitations)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
        print('COMPONENTS', json.dumps(components), flush=True)
        print('PROFILE_COMPLETE', args.out / 'report.json', flush=True)
    finally:
        for item in reversed(patches):
            item.stop()


if __name__ == '__main__':
    main()
