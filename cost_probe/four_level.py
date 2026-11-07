"""Four-level (L4 -> L3 -> L2 -> L1) H-JEPA planning amortization: parts, data, trainers, and evaluator.

Extends three_level.py to a 4-level hierarchy stack where each level can be a 1-layer
Transformer Decoder + GELU MLP head (with frozen world model in the loop, tf_ar_wm) or a
4-block residual MLP head:
  - Top Level 4: Proposal head (UpperProposal or TransformerUpperProposal, max_h=2) mapping
    (z4, g4, h4) -> 32 candidate L4 macro sequences inside [l4_lo, l4_hi], scored by l4_cost_model.
  - Upper-Middle Level 3: Macro decoder (L2MacroDecoder or TransformerL2MacroDecoder) mapping
    (z3, z4, g4_sub, m4) -> 16 candidate 2-step L3 macro sequences inside [l3_lo, l3_hi],
    scored by l3_cost_model.
  - Lower-Middle Level 2: Macro decoder (L2MacroDecoder or TransformerL2MacroDecoder) mapping
    (z2, z3, g3_sub, m3) -> 16 candidate 2-step L2 macro sequences inside [l2_lo, l2_hi],
    scored by l2_cost_model.
  - Bottom Level 1: ActionDecoder or TransformerActionDecoder mapping
    (z1, z2, g2_sub, m2) -> 16 candidate 2-block primitive action sequences,
    scored by l1_cost_model.
  - Bypass solves (without an upper plan, when upper levels are skipped at horizon 1) run fixed
    bypass_iters=10 GD steps with action-only gradients.
"""

import argparse
import copy
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'h_jepa'), str(ROOT / 'cost_probe')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MUJOCO_GL', 'egl')

import planning_eval as evaluation
import decoder as D
import profile_cost as PC
import three_level as TL
import transformer_heads as TH
import upper_proposal as U
from hierarchical_solver import HierarchicalSolver
from stable_worldmodel.solver.gd import GradientSolver

encode_upper = TL.encode_upper
level2_macro_cost = TL.level2_macro_cost
level3_macro_cost = TL.level2_macro_cost
L2MacroDecoder = TL.L2MacroDecoder
TransformerL2MacroDecoder = TL.TransformerL2MacroDecoder
L3MacroDecoder = TL.L2MacroDecoder
TransformerL3MacroDecoder = TL.TransformerL2MacroDecoder
load_l2_decoder = TL.load_l2_decoder
load_l3_decoder = TL.load_l2_decoder
l2_macro_bc_loss = TL.l2_macro_bc_loss
l3_macro_bc_loss = TL.l2_macro_bc_loss
augment_macros_l3 = TL.augment_macros_l3
l2_decoder_solve = TL.l2_decoder_solve


def level4_cost(l4_cost_model, z4, g4, cand_m4, pred_emb=None, *, action_cost_weight,
                cost_last_n, intermediate_cost_weight, goal_level=4):
    """Lean top-level cost, cand_m4 (B, K, h4, 4) -> (B, K)."""
    b, k = cand_m4.shape[:2]
    info = {'pixels': z4.new_zeros(b, k, 1),
            'embed_0': z4[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0': g4[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0_level': goal_level,
            'goal_embed_0_in_upper_space': False,
            'dense_upper_goal_projection': False,
            'action_cost_weight': action_cost_weight,
            'cost_last_n': cost_last_n,
            'intermediate_cost_weight': intermediate_cost_weight}
    if pred_emb is not None:
        return TH.cost_from_rollout(l4_cost_model, info, cand_m4, pred_emb)
    return l4_cost_model.get_cost(info, cand_m4)


class L4PlannerParts:
    """Frozen 4-level H-JEPA planner components, priors, clip bounds, and cost wrappers."""

    def __init__(self, device='cuda', model_seed=42, env='pusht', family='hjepa'):
        artifacts = ROOT / 'cost_probe/artifacts'
        cfg = evaluation.load_eval_config(f'{env}_l4')
        cfg.cache_dir = str(artifacts)
        cfg.policy = str(D.checkpoint_path(model_seed, env, family, levels=4))
        cfg.seed = model_seed
        self.env, self.family = env, family
        self.model_seed, self.checkpoint_sha256 = model_seed, D.sha256(cfg.policy)
        model = evaluation.load_model(cfg)
        model.requires_grad_(False)
        process = evaluation._build_eval_process_from_policy_normalizer(cfg, model=model)
        self.action_scaler = process['action']
        self.action_dim = len(self.action_scaler.mean_)
        solver = evaluation.build_solver(cfg, model)
        plan_config = evaluation._build_policy_plan_config(cfg)
        solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, self.action_dim),
                                                     dtype=np.float32),
                         n_envs=1, config=plan_config)
        self.cfg, self.model, self.solver = cfg, model, solver
        self.l1_cost_model = solver.level_solvers[1].model
        self.l2_cost_model = solver.level_solvers[2].model
        self.l3_cost_model = solver.level_solvers[3].model
        self.l4_cost_model = solver.level_solvers[4].model
        self.level2_jepa = solver.level_models[1]
        self.level3_jepa = solver.level_models[2]
        self.level4_jepa = solver.level_models[3]

        m2, s2 = solver.level_solvers[2]._get_action_prior_stats()
        self.l2_prior_mean, self.l2_prior_std = m2.to(device), s2.to(device)
        lo2, hi2 = solver.level_solvers[2]._get_action_clip_bounds()
        self.l2_lo, self.l2_hi = lo2.to(device), hi2.to(device)
        self.prior_mean, self.prior_std = self.l2_prior_mean, self.l2_prior_std

        m3, s3 = solver.level_solvers[3]._get_action_prior_stats()
        self.l3_prior_mean, self.l3_prior_std = m3.to(device), s3.to(device)
        lo3, hi3 = solver.level_solvers[3]._get_action_clip_bounds()
        self.l3_lo, self.l3_hi = lo3.to(device), hi3.to(device)

        m4, s4 = solver.level_solvers[4]._get_action_prior_stats()
        self.l4_prior_mean, self.l4_prior_std = m4.to(device), s4.to(device)
        lo4, hi4 = solver.level_solvers[4]._get_action_clip_bounds()
        self.l4_lo, self.l4_hi = lo4.to(device), hi4.to(device)

        self.action_mean = torch.as_tensor(self.action_scaler.mean_, dtype=torch.float32, device=device)
        self.action_std = torch.as_tensor(self.action_scaler.scale_, dtype=torch.float32, device=device)

        c1, c2, c3, c4 = plan_config.level1, plan_config.level2, plan_config.level3, plan_config.level4
        self.action_block = int(c1.action_block)
        self.l1_cost_kwargs = dict(action_cost_weight=float(c1.action_cost_weight),
                                   cost_last_n=int(c1.cost_last_n),
                                   intermediate_cost_weight=float(c1.intermediate_cost_weight))
        self.l2_cost_kwargs = dict(action_cost_weight=float(c2.action_cost_weight),
                                   cost_last_n=int(c2.cost_last_n),
                                   intermediate_cost_weight=float(c2.intermediate_cost_weight))
        self.l3_cost_kwargs = dict(action_cost_weight=float(c3.action_cost_weight),
                                   cost_last_n=int(c3.cost_last_n),
                                   intermediate_cost_weight=float(c3.intermediate_cost_weight))
        self.l4_cost_kwargs = dict(action_cost_weight=float(c4.action_cost_weight),
                                   cost_last_n=int(c4.cost_last_n),
                                   intermediate_cost_weight=float(c4.intermediate_cost_weight))
        self.top_level = 4 if int(c4.horizon) > 1 else (3 if int(c3.horizon) > 1 else 2)
        self.max_h4 = int(getattr(plan_config, f'level{self.top_level}').horizon)
        self.device = device

    @torch.no_grad()
    def encode_level2(self, z1):
        return encode_upper(self.level2_jepa, z1, 1)

    @torch.no_grad()
    def encode_level3(self, z2):
        return encode_upper(self.level3_jepa, z2, 2)

    @torch.no_grad()
    def encode_level4(self, z3):
        return encode_upper(self.level4_jepa, z3, 3)

    @torch.no_grad()
    def subgoal_l2(self, z2, m2):
        info = {'embed_0': z2[:, None, None], 'pixels': z2.new_zeros(z2.shape[0], 1, 1)}
        out = self.l2_cost_model.rollout(info, m2.view(-1, 1, 1, 4))
        return out['predicted_embed_0'][:, 0, 1]

    @torch.no_grad()
    def subgoal_l3(self, z3, m3):
        info = {'embed_0': z3[:, None, None], 'pixels': z3.new_zeros(z3.shape[0], 1, 1)}
        out = self.l3_cost_model.rollout(info, m3.view(-1, 1, 1, 4))
        return out['predicted_embed_0'][:, 0, 1]

    @torch.no_grad()
    def subgoal_l4(self, z4, m4):
        info = {'embed_0': z4[:, None, None], 'pixels': z4.new_zeros(z4.shape[0], 1, 1)}
        out = self.l4_cost_model.rollout(info, m4.view(-1, 1, 1, 4))
        return out['predicted_embed_0'][:, 0, 1]

    @torch.no_grad()
    def data_macro_l2(self, z1_seq, blocks):
        pooled = self.l1_cost_model._encode_actions_up(
            z1_seq[:, None], blocks[:, None], pad=0, dense=False)
        return pooled[:, 0, 0]

    @torch.no_grad()
    def data_macro_l3(self, z2_seq, m2_blocks):
        pooled = self.l2_cost_model._encode_actions_up(
            z2_seq[:, None], m2_blocks[:, None], pad=0, dense=False)
        return pooled[:, 0, 0]

    @torch.no_grad()
    def data_macro_l4(self, z3_seq, m3_blocks):
        pooled = self.l3_cost_model._encode_actions_up(
            z3_seq[:, None], m3_blocks[:, None], pad=0, dense=False)
        return pooled[:, 0, 0]

    def level1_cost(self, z1, g2, m2, cand_a1, pred_emb=None):
        if pred_emb is not None:
            return TH.level1_cost_from_rollout(self.l1_cost_model, z1, g2, m2, cand_a1, pred_emb,
                                               **self.l1_cost_kwargs)
        return D.level1_cost(self.l1_cost_model, z1, g2, m2, cand_a1, **self.l1_cost_kwargs)

    def level2_cost(self, z2, g3, m3, cand_m2, pred_emb=None):
        return level2_macro_cost(self.l2_cost_model, z2, g3, m3, cand_m2, pred_emb=pred_emb,
                                 **self.l2_cost_kwargs)

    def level3_macro_cost(self, z3, g4, m4, cand_m3, pred_emb=None):
        return level3_macro_cost(self.l3_cost_model, z3, g4, m4, cand_m3, pred_emb=pred_emb,
                                 **self.l3_cost_kwargs)

    def level4_cost(self, z4, g4, cand_m4, pred_emb=None):
        return level4_cost(self.l4_cost_model, z4, g4, cand_m4, pred_emb=pred_emb,
                           **self.l4_cost_kwargs)

    def top_level_cost(self, z, g, cand, pred_emb=None):
        if self.top_level == 2:
            return level4_cost(self.l2_cost_model, z, g, cand, pred_emb=pred_emb,
                               goal_level=2, **self.l2_cost_kwargs)
        if self.top_level == 3:
            return level4_cost(self.l3_cost_model, z, g, cand, pred_emb=pred_emb,
                               goal_level=3, **self.l3_cost_kwargs)
        return self.level4_cost(z, g, cand, pred_emb=pred_emb)


def augment_macros_l4(parts, m4_data, generator):
    """With prob AUG_PROB replace L4 macro: half fresh L4 prior draws, half perturbed data."""
    b = m4_data.shape[0]
    noise = torch.randn(b, 4, device=m4_data.device, generator=generator)
    fresh = parts.l4_prior_mean + parts.l4_prior_std * noise
    perturbed = m4_data + D.AUG_PERTURB_STD * parts.l4_prior_std * noise
    fresh = torch.minimum(torch.maximum(fresh, parts.l4_lo), parts.l4_hi)
    perturbed = torch.minimum(torch.maximum(perturbed, parts.l4_lo), parts.l4_hi)
    use_fresh = torch.rand(b, device=m4_data.device, generator=generator) < 0.5
    replace = torch.rand(b, device=m4_data.device, generator=generator) < D.AUG_PROB
    m4 = torch.where(use_fresh[:, None], fresh, perturbed)
    m4 = torch.where(replace[:, None], m4, m4_data)
    return m4, ~replace


class L4HierarchyData:
    """On-device L2/L3/L4 latents and macros constructed from a 4-level model's L1 cache h5."""

    def __init__(self, path, parts, holdout_every=100, chunk=65536, action_block=5):
        dev = parts.device
        self.device = dev
        with h5py.File(path, 'r') as f:
            embed = torch.from_numpy(f['embed'][:]).to(dev)
            action = torch.from_numpy(f['action'][:]).to(dev)
            episode = torch.from_numpy(np.asarray(f['episode'][:])).to(dev)
            step = torch.from_numpy(np.asarray(f['step'][:])).to(dev)
            self.attrs = dict(f.attrs)
        n = len(episode)
        self.num_frames = n
        self.action_block = action_block
        self.span_l2 = 2 * action_block
        self.span_l3 = 4 * action_block
        self.span_l4 = 8 * action_block
        self.remaining = U.run_remaining(episode, step)

        self.z2 = torch.cat([parts.encode_level2(embed[i:i + chunk]) for i in range(0, n, chunk)])
        self.z3 = torch.cat([parts.encode_level3(self.z2[i:i + chunk]) for i in range(0, n, chunk)])
        self.z4 = torch.cat([parts.encode_level4(self.z3[i:i + chunk]) for i in range(0, n, chunk)])

        self.m2 = torch.full((n, U.MACRO_DIM), float('nan'), device=dev)
        has_l2 = torch.nonzero(self.remaining >= self.span_l2, as_tuple=True)[0]
        off_l2 = torch.tensor([0, action_block, self.span_l2], device=dev)
        act_idx = torch.arange(self.span_l2, device=dev)
        block_dim = action_block * int(action.shape[1])
        for i in range(0, len(has_l2), chunk):
            t = has_l2[i:i + chunk]
            blocks = ((action[t[:, None] + act_idx] - parts.action_mean) / parts.action_std).reshape(-1, 2, block_dim)
            z1_seq = embed[t[:, None] + off_l2]
            self.m2[t] = parts.data_macro_l2(z1_seq, blocks)
        del embed

        self.m3 = torch.full((n, U.MACRO_DIM), float('nan'), device=dev)
        has_l3 = torch.nonzero(self.remaining >= self.span_l3, as_tuple=True)[0]
        off_l3 = torch.tensor([0, self.span_l2, self.span_l3], device=dev)
        off_m2 = torch.tensor([0, self.span_l2], device=dev)
        for i in range(0, len(has_l3), chunk):
            t = has_l3[i:i + chunk]
            z2_seq = self.z2[t[:, None] + off_l3]
            m2_pair = self.m2[t[:, None] + off_m2]
            self.m3[t] = parts.data_macro_l3(z2_seq, m2_pair)

        self.m4 = torch.full((n, U.MACRO_DIM), float('nan'), device=dev)
        has_l4 = torch.nonzero(self.remaining >= self.span_l4, as_tuple=True)[0]
        off_l4 = torch.tensor([0, self.span_l3, self.span_l4], device=dev)
        off_m3 = torch.tensor([0, self.span_l3], device=dev)
        for i in range(0, len(has_l4), chunk):
            t = has_l4[i:i + chunk]
            z3_seq = self.z3[t[:, None] + off_l4]
            m3_pair = self.m3[t[:, None] + off_m3]
            self.m4[t] = parts.data_macro_l4(z3_seq, m3_pair)

        self.episode = episode
        masks = ({'all': torch.ones(n, dtype=torch.bool, device=dev)} if holdout_every is None
                 else {'train': episode % holdout_every != 0, 'heldout': episode % holdout_every == 0})
        self.splits = {}
        self.valid_l2 = {}
        self.valid_l3 = {}
        for name, keep in masks.items():
            frames = torch.nonzero(keep, as_tuple=True)[0]
            frames = frames[torch.argsort(self.remaining[frames], stable=True)]
            self.splits[name] = (frames, self.remaining[frames].contiguous())
            self.valid_l2[name] = torch.nonzero(keep & (self.remaining >= self.span_l3), as_tuple=True)[0]
            self.valid_l3[name] = torch.nonzero(keep & (self.remaining >= self.span_l4), as_tuple=True)[0]

    def sample_l2(self, n, generator, split='train'):
        """Sample n valid 20-step frames for lower-middle L2 decoder -> (z2, z3, m2_pair (n,2,4), m3_data (n,4), idx)."""
        valid = self.valid_l2[split]
        pick = torch.randint(len(valid), (n,), device=self.device, generator=generator)
        idx = valid[pick]
        off_m2 = torch.tensor([0, self.span_l2], device=self.device)
        return self.z2[idx], self.z3[idx], self.m2[idx[:, None] + off_m2], self.m3[idx], idx

    def sample_l3_middle(self, n, generator, split='train'):
        """Sample n valid 40-step frames for upper-middle L3 decoder -> (z3, z4, m3_pair (n,2,4), m4_data (n,4), idx)."""
        valid = self.valid_l3[split]
        pick = torch.randint(len(valid), (n,), device=self.device, generator=generator)
        idx = valid[pick]
        off_m3 = torch.tensor([0, self.span_l3], device=self.device)
        return self.z3[idx], self.z4[idx], self.m3[idx[:, None] + off_m3], self.m4[idx], idx

    def problems_l4(self, t, h4, offset, top_level=4):
        span = {2: self.span_l2, 3: self.span_l3, 4: self.span_l4}[top_level]
        z = {2: self.z2, 3: self.z3, 4: self.z4}[top_level]
        m = {2: self.m2, 3: self.m3, 4: self.m4}[top_level]
        steps = (t[:, None] + span * torch.arange(h4, device=t.device)).clamp_max(self.num_frames - 1)
        return z[t], z[t + offset], m[steps]

    def sample_l4(self, n, h4, generator, split='train', match_prob=U.MATCH_PROB, jitter=U.JITTER,
                  offset_range=None, top_level=4):
        """Sample n top-level problems at horizon h4 (top_level=4: 40 steps; top_level=3: 20 steps; top_level=2: 10 steps)."""
        dev = self.device
        span = {2: self.span_l2, 3: self.span_l3, 4: self.span_l4}[top_level]
        frames, remaining = self.splits[split]
        base = span * h4 - 5
        if offset_range is not None:
            lo, hi = offset_range
            offset = torch.randint(lo, hi + 1, (n,), device=dev, generator=generator)
            mask = offset >= base
        else:
            mask = torch.rand(n, device=dev, generator=generator) < match_prob
            jittered = torch.randint(max(5, base + jitter[0]), base + jitter[1] + 1, (n,),
                                     device=dev, generator=generator)
            offset = torch.where(mask, torch.full_like(jittered, base), jittered)
        need = torch.where(mask, torch.maximum(offset, torch.full_like(offset, span * h4)),
                           offset)
        start = torch.searchsorted(remaining, need)
        count = len(frames) - start
        assert bool((count > 0).all()), 'no frame covers the requested L4 goal offset'
        pick = torch.minimum((torch.rand(n, device=dev, generator=generator) * count).long(), count - 1)
        t = frames[start + pick]
        z4, g4, expert = self.problems_l4(t, h4, offset, top_level=top_level)
        expert = torch.where(mask[:, None, None], expert, torch.zeros_like(expert))
        return z4, g4, expert, mask, offset, t


def l3_decoder_solve(original, dec_l3):
    """Route macro-conditioned level-3 solves (planning_horizon == 2) to L3MacroDecoder's best-of-K."""
    state = {}

    def solve(solver, info_dict, *args, **kwargs):
        if solver.level != 3 or not PC.has_upper_plan(info_dict) \
                or kwargs.get('planning_horizon') != 2:
            return original(solver, info_dict, *args, **kwargs)
        t0 = time.time()
        b = info_dict['embed_0'].shape[0]
        dev = solver.device
        z3 = info_dict['embed_0'][:, -1].to(dev)
        g4 = info_dict['goal_embed_0'][:, 0].to(dev)
        m4 = info_dict['goal_action'][:, 0].to(dev)
        arch = getattr(dec_l3, 'architecture', 'mlp')
        with torch.no_grad():
            z4 = encode_upper(solver.model.upper_models[0], z3, 3)
            if arch == 'mlp':
                cand = dec_l3(z3, z4, g4, m4)
                costs = level3_macro_cost(
                    solver.model, z3, g4, m4, cand,
                    action_cost_weight=float(info_dict['action_cost_weight']),
                    cost_last_n=int(solver._config.cost_last_n),
                    intermediate_cost_weight=float(solver._config.intermediate_cost_weight))
            else:
                cand, pred_emb, _ = dec_l3.generate(z3, z4, g4, m4, cost_model=solver.model)
                costs = level3_macro_cost(
                    solver.model, z3, g4, m4, cand, pred_emb=pred_emb,
                    action_cost_weight=float(info_dict['action_cost_weight']),
                    cost_last_n=int(solver._config.cost_last_n),
                    intermediate_cost_weight=float(solver._config.intermediate_cost_weight))
        if 'max_abs_diff' not in state:
            expanded = solver._expand_info_dict_for_samples(
                dict(info_dict), start_idx=0, end_idx=solver.n_envs, num_samples=dec_l3.heads)
            expanded['cost_last_n'] = solver._config.cost_last_n
            expanded['intermediate_cost_weight'] = solver._config.intermediate_cost_weight
            with torch.no_grad():
                full = solver.model.get_cost(expanded, cand)
            state['max_abs_diff'] = float((full - costs).abs().max())
        best = cand[torch.arange(b, device=dev), costs.argmin(dim=1)]
        actions = best.detach().cpu()
        predictions = solver._rollout_final_actions(info_dict, actions)
        elapsed = time.time() - t0
        solver.solve_records.append(dict(
            level=3, horizon=2, num_samples=dec_l3.heads, n_iters=[0] * b,
            wall_clock=elapsed, wall_clock_per_env=[elapsed] * b,
            l3_decoder=True, l3_decoder_best_cost=costs.min(dim=1).values.tolist()))
        return dict(actions=actions, predictions=predictions)

    solve.check = state
    return solve


def l4_proposal_solve(original, net_l4, top_level=4):
    """Route active top-level solves to the top proposal network (MLP or Transformer; no shifted warm start)."""
    state = {}

    def solve(solver, info_dict, *args, **kwargs):
        if solver.level != top_level:
            return original(solver, info_dict, *args, **kwargs)
        t0 = time.time()
        h4 = int(kwargs['planning_horizon'])
        b, dev = solver.n_envs, solver.device
        z4, g4 = info_dict['embed_0'][:, -1].to(dev), info_dict['goal_embed_0'][:, -1].to(dev)
        cost_kwargs = dict(action_cost_weight=float(info_dict['action_cost_weight']),
                           cost_last_n=max(1, int(solver._config.cost_last_n)),
                           intermediate_cost_weight=float(solver._config.intermediate_cost_weight),
                           goal_level=top_level)
        arch = getattr(net_l4, 'architecture', 'mlp')
        with torch.no_grad():
            if arch == 'mlp':
                cand = net_l4(z4, g4, torch.full((b,), h4, dtype=torch.long, device=dev))[:, :, :h4]
                costs = level4_cost(solver.model, z4, g4, cand, **cost_kwargs)
            else:
                cand, pred_emb, _ = net_l4.generate(
                    z4, g4, torch.full((b,), h4, dtype=torch.long, device=dev),
                    cost_model=solver.model)
                costs = level4_cost(solver.model, z4, g4, cand, pred_emb=pred_emb, **cost_kwargs)
        if 'max_abs_diff' not in state:
            expanded = solver._expand_info_dict_for_samples(
                dict(info_dict), start_idx=0, end_idx=b, num_samples=cand.shape[1])
            expanded['cost_last_n'] = cost_kwargs['cost_last_n']
            expanded['intermediate_cost_weight'] = cost_kwargs['intermediate_cost_weight']
            with torch.no_grad():
                full = solver.model.get_cost(expanded, cand)
            state['max_abs_diff'] = float((full - costs).abs().max())
        chosen = costs.argmin(dim=1)
        actions = cand[torch.arange(b, device=dev), chosen].detach().cpu()
        predictions = solver._rollout_final_actions(info_dict, actions)
        elapsed = time.time() - t0
        solver.solve_records.append(dict(
            level=top_level, horizon=h4, num_samples=int(cand.shape[1]), n_iters=[0] * b,
            wall_clock=elapsed, wall_clock_per_env=[elapsed] * b,
            proposal=True, proposal_architecture=arch,
            proposal_chosen=chosen.tolist(),
            proposal_best_cost=costs.min(dim=1).values.tolist()))
        return dict(actions=actions, predictions=predictions)

    solve.check = state
    return solve


def bypass_budget_4level(original, bypass_iterations=None, top_level=4):
    """Fix GD step count to bypass_iterations whenever a level below top_level solves without an upper plan."""
    def solve(solver, info_dict, *args, **kwargs):
        if bypass_iterations is None or solver.level >= top_level or PC.has_upper_plan(info_dict):
            return original(solver, info_dict, *args, **kwargs)
        saved = solver.n_steps, solver.early_stop_patience
        solver.n_steps, solver.early_stop_patience = bypass_iterations, 0
        try:
            return original(solver, info_dict, *args, **kwargs)
        finally:
            solver.n_steps, solver.early_stop_patience = saved
    return solve


def _cosine_lr(step, total_steps, base_lr):
    if step < 1000:
        return base_lr * (step + 1) / 1000
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - 1000) / max(1, total_steps - 1000)))


def cmd_train_l1(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda'
    parts = L4PlannerParts(device, args.model_seed, args.env, args.family)
    cache = D.cache_path('train', args.model_seed, args.env, args.family, levels=4)
    train = D.CacheData(cache, device, holdout=False, scaler=parts.action_scaler,
                        action_block=parts.action_block)
    held = D.CacheData(cache, device, holdout=True, scaler=parts.action_scaler,
                       action_block=parts.action_block)
    assert str(train.attrs['checkpoint_sha256']) == parts.checkpoint_sha256
    save_args = dict(vars(args), embed_dim=int(train.embed.shape[1]),
                     block_dim=parts.action_block * parts.action_dim, levels=4)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        dec = D.ActionDecoder(heads=args.heads, width=args.width, blocks=args.blocks,
                              embed_dim=save_args['embed_dim'],
                              block_dim=save_args['block_dim']).to(device)
    else:
        dec = TH.TransformerActionDecoder(
            mode=args.architecture, heads=args.heads, embed_dim=save_args['embed_dim'],
            block_dim=save_args['block_dim'], mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        dec.bind_world_model(parts.l1_cost_model)
        save_args.update(d_model=dec.d_model, layers=dec.layers,
                         attention_heads=dec.nhead, ffn_dim=dec.ffn_dim,
                         mlp_head_dim=dec.mlp_head_dim)
    dec.prior_mean.copy_(parts.l2_prior_mean)
    dec.prior_std.copy_(parts.l2_prior_std)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(save_args, git_commit=commit, cache=D.relpath(cache),
                  cache_sha256=D.sha256(cache), cache_attrs=train.attrs,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=TH.trainable_parameters(dec),
                  n_train=len(train.valid), n_heldout=len(held.valid))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        lr = _cosine_lr(step, args.steps, args.lr)
        for g in opt.param_groups:
            g['lr'] = lr
        z1, z1_seq, blocks, _ = train.sample_batch(args.batch, gen)
        with torch.no_grad():
            m2_data = parts.data_macro_l2(z1_seq, blocks)
            m2, keep = D.augment_macros(parts, z1_seq, blocks, m2_data, gen)
            z2 = parts.encode_level2(z1)
            g2 = parts.subgoal_l2(z2, m2)
        if is_mlp:
            out = dec(z1, z2, g2, m2)
            costs = parts.level1_cost(z1, g2, m2, out)
        else:
            out, pred_emb, _ = dec.generate(z1, z2, g2, m2, cost_model=parts.l1_cost_model)
            costs = parts.level1_cost(z1, g2, m2, out, pred_emb=pred_emb)
        l_cost = D.relaxed_wta(costs)
        l_bc = D.bc_loss(out, blocks, keep)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0 or step == 0 or step + 1 == args.steps:
            log.write(json.dumps(dict(step=step + 1, seconds=time.time() - t0,
                                      l_cost=float(l_cost), l_bc=float(l_bc),
                                      best_cost=float(costs.min(1).values.mean()))) + '\n')
            log.flush()
    torch.save(dict(state_dict=dec.state_dict(), args=save_args, git_commit=commit,
                    cache_sha256=config['cache_sha256'],
                    cache_attrs={k: str(v) for k, v in train.attrs.items()},
                    model_checkpoint_sha256=parts.checkpoint_sha256),
               args.out / 'decoder.pt')
    print('TRAIN_DONE', args.out, f'{(time.time() - t0) / args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', flush=True)


def cmd_train_l2(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda'
    parts = L4PlannerParts(device, args.model_seed, args.env, args.family)
    cache = D.cache_path('train', args.model_seed, args.env, args.family, levels=4)
    data = L4HierarchyData(cache, parts, holdout_every=100, action_block=parts.action_block)
    assert str(data.attrs['checkpoint_sha256']) == parts.checkpoint_sha256
    save_args = dict(vars(args), embed_dim=int(data.z2.shape[1]), macro_dim=4, levels=4)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        dec = L2MacroDecoder(heads=args.heads, width=args.width, blocks=args.blocks,
                             embed_dim=save_args['embed_dim'], macro_dim=4).to(device)
    else:
        dec = TransformerL2MacroDecoder(
            mode=args.architecture, heads=args.heads, embed_dim=save_args['embed_dim'],
            macro_dim=4, mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        dec.l2_prior_mean.copy_(parts.l2_prior_mean)
        dec.l2_prior_std.copy_(parts.l2_prior_std)
        dec.bind_world_model(parts.l2_cost_model)
        save_args.update(d_model=dec.d_model, layers=dec.layers,
                         attention_heads=dec.nhead, ffn_dim=dec.ffn_dim,
                         mlp_head_dim=dec.mlp_head_dim)
    dec.prior_mean.copy_(parts.l3_prior_mean)
    dec.prior_std.copy_(parts.l3_prior_std)
    dec.lo.copy_(parts.l2_lo)
    dec.hi.copy_(parts.l2_hi)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(save_args, git_commit=commit, cache=D.relpath(cache),
                  cache_sha256=D.sha256(cache), cache_attrs=data.attrs,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=TH.trainable_parameters(dec),
                  n_train=len(data.valid_l2['train']), n_heldout=len(data.valid_l2['heldout']))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        lr = _cosine_lr(step, args.steps, args.lr)
        for g in opt.param_groups:
            g['lr'] = lr
        z2, z3, m2_pair, m3_data, _ = data.sample_l2(args.batch, gen, 'train')
        with torch.no_grad():
            m3, keep = augment_macros_l3(parts, m3_data, gen)
            g3 = parts.subgoal_l3(z3, m3)
        if is_mlp:
            out = dec(z2, z3, g3, m3)
            costs = parts.level2_cost(z2, g3, m3, out)
        else:
            out, pred_emb, _ = dec.generate(z2, z3, g3, m3, cost_model=parts.l2_cost_model)
            costs = parts.level2_cost(z2, g3, m3, out, pred_emb=pred_emb)
        l_cost = D.relaxed_wta(costs)
        l_bc = l2_macro_bc_loss(out, m2_pair, parts.l2_prior_std, keep)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0 or step == 0 or step + 1 == args.steps:
            log.write(json.dumps(dict(step=step + 1, seconds=time.time() - t0,
                                      l_cost=float(l_cost), l_bc=float(l_bc),
                                      best_cost=float(costs.min(1).values.mean()))) + '\n')
            log.flush()
    torch.save(dict(state_dict=dec.state_dict(), args=save_args, git_commit=commit,
                    cache_sha256=config['cache_sha256'],
                    cache_attrs={k: str(v) for k, v in data.attrs.items()},
                    model_checkpoint_sha256=parts.checkpoint_sha256),
               args.out / 'l2_decoder.pt')
    print('TRAIN_DONE', args.out, f'{(time.time() - t0) / args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', flush=True)


def cmd_train_l3(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda'
    parts = L4PlannerParts(device, args.model_seed, args.env, args.family)
    cache = D.cache_path('train', args.model_seed, args.env, args.family, levels=4)
    data = L4HierarchyData(cache, parts, holdout_every=100, action_block=parts.action_block)
    assert str(data.attrs['checkpoint_sha256']) == parts.checkpoint_sha256
    save_args = dict(vars(args), embed_dim=int(data.z3.shape[1]), macro_dim=4, levels=4)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        dec = L3MacroDecoder(heads=args.heads, width=args.width, blocks=args.blocks,
                             embed_dim=save_args['embed_dim'], macro_dim=4).to(device)
    else:
        dec = TransformerL3MacroDecoder(
            mode=args.architecture, heads=args.heads, embed_dim=save_args['embed_dim'],
            macro_dim=4, mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        dec.l2_prior_mean.copy_(parts.l3_prior_mean)
        dec.l2_prior_std.copy_(parts.l3_prior_std)
        dec.bind_world_model(parts.l3_cost_model)
        save_args.update(d_model=dec.d_model, layers=dec.layers,
                         attention_heads=dec.nhead, ffn_dim=dec.ffn_dim,
                         mlp_head_dim=dec.mlp_head_dim)
    dec.prior_mean.copy_(parts.l4_prior_mean)
    dec.prior_std.copy_(parts.l4_prior_std)
    dec.lo.copy_(parts.l3_lo)
    dec.hi.copy_(parts.l3_hi)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(save_args, git_commit=commit, cache=D.relpath(cache),
                  cache_sha256=D.sha256(cache), cache_attrs=data.attrs,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=TH.trainable_parameters(dec),
                  n_train=len(data.valid_l3['train']), n_heldout=len(data.valid_l3['heldout']))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        lr = _cosine_lr(step, args.steps, args.lr)
        for g in opt.param_groups:
            g['lr'] = lr
        z3, z4, m3_pair, m4_data, _ = data.sample_l3_middle(args.batch, gen, 'train')
        with torch.no_grad():
            m4, keep = augment_macros_l4(parts, m4_data, gen)
            g4 = parts.subgoal_l4(z4, m4)
        if is_mlp:
            out = dec(z3, z4, g4, m4)
            costs = parts.level3_macro_cost(z3, g4, m4, out)
        else:
            out, pred_emb, _ = dec.generate(z3, z4, g4, m4, cost_model=parts.l3_cost_model)
            costs = parts.level3_macro_cost(z3, g4, m4, out, pred_emb=pred_emb)
        l_cost = D.relaxed_wta(costs)
        l_bc = l3_macro_bc_loss(out, m3_pair, parts.l3_prior_std, keep)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0 or step == 0 or step + 1 == args.steps:
            log.write(json.dumps(dict(step=step + 1, seconds=time.time() - t0,
                                      l_cost=float(l_cost), l_bc=float(l_bc),
                                      best_cost=float(costs.min(1).values.mean()))) + '\n')
            log.flush()
    torch.save(dict(state_dict=dec.state_dict(), args=save_args, git_commit=commit,
                    cache_sha256=config['cache_sha256'],
                    cache_attrs={k: str(v) for k, v in data.attrs.items()},
                    model_checkpoint_sha256=parts.checkpoint_sha256),
               args.out / 'l3_decoder.pt')
    print('TRAIN_DONE', args.out, f'{(time.time() - t0) / args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', flush=True)


def cmd_train_l4(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda'
    parts = L4PlannerParts(device, args.model_seed, args.env, args.family)
    top_level = getattr(parts, 'top_level', 4)
    max_h4 = args.max_h or parts.max_h4
    offset_range = tuple(args.offset_range) if args.offset_range else None
    cache = D.cache_path('train', args.model_seed, args.env, args.family, levels=4)
    data = L4HierarchyData(cache, parts, holdout_every=100, action_block=parts.action_block)
    assert str(data.attrs['checkpoint_sha256']) == parts.checkpoint_sha256
    p_mean = {2: parts.l2_prior_mean, 3: parts.l3_prior_mean, 4: parts.l4_prior_mean}[top_level]
    p_std = {2: parts.l2_prior_std, 3: parts.l3_prior_std, 4: parts.l4_prior_std}[top_level]
    p_lo = {2: parts.l2_lo, 3: parts.l3_lo, 4: parts.l4_lo}[top_level]
    p_hi = {2: parts.l2_hi, 3: parts.l3_hi, 4: parts.l4_hi}[top_level]
    p_cm = {2: parts.l2_cost_model, 3: parts.l3_cost_model, 4: parts.l4_cost_model}[top_level]
    save_args = dict(vars(args), embed_dim=int(data.z4.shape[1]), max_h=max_h4,
                     offset_range=offset_range, levels=4, top_level=top_level)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        net = U.UpperProposal(heads=args.heads, width=args.width, blocks=args.blocks,
                              embed_dim=save_args['embed_dim'], max_h=max_h4).to(device)
    else:
        net = TH.TransformerUpperProposal(
            mode=args.architecture, heads=args.heads, embed_dim=save_args['embed_dim'],
            max_h=max_h4, mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        net.prior_mean.copy_(p_mean)
        net.prior_std.copy_(p_std)
        net.bind_world_model(p_cm)
        save_args.update(d_model=net.d_model, layers=net.layers,
                         attention_heads=net.nhead, ffn_dim=net.ffn_dim,
                         mlp_head_dim=net.mlp_head_dim)
    net.lo.copy_(p_lo)
    net.hi.copy_(p_hi)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(save_args, git_commit=commit, cache=D.relpath(cache),
                  cache_sha256=D.sha256(cache), cache_attrs=data.attrs,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=TH.trainable_parameters(net))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    meta = dict(args=save_args, git_commit=commit, cache_sha256=config['cache_sha256'],
                cache_attrs={k: str(v) for k, v in data.attrs.items()},
                model_checkpoint_sha256=parts.checkpoint_sha256)
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        lr = _cosine_lr(step, args.steps, args.lr)
        for g in opt.param_groups:
            g['lr'] = lr
        h4 = U.MIN_H + step % (max_h4 - U.MIN_H + 1)
        z4, g4, expert, mask, _, _ = data.sample_l4(args.batch, h4, gen, 'train',
                                                    offset_range=offset_range,
                                                    top_level=top_level)
        horizon = torch.full((args.batch,), h4, dtype=torch.long, device=device)
        if is_mlp:
            out = net(z4, g4, horizon)[:, :, :h4]
            costs = parts.top_level_cost(z4, g4, out)
        else:
            out, pred_emb, _ = net.generate(z4, g4, horizon, cost_model=p_cm)
            costs = parts.top_level_cost(z4, g4, out, pred_emb=pred_emb)
        dist = (g4 - z4).pow(2).sum(-1).clamp_min(1.0)
        l_cost = D.relaxed_wta(costs / dist[:, None])
        l_bc = U.macro_bc(out, expert, p_std, mask)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0 or step == 0 or step + 1 == args.steps:
            log.write(json.dumps(dict(step=step + 1, seconds=time.time() - t0, h4=h4,
                                      l_cost=float(l_cost), l_bc=float(l_bc),
                                      best_cost=float(costs.min(1).values.mean()))) + '\n')
            log.flush()
    torch.save(dict(state_dict=net.state_dict(), **meta), args.out / 'proposal.pt')
    print('TRAIN_DONE', args.out, f'{(time.time() - t0) / args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', flush=True)


class Timing4Level(PC.Timing):
    """Extend Timing to record level 1, 2, 3, and 4 seconds per planning call."""

    def install(self):
        patches = super().install()
        original_plan = self.original_plan

        def plan(solver, info_dict, steps_taken, eval_budget):
            if self.first_plan is None:
                self.first_plan = (PC.detached(info_dict), steps_taken, eval_budget,
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
            for level in (1, 2, 3, 4):
                row[f'level{level}_seconds'] = sum(
                    x['seconds'] for x in self.levels[first:] if x['level'] == level)
            row['other_seconds'] = seconds - sum(
                row[k] for k in ('encoding_seconds', 'level1_seconds', 'level2_seconds',
                                 'level3_seconds', 'level4_seconds'))
            self.plans.append(row)
            return result

        patches[0] = patch.object(HierarchicalSolver, 'solve', plan)
        return patches


def cmd_eval(args):
    task_path = (args.task_file.expanduser().resolve() if args.task_file is not None
                 else PC.TASK_FILES[args.env].resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    artifacts = ROOT / 'cost_probe/artifacts'
    cfg = evaluation.load_eval_config(f'{args.env}_l4')
    cfg.cache_dir = str(artifacts)
    cfg.policy = str(D.checkpoint_path(args.model_seed, args.env, args.family, levels=4))
    cfg.seed = args.model_seed
    cfg.load_eval_trajs_path = str(task_path)
    task_sha = PC.sha256(task_path)
    cfg.eval.num_eval = 1
    cfg.eval.chunk_size = 1
    model = evaluation.load_model(cfg)
    model_sha = PC.sha256(Path(cfg.policy))

    dec1_obj = dec1_ck = dec2_obj = dec2_ck = dec3_obj = dec3_ck = up4_obj = up4_ck = None
    if args.l1_decoder is not None:
        dec1_obj, dec1_ck = D.load_decoder(args.l1_decoder, torch.device('cuda'))
    if args.l2_decoder is not None:
        dec2_obj, dec2_ck = load_l2_decoder(args.l2_decoder, torch.device('cuda'))
    if args.l3_decoder is not None:
        dec3_obj, dec3_ck = load_l3_decoder(args.l3_decoder, torch.device('cuda'))
    if args.l4_proposal is not None:
        up4_obj, up4_ck = U.load_upper_proposal(args.l4_proposal, torch.device('cuda'))
    for flag, ck in (('--l1-decoder', dec1_ck), ('--l2-decoder', dec2_ck),
                     ('--l3-decoder', dec3_ck), ('--l4-proposal', up4_ck)):
        if ck is not None and ck.get('model_checkpoint_sha256') != model_sha:
            print(f'WARNING: {flag} was trained for world model {str(ck.get("model_checkpoint_sha256"))[:12]}, '
                  f'loaded {model_sha[:12]}', flush=True)

    def ck_entry(path, ck):
        if ck is None:
            return None
        resolved = path.expanduser().resolve()
        return dict(path=D.relpath(resolved),
                    sha256=PC.sha256(path),
                    architecture=ck.get('args', {}).get('architecture', 'mlp'),
                    git_commit=ck['git_commit'])

    metadata = dict(
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                               cwd=ROOT, text=True).strip()),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        dependencies={name: version(name) for name in ('transformers', 'torchvision', 'gymnasium', 'pymunk', 'numpy')},
        gpu=torch.cuda.get_device_name(), gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checkpoint_sha256=model_sha, env=args.env, family=args.family, levels=4,
        model_seed=args.model_seed, planner_seed=int(cfg.seed),
        tasks_sha256=task_sha, tasks=list(range(args.tasks)),
        action_grads_only=args.action_grads_only,
        bypass_iters=args.bypass_iters,
        l1_decoder=ck_entry(args.l1_decoder, dec1_ck),
        l2_decoder=ck_entry(args.l2_decoder, dec2_ck),
        l3_decoder=ck_entry(args.l3_decoder, dec3_ck),
        l4_proposal=ck_entry(args.l4_proposal, up4_ck),
    )
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    print('METADATA', json.dumps(metadata), flush=True)

    hp = cfg.hierarchical_plan_config
    cfg_top_level = 4 if int(hp.level4.horizon) > 1 else (3 if int(hp.level3.horizon) > 1 else 2)
    top_level = int((up4_ck or {}).get('args', {}).get('top_level', cfg_top_level))
    patches = []
    if args.bypass_iters is not None:
        p = patch.object(GradientSolver, 'solve',
                         bypass_budget_4level(GradientSolver.solve, args.bypass_iters, top_level=top_level))
        p.start(); patches.append(p)
    w1 = w2 = w3 = w4 = None
    if dec1_obj is not None:
        w1 = PC.decoder_solve(GradientSolver.solve, dec1_obj)
        p = patch.object(GradientSolver, 'solve', w1); p.start(); patches.append(p)
    if dec2_obj is not None:
        w2 = l2_decoder_solve(GradientSolver.solve, dec2_obj)
        p = patch.object(GradientSolver, 'solve', w2); p.start(); patches.append(p)
    if dec3_obj is not None:
        w3 = l3_decoder_solve(GradientSolver.solve, dec3_obj)
        p = patch.object(GradientSolver, 'solve', w3); p.start(); patches.append(p)
    if up4_obj is not None:
        w4 = l4_proposal_solve(GradientSolver.solve, up4_obj, top_level=top_level)
        p = patch.object(GradientSolver, 'solve', w4); p.start(); patches.append(p)

    use_variant = None
    if args.action_grads_only:
        restriction, solve, backward = PC.action_grads_only(GradientSolver.solve, torch.Tensor.backward)
        for item in (patch.object(GradientSolver, 'solve', solve),
                     patch.object(torch.Tensor, 'backward', backward)):
            item.start(); patches.append(item)

        def use_variant(variant):
            restriction.enabled = variant
            return variant

    timing = Timing4Level()
    for item in timing.install():
        item.start(); patches.append(item)
    try:
        print('WARMUP_START', flush=True)
        evaluation.run_planning_eval(cfg, model=model, results_dir=args.out / 'warmup')
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            parity = PC.check_timing_parity(cfg, model, timing)
            if use_variant is not None:
                info, steps, budget, action_space, plan_config = timing.first_plan
                timing.enabled = False
                actions = []
                for variant in (False, True):
                    use_variant(variant)
                    s = evaluation.build_solver(cfg, model)
                    s.configure(action_space=action_space, n_envs=1, config=plan_config)
                    actions.append(s.solve(PC.detached(info), steps, budget)['actions'])
                use_variant(True)
                timing.clear()
                parity['released_vs_variant'] = dict(actions_bit_exact=bool(torch.equal(*actions)))
        if w1 is not None:
            parity['decoder_cost_vs_planner_info'] = w1.check.get('max_abs_diff')
            assert parity['decoder_cost_vs_planner_info'] == 0.0
        if w2 is not None and (top_level >= 3 or 'max_abs_diff' in w2.check):
            parity['l2_decoder_cost_vs_planner_info'] = w2.check.get('max_abs_diff')
            assert parity['l2_decoder_cost_vs_planner_info'] == 0.0
        if w3 is not None and (top_level >= 4 or 'max_abs_diff' in w3.check):
            parity['l3_decoder_cost_vs_planner_info'] = w3.check.get('max_abs_diff')
            assert parity['l3_decoder_cost_vs_planner_info'] == 0.0
        if w4 is not None:
            parity['l4_proposal_cost_vs_planner_info'] = w4.check.get('max_abs_diff')
            assert parity['l4_proposal_cost_vs_planner_info'] == 0.0
        (args.out / 'parity.json').write_text(json.dumps(parity, indent=2))
        print('PARITY', json.dumps(parity), flush=True)

        cfg.eval.num_eval = args.tasks
        timing.enabled = True
        torch.cuda.reset_peak_memory_stats()
        metrics = evaluation.run_planning_eval(cfg, model=model, results_dir=args.out / 'evaluation')
        total = sum(r['seconds'] for r in timing.plans)
        s_l1 = sum(r['level1_seconds'] for r in timing.plans)
        s_l2 = sum(r['level2_seconds'] for r in timing.plans)
        s_l3 = sum(r['level3_seconds'] for r in timing.plans)
        s_l4 = sum(r['level4_seconds'] for r in timing.plans)
        latencies = np.array([r['seconds'] for r in timing.plans])
        summary = dict(
            planning_calls=len(timing.plans),
            evaluation_seconds=metrics['evaluation_time'],
            planning_seconds=total,
            lower_seconds=s_l1,
            level1_seconds=s_l1,
            level2_seconds=s_l2,
            level3_seconds=s_l3,
            level4_seconds=s_l4,
            upper_seconds=s_l4,
            encoding_seconds=sum(r['encoding_seconds'] for r in timing.plans),
            median_plan_seconds=float(np.median(latencies)),
            p95_plan_seconds=float(np.quantile(latencies, 0.95)),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
            level_iterations={str(lv): [n for r in timing.levels if r['level'] == lv for n in r['n_iters']]
                              for lv in (1, 2, 3, 4)},
        )
        (args.out / 'planning_summary.json').write_text(json.dumps(summary, indent=2))
        (args.out / 'planning_calls.json').write_text(json.dumps(timing.plans, indent=2))
        (args.out / 'level_calls.json').write_text(json.dumps(timing.levels, indent=2))
        with (args.out / 'planning_calls.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing.plans[0]))
            writer.writeheader()
            writer.writerows(timing.plans)
        report = dict(metadata=metadata, parity=parity, planning=summary,
                      evaluation_metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                          for k, v in metrics.items()})
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        print('PROFILE_COMPLETE', args.out, flush=True)
    finally:
        for item in reversed(patches):
            item.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    for name, default_heads, default_bc in (
        ('train-l1', 16, 1.0),
        ('train-l2', 16, 1.0),
        ('train-l3', 16, 1.0),
        ('train-l4', 32, 0.05),
    ):
        sp = sub.add_parser(name)
        sp.add_argument('--out', type=Path, required=True)
        sp.add_argument('--steps', type=int, default=30000)
        sp.add_argument('--batch', type=int, default=256)
        sp.add_argument('--heads', type=int, default=default_heads)
        sp.add_argument('--lambda-bc', type=float, default=default_bc)
        sp.add_argument('--lr', type=float, default=3e-4)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--width', type=int, default=1024)
        sp.add_argument('--blocks', type=int, default=4)
        sp.add_argument('--model-seed', type=int, default=42)
        sp.add_argument('--env', default='pusht', choices=('pusht', 'fourroom', 'cube', 'ant'))
        sp.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
        sp.add_argument('--architecture', default='mlp', choices=TH.ARCHS)
        if name == 'train-l4':
            sp.add_argument('--max-h', type=int, default=None)
            sp.add_argument('--offset-range', type=int, nargs=2, default=None, metavar=('LO', 'HI'))

    ep = sub.add_parser('eval')
    ep.add_argument('--out', type=Path, required=True)
    ep.add_argument('--tasks', type=int, default=50)
    ep.add_argument('--task-file', type=Path, default=None)
    ep.add_argument('--model-seed', type=int, default=42)
    ep.add_argument('--env', default='pusht', choices=('pusht', 'fourroom', 'cube', 'ant'))
    ep.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
    ep.add_argument('--action-grads-only', action='store_true')
    ep.add_argument('--bypass-iters', type=int, default=None)
    ep.add_argument('--l1-decoder', type=Path, default=None)
    ep.add_argument('--l2-decoder', type=Path, default=None)
    ep.add_argument('--l3-decoder', type=Path, default=None)
    ep.add_argument('--l4-proposal', type=Path, default=None)

    args = p.parse_args()
    {'train-l1': cmd_train_l1, 'train-l2': cmd_train_l2,
     'train-l3': cmd_train_l3, 'train-l4': cmd_train_l4, 'eval': cmd_eval}[args.cmd](args)


if __name__ == '__main__':
    main()
