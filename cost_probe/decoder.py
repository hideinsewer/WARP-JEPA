"""Offline level-1 Action Decoder for the H-JEPA Push-T cost probe.

PlannerParts wraps the released objects (LevelCostModels, level-2 JEPA, level-2
latent-action prior/clip bounds, action StandardScaler) exactly as the planner
uses them; ActionDecoder predicts K candidate level-1 macro-action plans.
"""

import hashlib
import os
from pathlib import Path
import sys

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'h_jepa')]

import planning_eval as evaluation

WTA_EPS = 0.05
AUG_PROB = 0.5
AUG_PERTURB_STD = 0.5


def relpath(path):
    """Path as recorded in outputs: relative to the repository root, or to the cwd outside it."""
    resolved = Path(path).expanduser().resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else os.path.relpath(resolved)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(model_seed=42, env='pusht', family='hjepa', levels=2):
    """Released or self-trained checkpoint of a training seed for an env/model family."""
    return ROOT / f'cost_probe/artifacts/models/{env}/{env}_{family}_l{levels}/seed{model_seed}/{env}_{family}_l{levels}_object.ckpt'


def cache_path(split, model_seed=42, env='pusht', family='hjepa', levels=2):
    """Level-1 embedding cache of a split; two-level Push-T/hjepa keeps the original names."""
    if levels == 2 and env == 'pusht' and family == 'hjepa':
        suffix = '' if model_seed == 42 else f'_seed{model_seed}'
        return ROOT / f'cost_probe/artifacts/data/cache/pusht_l1_{split}{suffix}.h5'
    if levels == 2:
        return ROOT / f'cost_probe/artifacts/data/cache/{env}_{family}_l1_{split}_seed{model_seed}.h5'
    return ROOT / f'cost_probe/artifacts/data/cache/{env}_{family}_l{levels}_l1_{split}_seed{model_seed}.h5'


class PlannerParts:
    """The released planner pieces the decoder imitates, all parameters frozen."""

    def __init__(self, device='cuda', model_seed=42, env='pusht', family='hjepa'):
        artifacts = ROOT / 'cost_probe/artifacts'
        cfg = evaluation.load_eval_config(f'{env}_l2')
        cfg.cache_dir = str(artifacts)
        cfg.policy = str(checkpoint_path(model_seed, env, family))
        cfg.seed = model_seed
        self.env, self.family = env, family
        self.model_seed, self.checkpoint_sha256 = model_seed, sha256(cfg.policy)
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
        if solver.level_solvers[1]._get_action_clip_bounds() is not None:
            raise RuntimeError('level-1 solver has action clip bounds; decoder assumptions invalid')
        self.cfg, self.model, self.solver = cfg, model, solver
        self.l1_cost_model = solver.level_solvers[1].model
        self.l2_cost_model = solver.level_solvers[2].model
        self.level2_jepa = solver.level_models[1]
        mean, std = solver.level_solvers[2]._get_action_prior_stats()
        self.prior_mean, self.prior_std = mean.to(device), std.to(device)
        lo, hi = solver.level_solvers[2]._get_action_clip_bounds()
        self.l2_lo, self.l2_hi = lo.to(device), hi.to(device)
        self.action_mean = torch.as_tensor(self.action_scaler.mean_, dtype=torch.float32, device=device)
        self.action_std = torch.as_tensor(self.action_scaler.scale_, dtype=torch.float32, device=device)
        level1 = plan_config.level1
        self.action_block = int(level1.action_block)
        self.action_cost_weight = float(level1.action_cost_weight)
        self.cost_last_n = int(level1.cost_last_n)
        self.intermediate_cost_weight = float(level1.intermediate_cost_weight)
        self.device = device

    @torch.no_grad()
    def encode_level2(self, z1):
        """(B,192) level-1 latent -> (B,192) level-2 latent, as _encode_all_levels does."""
        return encode_level2(self.level2_jepa, z1)

    @torch.no_grad()
    def subgoal(self, z2, m):
        """(B,192) level-2 latent + (B,4) macro -> (B,192) level-2 first prediction."""
        info = {'embed_0': z2[:, None, None], 'pixels': z2.new_zeros(z2.shape[0], 1, 1)}
        out = self.l2_cost_model.rollout(info, m.view(-1, 1, 1, 4))
        return out['predicted_embed_0'][:, 0, 1]

    @torch.no_grad()
    def data_macro(self, z1_seq, blocks):
        """(B,3,192) z1 at t/t+5/t+10 + (B,2,10) blocks -> (B,4) pooled data macro."""
        pooled = self.l1_cost_model._encode_actions_up(
            z1_seq[:, None], blocks[:, None], pad=0, dense=False)
        return pooled[:, 0, 0]

    def level1_cost(self, z1, g2, m, cand):
        """Lean macro-conditioned level-1 cost, (B,K,2,10) -> (B,K), grad w.r.t. cand."""
        return level1_cost(self.l1_cost_model, z1, g2, m, cand,
                           action_cost_weight=self.action_cost_weight,
                           cost_last_n=self.cost_last_n,
                           intermediate_cost_weight=self.intermediate_cost_weight)


@torch.no_grad()
def encode_level2(level2_jepa, z1):
    """(B,192) level-1 latent -> (B,192) level-2 latent, as _encode_all_levels does."""
    out = level2_jepa.encode({'embed_1': z1[:, None]}, key='embed_1',
                             chunk_temporal_inputs=True)
    return out['embed_0'][:, -1]


def level1_cost(l1_cost_model, z1, g2, m, cand, *, action_cost_weight,
                cost_last_n, intermediate_cost_weight):
    """Lean macro-conditioned level-1 cost, (B,K,2,10) -> (B,K), grad w.r.t. cand."""
    b, k = cand.shape[:2]
    info = {'pixels': z1.new_zeros(b, k, 1),
            'embed_0': z1[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0': g2[:, None, None].expand(b, k, 1, -1),
            'goal_action': m[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0_in_upper_space': True,
            'action_cost_weight': action_cost_weight,
            'cost_last_n': cost_last_n,
            'intermediate_cost_weight': intermediate_cost_weight}
    return l1_cost_model.get_cost(info, cand)


def load_decoder(path, device):
    """Load a decoder checkpoint -> (decoder module in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck['args']
    arch = a.get('architecture', 'mlp')
    if arch == 'mlp':
        dec = ActionDecoder(heads=a['heads'], width=a['width'], blocks=a['blocks'],
                            embed_dim=a.get('embed_dim', 192),
                            block_dim=a.get('block_dim', 10)).to(device)
    else:
        import transformer_heads as TH
        mlp_head_dim = a.get('mlp_head_dim',
                             TH.DEFAULT_MLP_HEAD_DIM if 'out_mlp.0.weight' in ck['state_dict'] else 0)
        dec = TH.TransformerActionDecoder(
            mode=arch, heads=a['heads'],
            d_model=a.get('d_model', TH.DEFAULT_D_MODEL),
            layers=a.get('layers', TH.DEFAULT_LAYERS),
            nhead=a.get('attention_heads', TH.DEFAULT_ATTN_HEADS),
            ffn_dim=a.get('ffn_dim'),
            mlp_head_dim=mlp_head_dim,
            embed_dim=a.get('embed_dim', 192),
            block_dim=a.get('block_dim', 10)).to(device)
    dec.load_state_dict(ck['state_dict'])
    dec.eval()
    return dec, ck


class ActionDecoder(nn.Module):
    """MLP head-ensemble mapping (z1, z2, g2, m) -> (B, heads, 2, 10) action blocks."""

    def __init__(self, heads=16, width=1024, blocks=4, embed_dim=192, macro_dim=4,
                 block_dim=10):
        super().__init__()
        self.architecture = 'mlp'
        self.heads = heads
        self.block_dim = block_dim
        self.ln_z1 = nn.LayerNorm(embed_dim)
        self.ln_z2 = nn.LayerNorm(embed_dim)
        self.ln_g2 = nn.LayerNorm(embed_dim)
        self.ln_d = nn.LayerNorm(embed_dim)
        self.register_buffer('prior_mean', torch.zeros(macro_dim))
        self.register_buffer('prior_std', torch.ones(macro_dim))
        self.input = nn.Linear(4 * embed_dim + macro_dim, width)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict(dict(
                ln=nn.LayerNorm(width), f1=nn.Linear(width, width), f2=nn.Linear(width, width))))
        self.out_ln = nn.LayerNorm(width)
        self.output = nn.Linear(width, heads * 2 * block_dim)

    def forward(self, z1, z2, g2, m):
        x = torch.cat([self.ln_z1(z1), self.ln_z2(z2), self.ln_g2(g2), self.ln_d(g2 - z2),
                       (m - self.prior_mean) / self.prior_std], dim=-1)
        h = self.input(x)
        for block in self.blocks:
            h = h + block['f2'](F.gelu(block['f1'](block['ln'](h))))
        return self.output(self.out_ln(h)).view(-1, self.heads, 2, self.block_dim)


class CacheData:
    """Valid (episode-continuous) frames of a level-1 cache h5, on-device."""

    def __init__(self, path, device, holdout_every=100, holdout=False, scaler=None,
                 action_block=5, embed_device=None):
        with h5py.File(path, 'r') as f:
            self.embed = torch.from_numpy(f['embed'][:]).to(embed_device or device)
            self.action = torch.from_numpy(f['action'][:]).to(device)
            episode = torch.from_numpy(np.asarray(f['episode'][:])).to(device)
            step = torch.from_numpy(np.asarray(f['step'][:])).to(device)
            self.attrs = dict(f.attrs)
        self.device = device
        self.action_block = action_block
        self.action_dim = int(self.action.shape[1])
        self.span = 2 * action_block
        self.scaler_mean = self.scaler_std = None
        if scaler is not None:
            self.scaler_mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
            self.scaler_std = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        span = self.span
        valid = (episode[span:] == episode[:-span]) & (step[span:] == step[:-span] + span)
        self.valid = torch.nonzero(valid, as_tuple=True)[0]
        if holdout_every is not None:
            keep = (episode[self.valid] % holdout_every == 0) if holdout else \
                (episode[self.valid] % holdout_every != 0)
            self.valid = self.valid[keep]

    def normalized_blocks(self, indices):
        """(n,) frame indices -> (n,2,action_block*action_dim): two normalized action
        blocks, row-major."""
        offsets = torch.arange(self.span, device=self.device)
        raw = self.action[indices[:, None] + offsets]  # (n,span,action_dim)
        if self.scaler_mean is not None:
            raw = (raw - self.scaler_mean) / self.scaler_std
        return raw.reshape(-1, 2, self.action_block * self.action_dim)

    def sample_batch(self, n, generator=None):
        """n valid frames -> (z1 (n,192), z1_seq (n,3,192), blocks (n,2,block_dim), idx)."""
        pick = torch.randint(len(self.valid), (n,), device=self.device, generator=generator)
        idx = self.valid[pick]
        if self.embed.device.type == 'cpu':
            hidx = idx.cpu()
            z1 = self.embed[hidx].to(self.device, non_blocking=True)
            offsets = torch.tensor([0, self.action_block, self.span])
            z1_seq = self.embed[hidx[:, None] + offsets].to(self.device,
                                                          non_blocking=True)
        else:
            z1 = self.embed[idx]
            offsets = torch.tensor([0, self.action_block, self.span], device=self.device)
            z1_seq = self.embed[idx[:, None] + offsets]
        return z1, z1_seq, self.normalized_blocks(idx), idx


def relaxed_wta(costs, eps=WTA_EPS):
    """Per-sample relaxed winner-take-all: (1-eps)*min_k + eps*mean_k, then mean over B."""
    return ((1 - eps) * costs.min(dim=1).values + eps * costs.mean(dim=1)).mean()


def bc_loss(outputs, target, mask=None):
    """min_k sum of squared errors over the 20 action dims vs the data blocks."""
    sse = (outputs - target[:, None]).pow(2).flatten(2).sum(-1)  # (B,K)
    best = sse.min(dim=1).values
    if mask is not None:
        return best[mask].mean() if mask.any() else best.sum() * 0.0
    return best.mean()


def augment_macros(parts, z1_seq, blocks, m_data, generator):
    """With prob AUG_PROB replace the macro: half fresh prior draws, half perturbed data."""
    b = m_data.shape[0]
    noise = torch.randn(b, 4, device=m_data.device, generator=generator)
    fresh = parts.prior_mean + parts.prior_std * noise
    perturbed = m_data + AUG_PERTURB_STD * parts.prior_std * noise
    fresh = torch.minimum(torch.maximum(fresh, parts.l2_lo), parts.l2_hi)
    perturbed = torch.minimum(torch.maximum(perturbed, parts.l2_lo), parts.l2_hi)
    use_fresh = torch.rand(b, device=m_data.device, generator=generator) < 0.5
    replace = torch.rand(b, device=m_data.device, generator=generator) < AUG_PROB
    m = torch.where(use_fresh[:, None], fresh, perturbed)
    m = torch.where(replace[:, None], m, m_data)
    return m, ~replace
