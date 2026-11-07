"""Level-2 proposal network for the H-JEPA Push-T cost probe.

UpperProposal maps the current level-2 latent z2, the goal's level-2 latent g2 and the
level-2 planning horizon h to K candidate macro-action plans inside the planner's level-2
clip box; the candidate with the lowest planner level-2 cost is used. L2Data serves
training and evaluation problems from a frozen level-1 cache.
"""

from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

MIN_H, MAX_H = 2, 8
MACRO_DIM = 4
STEPS_PER_MACRO = 10
MATCH_PROB = 0.5
JITTER = (-30, 20)


def level2_cost(l2_cost_model, z2, g2, cand, *, action_cost_weight, cost_last_n,
                intermediate_cost_weight):
    """Lean top-level cost of (B,K,h,4) macro plans from z2 toward g2 -> (B,K)."""
    b, k = cand.shape[:2]
    info = {'pixels': z2.new_zeros(b, k, 1),
            'embed_0': z2[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0': g2[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0_level': 2,
            'goal_embed_0_in_upper_space': False,
            'dense_upper_goal_projection': False,
            'action_cost_weight': action_cost_weight,
            'cost_last_n': cost_last_n,
            'intermediate_cost_weight': intermediate_cost_weight}
    return l2_cost_model.get_cost(info, cand)


def level2_cost_kwargs(parts):
    """Level-2 cost settings of the planner config held by decoder.PlannerParts."""
    cfg = parts.solver._level_plan_config(2)
    return dict(action_cost_weight=float(cfg.action_cost_weight), cost_last_n=int(cfg.cost_last_n),
                intermediate_cost_weight=float(cfg.intermediate_cost_weight))


class UpperProposal(nn.Module):
    """MLP head-ensemble mapping (z2, g2, h) -> (B, heads, MAX_H, 4) plans inside [lo, hi]."""

    def __init__(self, heads=32, width=1024, blocks=4, embed_dim=192, max_h=MAX_H):
        super().__init__()
        self.architecture = 'mlp'
        self.heads = heads
        self.max_h = max_h
        self.ln_z2 = nn.LayerNorm(embed_dim)
        self.ln_g2 = nn.LayerNorm(embed_dim)
        self.ln_d = nn.LayerNorm(embed_dim)
        self.register_buffer('lo', torch.full((MACRO_DIM,), -1.))
        self.register_buffer('hi', torch.full((MACRO_DIM,), 1.))
        self.input = nn.Linear(3 * embed_dim + max_h - MIN_H + 1, width)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict(dict(
                ln=nn.LayerNorm(width), f1=nn.Linear(width, width), f2=nn.Linear(width, width))))
        self.out_ln = nn.LayerNorm(width)
        self.output = nn.Linear(width, heads * max_h * MACRO_DIM)

    def forward(self, z2, g2, h):
        """h: (B,) int64 horizons in [MIN_H, max_h]; returns all max_h macros (use [:, :, :h])."""
        horizon = F.one_hot(h - MIN_H, self.max_h - MIN_H + 1).to(z2.dtype)
        x = torch.cat([self.ln_z2(z2), self.ln_g2(g2), self.ln_d(g2 - z2), horizon], dim=-1)
        y = self.input(x)
        for block in self.blocks:
            y = y + block['f2'](F.gelu(block['f1'](block['ln'](y))))
        raw = self.output(self.out_ln(y)).view(-1, self.heads, self.max_h, MACRO_DIM)
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)


def load_upper_proposal(path, device):
    """Load a proposal checkpoint -> (proposal module in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck['args']
    arch = a.get('architecture', 'mlp')
    if arch == 'mlp':
        net = UpperProposal(heads=a['heads'], width=a['width'], blocks=a['blocks'],
                            embed_dim=a.get('embed_dim', 192),
                            max_h=a.get('max_h', MAX_H)).to(device)
    else:
        import transformer_heads as TH
        mlp_head_dim = a.get('mlp_head_dim',
                             TH.DEFAULT_MLP_HEAD_DIM if 'out_mlp.0.weight' in ck['state_dict'] else 0)
        net = TH.TransformerUpperProposal(
            mode=arch, heads=a['heads'],
            d_model=a.get('d_model', TH.DEFAULT_D_MODEL),
            layers=a.get('layers', TH.DEFAULT_LAYERS),
            nhead=a.get('attention_heads', TH.DEFAULT_ATTN_HEADS),
            ffn_dim=a.get('ffn_dim'),
            mlp_head_dim=mlp_head_dim,
            embed_dim=a.get('embed_dim', 192),
            max_h=a.get('max_h', MAX_H)).to(device)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck


def macro_bc(out, expert, prior_std, mask):
    """min over heads of the per-macro squared error in prior-std units (sum over the 4 dims,
    mean over the h macros), averaged over the masked samples. out (B,K,h,4), expert (B,h,4)."""
    err = ((out - expert[:, None]) / prior_std).pow(2).sum(-1).mean(-1)
    best = err.min(dim=1).values
    return best[mask].mean() if mask.any() else best.sum() * 0.0


def run_remaining(episode, step):
    """Frames left after each frame in its contiguous run (same episode, consecutive steps)."""
    n = len(episode)
    new_run = torch.ones(n, dtype=torch.bool, device=episode.device)
    new_run[1:] = (episode[1:] != episode[:-1]) | (step[1:] != step[:-1] + 1)
    run_id = torch.cumsum(new_run.long(), 0) - 1
    index = torch.arange(n, device=episode.device)
    run_end = torch.zeros(int(run_id[-1]) + 1, dtype=torch.long, device=episode.device)
    run_end = run_end.scatter_reduce(0, run_id, index, 'amax', include_self=False)
    return run_end[run_id] - index


class L2Data:
    """Level-2 problems from a level-1 cache h5, on-device.

    z2[t] is the level-2 encoding of frame t; macro[t] is the planner's level-2 action
    encoding of the data actions t..t+9 (NaN where the run has fewer than 10 frames left).
    A problem at horizon h is (z2[t], z2[t + offset], h) with expert plan
    macro[t], macro[t+10], ..., macro[t+10(h-1)]. Splits: 'train' (episode % holdout_every
    != 0) and 'heldout' (== 0), or 'all' when holdout_every is None.
    """

    def __init__(self, path, parts, holdout_every=100, chunk=65536, action_block=5,
                 z2_device=None):
        dev = parts.device
        self.device = dev
        cpu = z2_device == 'cpu'
        with h5py.File(path, 'r') as f:
            embed = torch.from_numpy(f['embed'][:]).to('cpu' if cpu else dev)
            action = torch.from_numpy(f['action'][:]).to(dev)
            episode = torch.from_numpy(np.asarray(f['episode'][:])).to(dev)
            step = torch.from_numpy(np.asarray(f['step'][:])).to(dev)
            self.attrs = dict(f.attrs)
        n = len(episode)
        self.action_block = action_block
        self.span = 2 * action_block
        self.remaining = run_remaining(episode, step)
        if cpu:
            self.z2 = None
            for i in range(0, n, chunk):
                z = parts.encode_level2(embed[i:i + chunk].to(dev, non_blocking=True))
                if self.z2 is None:
                    self.z2 = torch.empty((n, z.shape[1]), dtype=z.dtype)
                self.z2[i:i + z.shape[0]] = z.cpu()
        else:
            self.z2 = torch.cat(
                [parts.encode_level2(embed[i:i + chunk]) for i in range(0, n, chunk)])
        self.macro = torch.full((n, MACRO_DIM), float('nan'), device=dev)
        has = torch.nonzero(self.remaining >= self.span, as_tuple=True)[0]
        offsets = torch.tensor([0, action_block, self.span], device=dev)
        actions = torch.arange(self.span, device=dev)
        block_dim = action_block * int(action.shape[1])
        for i in range(0, len(has), chunk):
            t = has[i:i + chunk]
            blocks = ((action[t[:, None] + actions] - parts.action_mean) / parts.action_std).reshape(-1, 2, block_dim)
            if cpu:
                z1_seq = embed[t.cpu()[:, None] + offsets.cpu()].to(dev,
                                                                    non_blocking=True)
            else:
                z1_seq = embed[t[:, None] + offsets]
            self.macro[t] = parts.data_macro(z1_seq, blocks)
        del embed
        self.episode = episode
        self.num_frames = n
        if holdout_every is None:
            masks = {'all': torch.ones(n, dtype=torch.bool, device=dev)}
        else:
            masks = {'train': episode % holdout_every != 0, 'heldout': episode % holdout_every == 0}
        self.splits = {}
        for name, keep in masks.items():
            frames = torch.nonzero(keep, as_tuple=True)[0]
            frames = frames[torch.argsort(self.remaining[frames], stable=True)]
            self.splits[name] = (frames, self.remaining[frames].contiguous())

    def frames_with(self, need, split):
        """Frames of a split with at least `need` frames left in their run (ascending index)."""
        frames, remaining = self.splits[split]
        start = int(torch.searchsorted(remaining, torch.tensor(need, device=frames.device)))
        return torch.sort(frames[start:]).values

    def problems(self, t, h, offset):
        """Frames t (n,), horizon h, goal offsets (n,) -> z2 (n,192), g2 (n,192), expert (n,h,4).
        Expert rows are only meaningful where t + 10h stays inside the run (NaN-free there)."""
        steps = (t[:, None] + self.span * torch.arange(h, device=t.device)).clamp_max(self.num_frames - 1)
        if self.z2.device.type == 'cpu':
            return (self.z2[t.cpu()].to(self.device, non_blocking=True),
                    self.z2[(t + offset).cpu()].to(self.device, non_blocking=True),
                    self.macro[steps])
        return self.z2[t], self.z2[t + offset], self.macro[steps]

    def sample(self, n, h, generator, split, match_prob=MATCH_PROB, jitter=JITTER,
               offset_range=None):
        """n training problems at horizon h from a split.

        With offset_range=None (the Push-T rule): with probability match_prob the goal
        offset is 2*action_block*h - 5 (the remaining distance of a planner call that is
        on the expert's pace: goal 75 steps ahead, calls every 10 steps, h = 8 - call
        index) and the expert plan is a valid imitation target (mask True); otherwise the
        offset is uniform in [max(5, base + jitter[0]), base + jitter[1]] and the mask is
        False. With offset_range=(lo, hi) the offset is uniform in [lo, hi] and
        mask = offset >= 2*action_block*h - 5 (the expert's first h macros do not pass the
        goal). Frames are uniform among those whose run covers the need (2*action_block*h
        frames when the expert is a valid target, the offset otherwise).
        Returns z2, g2, expert (zeros where mask is False), mask, offset, t.
        """
        dev = self.device
        frames, remaining = self.splits[split]
        base = self.span * h - 5
        if offset_range is not None:
            lo, hi = offset_range
            offset = torch.randint(lo, hi + 1, (n,), device=dev, generator=generator)
            mask = offset >= base
        else:
            mask = torch.rand(n, device=dev, generator=generator) < match_prob
            jittered = torch.randint(max(5, base + jitter[0]), base + jitter[1] + 1, (n,),
                                     device=dev, generator=generator)
            offset = torch.where(mask, torch.full_like(jittered, base), jittered)
        need = torch.where(mask, torch.maximum(offset, torch.full_like(offset, self.span * h)),
                           offset)
        start = torch.searchsorted(remaining, need)
        count = len(frames) - start
        assert bool((count > 0).all()), 'no frame covers the requested goal offset'
        pick = torch.minimum((torch.rand(n, device=dev, generator=generator) * count).long(), count - 1)
        t = frames[start + pick]
        z2, g2, expert = self.problems(t, h, offset)
        expert = torch.where(mask[:, None, None], expert, torch.zeros_like(expert))
        return z2, g2, expert, mask, offset, t
