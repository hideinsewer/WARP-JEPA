"""Conditional Transformer decoder heads for H-JEPA level-2 proposals and level-1 action decoding.

Supports three generation modes with identical module structure and parameter counts per (env, role):
  - tf_parallel: single forward pass over h step queries (initial state, empty action prefix, causal mask)
  - tf_ar_action: step-by-step autoregressive generation feeding back generated actions (initial state)
  - tf_ar_wm: step-by-step autoregressive generation advancing the frozen world model F at each step
"""

import time
import torch
import torch.nn.functional as F
from torch import nn

MIN_H, MAX_H = 2, 8
MACRO_DIM = 4
LOWER_H = 2
TF_MODES = ('tf_parallel', 'tf_ar_action', 'tf_ar_wm')
ARCHS = ('mlp',) + TF_MODES
DEFAULT_LAYERS = 1
DEFAULT_D_MODEL = 384
DEFAULT_ATTN_HEADS = 6
DEFAULT_FFN_DIM = 1536
DEFAULT_MLP_HEAD_DIM = 1536
FFN_CANDIDATES = tuple(range(512, 3073, 64))


def trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def mlp_upper_param_count(heads=32, width=1024, blocks=4, embed_dim=192, max_h=MAX_H,
                          min_h=MIN_H, macro_dim=MACRO_DIM):
    in_dim = 3 * embed_dim + max_h - min_h + 1
    out_dim = heads * max_h * macro_dim
    ln_in = 3 * (2 * embed_dim)
    input_lin = in_dim * width + width
    res_blocks = blocks * (2 * width + 2 * (width * width + width))
    out_ln = 2 * width
    out_lin = width * out_dim + out_dim
    return ln_in + input_lin + res_blocks + out_ln + out_lin


def mlp_lower_param_count(heads=16, width=1024, blocks=4, embed_dim=192,
                          macro_dim=MACRO_DIM, block_dim=10):
    in_dim = 4 * embed_dim + macro_dim
    out_dim = heads * LOWER_H * block_dim
    ln_in = 4 * (2 * embed_dim)
    input_lin = in_dim * width + width
    res_blocks = blocks * (2 * width + 2 * (width * width + width))
    out_ln = 2 * width
    out_lin = width * out_dim + out_dim
    return ln_in + input_lin + res_blocks + out_ln + out_lin


def advance_jepa(jepa, emb, act, a_t, h_total):
    """Advance a frozen JEPA predictor by one step matching JEPA.rollout(history_size=1) bit-for-bit.

    emb: (B*S, t+1, D) or None (initialized from z0); act: (B*S, t, A) or None;
    a_t: (B, S, A); h_total: total horizon H so the step_view stride matches torch.split(cand, [1, H-1], dim=2).
    Returns (emb (B*S, t+2, D), act (B*S, t+1, A), next_state (B, S, D)).
    """
    b, s, adim = a_t.shape
    padded = (torch.cat([a_t[:, :, None], a_t.new_zeros(b, s, h_total - 1, adim)], dim=2)
              if h_total > 1 else a_t[:, :, None])
    step_view = padded[:, :, :1].reshape(b * s, 1, adim)
    act = step_view if act is None else torch.cat([act, step_view], dim=1)
    act_emb = jepa.action_embed(act)
    pred_emb = jepa.predict(emb[:, -1:], act_emb[:, -1:])[:, -1:]
    emb = torch.cat([emb, pred_emb], dim=1)
    return emb, act, pred_emb.view(b, s, -1)


def cost_from_rollout(cost_model, info_dict, action_candidates, pred_emb):
    """Compute LevelCostModel cost from an already rolled-out predicted_embed_0 (B, K, H+1, D)."""
    use_upper_space = bool(info_dict.get('goal_embed_0_in_upper_space', True))
    goal_level = int(info_dict.get('goal_embed_0_level', cost_model.level + 1))
    action_weight = float(info_dict.get('action_cost_weight', 0.0))
    pred_for_cost, action_for_cost = cost_model._project_to_goal_space(
        pred_emb,
        action_candidates if action_weight != 0.0 else None,
        use_upper_space=use_upper_space,
        dense=bool(info_dict.get('dense_upper_goal_projection', False)),
        goal_level=goal_level,
    )
    cost_info = dict(info_dict)
    cost_info['predicted_embed_0'] = pred_for_cost
    cost = cost_model.model.criterion(cost_info)
    if action_weight != 0.0:
        action_cost = cost_model._action_cost(action_for_cost, cost_info)
        if action_cost is not None:
            cost = cost + action_weight * action_cost
    return cost


def level2_cost_from_rollout(l2_cost_model, z2, g2, cand, pred_emb, *, action_cost_weight,
                             cost_last_n, intermediate_cost_weight):
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
    if pred_emb is None:
        pred_emb = l2_cost_model.rollout(dict(info), cand)['predicted_embed_0']
    return cost_from_rollout(l2_cost_model, info, cand, pred_emb)


def level1_cost_from_rollout(l1_cost_model, z1, g2, m, cand, pred_emb, *, action_cost_weight,
                             cost_last_n, intermediate_cost_weight):
    b, k = cand.shape[:2]
    info = {'pixels': z1.new_zeros(b, k, 1),
            'embed_0': z1[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0': g2[:, None, None].expand(b, k, 1, -1),
            'goal_action': m[:, None, None].expand(b, k, 1, -1),
            'goal_embed_0_in_upper_space': True,
            'action_cost_weight': action_cost_weight,
            'cost_last_n': cost_last_n,
            'intermediate_cost_weight': intermediate_cost_weight}
    if pred_emb is None:
        pred_emb = l1_cost_model.rollout(dict(info), cand)['predicted_embed_0']
    return cost_from_rollout(l1_cost_model, info, cand, pred_emb)


class _ConditionalTransformerCore(nn.Module):
    """Shared pre-LN 1-layer causal TransformerDecoder backbone + GELU MLP output head."""

    def __init__(self, *, mode, heads, max_h, action_dim, num_mem_tokens, embed_dim=192,
                 d_model=DEFAULT_D_MODEL, layers=DEFAULT_LAYERS, nhead=DEFAULT_ATTN_HEADS,
                 ffn_dim=DEFAULT_FFN_DIM, mlp_head_dim=DEFAULT_MLP_HEAD_DIM):
        super().__init__()
        if mode not in TF_MODES:
            raise ValueError(f'unsupported transformer mode {mode!r}; expected one of {TF_MODES}')
        self.mode = mode
        self.architecture = mode
        self.heads = int(heads)
        self.max_h = int(max_h)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.layers = int(layers)
        self.nhead = int(nhead)
        self.ffn_dim = int(ffn_dim)
        self.mlp_head_dim = int(mlp_head_dim)

        self.cand_emb = nn.Embedding(self.heads, self.d_model)
        self.mem_type_emb = nn.Parameter(torch.zeros(num_mem_tokens, self.d_model))
        nn.init.normal_(self.mem_type_emb, std=0.02)

        self.step_ln_s = nn.LayerNorm(self.embed_dim)
        self.step_state_proj = nn.Linear(self.embed_dim, self.d_model)
        self.prev_action_proj = nn.Linear(self.action_dim, self.d_model, bias=False)
        self.pos_emb = nn.Embedding(self.max_h, self.d_model)
        self.rem_emb = nn.Embedding(self.max_h, self.d_model)
        self.cand_query_proj = nn.Linear(self.d_model, self.d_model)
        self.bos_emb = nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.bos_emb, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.ffn_dim,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=self.layers, norm=nn.LayerNorm(self.d_model))
        self.out_ln = nn.LayerNorm(self.d_model)
        if self.mlp_head_dim > 0:
            self.out_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.mlp_head_dim),
                nn.GELU(),
                nn.Linear(self.mlp_head_dim, self.action_dim),
            )
        else:
            self.output = nn.Linear(self.d_model, self.action_dim)

    def bind_world_model(self, cost_model):
        """Attach a frozen LevelCostModel without registering it as an nn.Module child."""
        self.__dict__['_cost_model'] = cost_model
        return self

    def _resolve_cost_model(self, cost_model):
        cm = cost_model if cost_model is not None else self.__dict__.get('_cost_model')
        if cm is None and self.mode == 'tf_ar_wm':
            raise RuntimeError('tf_ar_wm requires cost_model (pass cost_model=... or call bind_world_model)')
        return cm

    def _causal_mask(self, length, device):
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _step_query(self, s_t, a_prev_norm, pos_idx, rem_idx, cand_q, is_first):
        q = (self.step_state_proj(self.step_ln_s(s_t))
             + self.prev_action_proj(a_prev_norm)
             + self.pos_emb(pos_idx)
             + self.rem_emb(rem_idx)
             + cand_q)
        if is_first:
            q = q + self.bos_emb
        return q

    def _decode_queries(self, queries, memory):
        mask = self._causal_mask(queries.shape[1], queries.device)
        hidden = self.out_ln(self.decoder(tgt=queries, memory=memory, tgt_mask=mask, tgt_is_causal=True))
        return self.out_mlp(hidden) if self.mlp_head_dim > 0 else self.output(hidden)


class TransformerUpperProposal(_ConditionalTransformerCore):
    """Conditional Transformer decoder mapping (z2, g2, h) -> (B, heads, H, 4) macro plans."""

    def __init__(self, mode='tf_ar_wm', heads=32, d_model=DEFAULT_D_MODEL, layers=DEFAULT_LAYERS,
                 nhead=DEFAULT_ATTN_HEADS, ffn_dim=None, mlp_head_dim=DEFAULT_MLP_HEAD_DIM,
                 embed_dim=192, max_h=MAX_H, mlp_width=1024, mlp_blocks=4):
        if ffn_dim is None:
            ffn_dim = DEFAULT_FFN_DIM if layers == 1 else match_upper_ffn_dim(
                heads=heads, d_model=d_model, layers=layers, nhead=nhead,
                mlp_head_dim=mlp_head_dim, embed_dim=embed_dim, max_h=max_h,
                mlp_width=mlp_width, mlp_blocks=mlp_blocks)
        super().__init__(mode=mode, heads=heads, max_h=max_h, action_dim=MACRO_DIM,
                         num_mem_tokens=5, embed_dim=embed_dim, d_model=d_model,
                         layers=layers, nhead=nhead, ffn_dim=ffn_dim, mlp_head_dim=mlp_head_dim)
        self.register_buffer('lo', torch.full((MACRO_DIM,), -1.))
        self.register_buffer('hi', torch.full((MACRO_DIM,), 1.))
        self.register_buffer('prior_mean', torch.zeros(MACRO_DIM))
        self.register_buffer('prior_std', torch.ones(MACRO_DIM))

        self.ln_z2 = nn.LayerNorm(embed_dim)
        self.ln_g2 = nn.LayerNorm(embed_dim)
        self.ln_d = nn.LayerNorm(embed_dim)
        self.proj_z2 = nn.Linear(embed_dim, d_model)
        self.proj_g2 = nn.Linear(embed_dim, d_model)
        self.proj_d = nn.Linear(embed_dim, d_model)
        self.proj_h = nn.Linear(max_h - MIN_H + 1, d_model)

    def _squash(self, raw):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)

    def _norm_macro(self, m):
        return (m - self.prior_mean) / self.prior_std

    def _memory(self, z2, g2, h):
        b = z2.shape[0]
        k = self.heads
        horizon_oh = F.one_hot(h - MIN_H, self.max_h - MIN_H + 1).to(z2.dtype)
        t0 = self.proj_z2(self.ln_z2(z2))[:, None].expand(b, k, -1)
        t1 = self.proj_g2(self.ln_g2(g2))[:, None].expand(b, k, -1)
        t2 = self.proj_d(self.ln_d(g2 - z2))[:, None].expand(b, k, -1)
        t3 = self.proj_h(horizon_oh)[:, None].expand(b, k, -1)
        t4 = self.cand_emb.weight[None].expand(b, k, -1)
        mem = torch.stack([t0, t1, t2, t3, t4], dim=2).reshape(b * k, 5, self.d_model)
        return mem + self.mem_type_emb[None]

    def generate(self, z2, g2, h, *, cost_model=None, extra_actions=None, pad_to_max_h=False):
        """Generate candidate macro plans.

        Returns (cand, pred_emb, stats):
          - cand: (B, heads (+ extra), H_out, 4) where H_out is max_h if pad_to_max_h else int(h[0])
          - pred_emb: (B, heads (+ extra), h+1, D) when mode == 'tf_ar_wm' and uniform h, else None
          - stats: dict with head_seconds and wm_rollout_seconds
        """
        if bool((h < MIN_H).any() or (h > self.max_h).any()):
            raise RuntimeError(f'horizon out of range [{MIN_H}, {self.max_h}]: {h.tolist()}')
        b, dev, dtype = z2.shape[0], z2.device, z2.dtype
        k = self.heads
        uniform_h = bool((h == h[0]).all())
        h_steps = self.max_h if (pad_to_max_h and not uniform_h) else int(h.max().item())
        if pad_to_max_h and self.mode == 'tf_parallel':
            h_steps = self.max_h

        t_head0 = time.perf_counter()
        memory = self._memory(z2, g2, h)
        cand_q = self.cand_query_proj(self.cand_emb.weight)[None].expand(b, k, -1).reshape(b * k, -1)
        s0 = z2[:, None].expand(b, k, -1).reshape(b * k, -1)
        zero_a = z2.new_zeros(b * k, MACRO_DIM)
        h_exp = h[:, None].expand(b, k).reshape(b * k)

        if self.mode == 'tf_parallel':
            qs = []
            for t in range(h_steps):
                pos = torch.full((b * k,), t, dtype=torch.long, device=dev)
                rem = (h_exp - t).clamp(min=1, max=self.max_h) - 1
                qs.append(self._step_query(s0, zero_a, pos, rem, cand_q, is_first=(t == 0)))
            raw = self._decode_queries(torch.stack(qs, dim=1), memory)
            cand = self._squash(raw).view(b, k, h_steps, MACRO_DIM)
            head_seconds = time.perf_counter() - t_head0
            if pad_to_max_h and h_steps < self.max_h:
                mid = 0.5 * (self.lo + self.hi)
                pad = mid.view(1, 1, 1, MACRO_DIM).expand(b, k, self.max_h - h_steps, MACRO_DIM)
                cand = torch.cat([cand, pad.to(dtype)], dim=2)
            if extra_actions is not None:
                cand = torch.cat([cand[:, :, :extra_actions.shape[2]], extra_actions.to(dev, dtype)], dim=1)
            return cand, None, dict(head_seconds=head_seconds, wm_rollout_seconds=0.0,
                                    step_actions=list(cand.unbind(2)))

        cm = self._resolve_cost_model(cost_model) if self.mode == 'tf_ar_wm' else None
        jepa = cm.model if cm is not None else None
        extra_k = 0 if extra_actions is None else int(extra_actions.shape[1])
        total_k = k + extra_k
        emb = (z2[:, None, None].expand(b, total_k, 1, -1).reshape(b * total_k, 1, -1).clone()
               if jepa is not None else None)
        act_wm = None
        s_cur = s0
        a_prev_norm = zero_a
        qs, steps = [], []
        head_seconds = 0.0
        wm_seconds = 0.0

        for t in range(h_steps):
            t0 = time.perf_counter()
            pos = torch.full((b * k,), t, dtype=torch.long, device=dev)
            rem = (h_exp - t).clamp(min=1, max=self.max_h) - 1
            q_t = self._step_query(s_cur, a_prev_norm, pos, rem, cand_q, is_first=(t == 0))
            qs.append(q_t)
            raw_t = self._decode_queries(torch.stack(qs, dim=1), memory)[:, -1]
            m_t = self._squash(raw_t).view(b, k, MACRO_DIM)
            steps.append(m_t)
            a_prev_norm = self._norm_macro(m_t.view(b * k, MACRO_DIM))
            head_seconds += time.perf_counter() - t0

            if jepa is not None:
                tw0 = time.perf_counter()
                a_step = (torch.cat([m_t, extra_actions[:, :, t].to(dev, dtype)], dim=1)
                          if extra_k else m_t)
                emb, act_wm, next_s = advance_jepa(jepa, emb, act_wm, a_step, h_steps)
                s_cur = next_s[:, :k].reshape(b * k, -1)
                wm_seconds += time.perf_counter() - tw0

        cand = torch.stack(steps, dim=2)
        pred_emb = emb.view(b, total_k, h_steps + 1, -1) if emb is not None and uniform_h else None
        if extra_k:
            cand = torch.cat([cand, extra_actions[:, :, :h_steps].to(dev, dtype)], dim=1)
        if pad_to_max_h and h_steps < self.max_h:
            mid = 0.5 * (self.lo + self.hi)
            pad = mid.view(1, 1, 1, MACRO_DIM).expand(
                b, cand.shape[1], self.max_h - h_steps, MACRO_DIM)
            cand = torch.cat([cand, pad.to(dtype)], dim=2)
        return cand, pred_emb, dict(head_seconds=head_seconds, wm_rollout_seconds=wm_seconds,
                                    step_actions=steps)

    def forward(self, z2, g2, h, cost_model=None):
        cand, _, _ = self.generate(z2, g2, h, cost_model=cost_model, pad_to_max_h=True)
        return cand


class TransformerActionDecoder(_ConditionalTransformerCore):
    """Conditional Transformer decoder mapping (z1, z2, g2, m) -> (B, heads, 2, block_dim)."""

    def __init__(self, mode='tf_ar_wm', heads=16, d_model=DEFAULT_D_MODEL, layers=DEFAULT_LAYERS,
                 nhead=DEFAULT_ATTN_HEADS, ffn_dim=None, mlp_head_dim=DEFAULT_MLP_HEAD_DIM,
                 embed_dim=192, macro_dim=MACRO_DIM, block_dim=10, mlp_width=1024, mlp_blocks=4):
        if ffn_dim is None:
            ffn_dim = DEFAULT_FFN_DIM if layers == 1 else match_lower_ffn_dim(
                heads=heads, d_model=d_model, layers=layers, nhead=nhead,
                mlp_head_dim=mlp_head_dim, embed_dim=embed_dim, macro_dim=macro_dim,
                block_dim=block_dim, mlp_width=mlp_width, mlp_blocks=mlp_blocks)
        super().__init__(mode=mode, heads=heads, max_h=LOWER_H, action_dim=block_dim,
                         num_mem_tokens=6, embed_dim=embed_dim, d_model=d_model,
                         layers=layers, nhead=nhead, ffn_dim=ffn_dim, mlp_head_dim=mlp_head_dim)
        self.block_dim = int(block_dim)
        self.macro_dim = int(macro_dim)
        self.register_buffer('prior_mean', torch.zeros(macro_dim))
        self.register_buffer('prior_std', torch.ones(macro_dim))

        self.ln_z1 = nn.LayerNorm(embed_dim)
        self.ln_z2 = nn.LayerNorm(embed_dim)
        self.ln_g2 = nn.LayerNorm(embed_dim)
        self.ln_d = nn.LayerNorm(embed_dim)
        self.proj_z1 = nn.Linear(embed_dim, d_model)
        self.proj_z2 = nn.Linear(embed_dim, d_model)
        self.proj_g2 = nn.Linear(embed_dim, d_model)
        self.proj_d = nn.Linear(embed_dim, d_model)
        self.proj_m = nn.Linear(macro_dim, d_model)

    def _memory(self, z1, z2, g2, m):
        b = z1.shape[0]
        k = self.heads
        m_norm = (m - self.prior_mean) / self.prior_std
        t0 = self.proj_z1(self.ln_z1(z1))[:, None].expand(b, k, -1)
        t1 = self.proj_z2(self.ln_z2(z2))[:, None].expand(b, k, -1)
        t2 = self.proj_g2(self.ln_g2(g2))[:, None].expand(b, k, -1)
        t3 = self.proj_d(self.ln_d(g2 - z2))[:, None].expand(b, k, -1)
        t4 = self.proj_m(m_norm)[:, None].expand(b, k, -1)
        t5 = self.cand_emb.weight[None].expand(b, k, -1)
        mem = torch.stack([t0, t1, t2, t3, t4, t5], dim=2).reshape(b * k, 6, self.d_model)
        return mem + self.mem_type_emb[None]

    def generate(self, z1, z2, g2, m, *, cost_model=None):
        """Generate 2-block candidate action sequences.

        Returns (cand (B, heads, 2, block_dim), pred_emb (B, heads, 3, D) or None, stats).
        """
        b, dev = z1.shape[0], z1.device
        k = self.heads
        t_head0 = time.perf_counter()
        memory = self._memory(z1, z2, g2, m)
        cand_q = self.cand_query_proj(self.cand_emb.weight)[None].expand(b, k, -1).reshape(b * k, -1)
        s0 = z1[:, None].expand(b, k, -1).reshape(b * k, -1)
        zero_a = z1.new_zeros(b * k, self.block_dim)

        if self.mode == 'tf_parallel':
            qs = []
            for t in range(LOWER_H):
                pos = torch.full((b * k,), t, dtype=torch.long, device=dev)
                rem = torch.full((b * k,), LOWER_H - t - 1, dtype=torch.long, device=dev)
                qs.append(self._step_query(s0, zero_a, pos, rem, cand_q, is_first=(t == 0)))
            out = self._decode_queries(torch.stack(qs, dim=1), memory).view(b, k, LOWER_H, self.block_dim)
            return out, None, dict(head_seconds=time.perf_counter() - t_head0, wm_rollout_seconds=0.0,
                                   step_actions=list(out.unbind(2)))

        cm = self._resolve_cost_model(cost_model) if self.mode == 'tf_ar_wm' else None
        jepa = cm.model if cm is not None else None
        emb = (z1[:, None, None].expand(b, k, 1, -1).reshape(b * k, 1, -1).clone()
               if jepa is not None else None)
        act_wm = None
        s_cur = s0
        a_prev = zero_a
        qs, steps = [], []
        head_seconds = 0.0
        wm_seconds = 0.0

        for t in range(LOWER_H):
            t0 = time.perf_counter()
            pos = torch.full((b * k,), t, dtype=torch.long, device=dev)
            rem = torch.full((b * k,), LOWER_H - t - 1, dtype=torch.long, device=dev)
            q_t = self._step_query(s_cur, a_prev, pos, rem, cand_q, is_first=(t == 0))
            qs.append(q_t)
            a_t = self._decode_queries(torch.stack(qs, dim=1), memory)[:, -1].view(b, k, self.block_dim)
            steps.append(a_t)
            a_prev = a_t.view(b * k, self.block_dim)
            head_seconds += time.perf_counter() - t0

            if jepa is not None:
                tw0 = time.perf_counter()
                emb, act_wm, next_s = advance_jepa(jepa, emb, act_wm, a_t, LOWER_H)
                s_cur = next_s.reshape(b * k, -1)
                wm_seconds += time.perf_counter() - tw0

        cand = torch.stack(steps, dim=2)
        pred_emb = emb.view(b, k, LOWER_H + 1, -1) if emb is not None else None
        return cand, pred_emb, dict(head_seconds=head_seconds, wm_rollout_seconds=wm_seconds,
                                    step_actions=steps)

    def forward(self, z1, z2, g2, m, cost_model=None):
        cand, _, _ = self.generate(z1, z2, g2, m, cost_model=cost_model)
        return cand


def match_upper_ffn_dim(*, heads=32, d_model=DEFAULT_D_MODEL, layers=DEFAULT_LAYERS,
                        nhead=DEFAULT_ATTN_HEADS, mlp_head_dim=DEFAULT_MLP_HEAD_DIM,
                        embed_dim=192, max_h=MAX_H, mlp_width=1024, mlp_blocks=4,
                        candidates=FFN_CANDIDATES):
    target = mlp_upper_param_count(heads=heads, width=mlp_width, blocks=mlp_blocks,
                                   embed_dim=embed_dim, max_h=max_h)
    ffn0 = candidates[0]
    p0 = trainable_parameters(TransformerUpperProposal(
        mode='tf_parallel', heads=heads, d_model=d_model, layers=layers,
        nhead=nhead, ffn_dim=ffn0, mlp_head_dim=mlp_head_dim, embed_dim=embed_dim, max_h=max_h))
    slope = layers * (2 * d_model + 1)
    return min(candidates, key=lambda ffn: (abs(p0 + (ffn - ffn0) * slope - target), ffn))


def match_lower_ffn_dim(*, heads=16, d_model=DEFAULT_D_MODEL, layers=DEFAULT_LAYERS,
                        nhead=DEFAULT_ATTN_HEADS, mlp_head_dim=DEFAULT_MLP_HEAD_DIM,
                        embed_dim=192, macro_dim=MACRO_DIM, block_dim=10, mlp_width=1024,
                        mlp_blocks=4, candidates=FFN_CANDIDATES):
    target = mlp_lower_param_count(heads=heads, width=mlp_width, blocks=mlp_blocks,
                                   embed_dim=embed_dim, macro_dim=macro_dim, block_dim=block_dim)
    ffn0 = candidates[0]
    p0 = trainable_parameters(TransformerActionDecoder(
        mode='tf_parallel', heads=heads, d_model=d_model, layers=layers,
        nhead=nhead, ffn_dim=ffn0, mlp_head_dim=mlp_head_dim,
        embed_dim=embed_dim, macro_dim=macro_dim, block_dim=block_dim))
    slope = layers * (2 * d_model + 1)
    return min(candidates, key=lambda ffn: (abs(p0 + (ffn - ffn0) * slope - target), ffn))
