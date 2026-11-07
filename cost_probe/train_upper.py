"""Train the level-2 UpperProposal offline on the cached level-1 embeddings."""

import argparse
import json
import math
import subprocess
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

import decoder as D
import transformer_heads as TH
import upper_proposal as U

HELDOUT_PER_H = 512


def lr_at(step, total, lr, warmup=1000):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * t))


def heldout_sets(data, per_h, max_h=U.MAX_H, offset_range=None, seed=0):
    """Fixed problems per horizon from the held-out episodes -> h: (z2, g2, expert, mask);
    mask marks the rows whose expert plan is a valid target (all rows in the Push-T mode)."""
    gen = torch.Generator(device=data.device).manual_seed(seed)
    if offset_range is not None:
        return {h: data.sample(per_h, h, gen, 'heldout', offset_range=offset_range)[:4]
                for h in range(U.MIN_H, max_h + 1)}
    return {h: data.sample(per_h, h, gen, 'heldout', match_prob=1.0)[:4]
            for h in range(U.MIN_H, max_h + 1)}


@torch.no_grad()
def evaluate(net, parts, sets, cost_kwargs):
    rows = {}
    is_mlp = getattr(net, 'architecture', 'mlp') == 'mlp'
    for h, (z2, g2, expert, mask) in sets.items():
        horizon = torch.full((len(z2),), h, dtype=torch.long, device=z2.device)
        if is_mlp:
            out = net(z2, g2, horizon)[:, :, :h]
            costs = U.level2_cost(parts.l2_cost_model, z2, g2, out, **cost_kwargs)
        else:
            out, pred_emb, _ = net.generate(z2, g2, horizon, cost_model=parts.l2_cost_model)
            costs = TH.level2_cost_from_rollout(parts.l2_cost_model, z2, g2, out, pred_emb,
                                                **cost_kwargs)
        best = costs.min(1)
        dist = (g2 - z2).pow(2).sum(-1).clamp_min(1.0)
        first_std = float(out[:, :, 0].std(dim=1).mean())
        rows[h] = dict(mean_best_cost=float(best.values.mean()),
                       median_best_cost=float(best.values.median()),
                       mean_best_relative=float((best.values / dist).mean()),
                       bc=float(U.macro_bc(out, expert, parts.prior_std, mask)),
                       first_macro_head_std=first_std,
                       head_usage=torch.bincount(best.indices, minlength=net.heads).tolist())
    return rows


@torch.no_grad()
def reference_costs(parts, sets, cost_kwargs):
    """Held-out medians of the expert plan's cost and of the start-to-goal distance."""
    rows = {}
    for h, (z2, g2, expert, mask) in sets.items():
        cost = U.level2_cost(parts.l2_cost_model, z2, g2, expert[:, None], **cost_kwargs)[:, 0]
        rows[h] = dict(expert_median_cost=float(cost[mask].median()) if mask.any() else None,
                       median_distance=float((g2 - z2).pow(2).sum(-1).median()))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--lambda-bc', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--tf32', action='store_true', help='TF32 matmuls during training')
    p.add_argument('--model-seed', type=int, default=42, help='released H-JEPA checkpoint seed')
    p.add_argument('--env', default='pusht', choices=('pusht', 'fourroom', 'cube', 'ant'))
    p.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
    p.add_argument('--max-h', type=int,
                   help='largest level-2 horizon trained (default: the env config horizon)')
    p.add_argument('--offset-range', type=int, nargs=2, metavar=('LO', 'HI'),
                   help='uniform goal-offset range; default keeps the Push-T match/jitter rule')
    p.add_argument('--data-on-cpu', action='store_true',
                   help='keep cache embeddings and level-2 encodings in pinned host memory')
    p.add_argument('--architecture', default='mlp', choices=TH.ARCHS,
                   help='proposal head architecture (default: mlp)')
    p.add_argument('--d-model', type=int, default=TH.DEFAULT_D_MODEL)
    p.add_argument('--layers', type=int, default=TH.DEFAULT_LAYERS)
    p.add_argument('--attention-heads', type=int, default=TH.DEFAULT_ATTN_HEADS)
    p.add_argument('--ffn-dim', type=int, default=None,
                   help='Transformer FFN dim (default: 1536 for 1-layer decoder)')
    p.add_argument('--mlp-head-dim', type=int, default=TH.DEFAULT_MLP_HEAD_DIM,
                   help='post-decoder GELU MLP hidden dim')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    device = 'cuda'
    parts = D.PlannerParts(device, args.model_seed, args.env, args.family)
    if args.max_h is None:
        args.max_h = int(parts.solver._level_plan_config(2).horizon)
    offset_range = tuple(args.offset_range) if args.offset_range else None
    cost_kwargs = U.level2_cost_kwargs(parts)
    cache = D.cache_path('train', args.model_seed, args.env, args.family)
    t_load = time.time()
    data = U.L2Data(cache, parts, holdout_every=100, action_block=parts.action_block,
                    z2_device='cpu' if args.data_on_cpu else None)
    load_seconds = time.time() - t_load
    assert str(data.attrs['checkpoint_sha256']) == parts.checkpoint_sha256, 'cache built with another checkpoint'
    held = heldout_sets(data, HELDOUT_PER_H, max_h=args.max_h, offset_range=offset_range)
    save_args = dict(vars(args), embed_dim=int(data.z2.shape[1]),
                     offset_range=offset_range)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        net = U.UpperProposal(heads=args.heads, width=args.width, blocks=args.blocks,
                              embed_dim=save_args['embed_dim'], max_h=args.max_h).to(device)
    else:
        net = TH.TransformerUpperProposal(
            mode=args.architecture, heads=args.heads, d_model=args.d_model,
            layers=args.layers, nhead=args.attention_heads, ffn_dim=args.ffn_dim,
            mlp_head_dim=args.mlp_head_dim,
            embed_dim=save_args['embed_dim'], max_h=args.max_h,
            mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        net.prior_mean.copy_(parts.prior_mean)
        net.prior_std.copy_(parts.prior_std)
        net.bind_world_model(parts.l2_cost_model)
        save_args['ffn_dim'] = net.ffn_dim
        save_args['mlp_head_dim'] = net.mlp_head_dim
    net.lo.copy_(parts.l2_lo)
    net.hi.copy_(parts.l2_hi)
    n_params = TH.trainable_parameters(net)
    mlp_target = TH.mlp_upper_param_count(heads=args.heads, width=args.width, blocks=args.blocks,
                                          embed_dim=save_args['embed_dim'], max_h=args.max_h)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cache_sha = D.sha256(cache)
    config = dict(save_args, git_commit=commit, cache=D.relpath(cache), cache_sha256=cache_sha,
                  cache_attrs=data.attrs, load_seconds=load_seconds,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=n_params, mlp_target_parameters=mlp_target,
                  head_init_seed=args.seed, data_sampling_seed=args.seed,
                  n_train_frames=len(data.splits['train'][0]),
                  n_heldout_frames=len(data.splits['heldout'][0]),
                  match_prob=U.MATCH_PROB, jitter=U.JITTER, cost_kwargs=cost_kwargs,
                  box_lo=parts.l2_lo.tolist(), box_hi=parts.l2_hi.tolist(),
                  prior_mean=parts.prior_mean.tolist(), prior_std=parts.prior_std.tolist(),
                  heldout_per_h=HELDOUT_PER_H, attention_backend='MATH',
                  heldout_reference=reference_costs(parts, held, cost_kwargs))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    meta = dict(args=save_args, git_commit=commit, cache_sha256=cache_sha,
                cache_attrs={k: str(v) for k, v in data.attrs.items()}, cost_kwargs=cost_kwargs,
                model_checkpoint_sha256=parts.checkpoint_sha256,
                trainable_parameters=n_params, mlp_target_parameters=mlp_target)
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, args.steps, args.lr)
        h = U.MIN_H + step % (args.max_h - U.MIN_H + 1)
        z2, g2, expert, mask, _, _ = data.sample(args.batch, h, gen, 'train',
                                                offset_range=offset_range)
        horizon = torch.full((args.batch,), h, dtype=torch.long, device=device)
        if is_mlp:
            out = net(z2, g2, horizon)[:, :, :h]
            costs = U.level2_cost(parts.l2_cost_model, z2, g2, out, **cost_kwargs)
        else:
            out, pred_emb, _ = net.generate(z2, g2, horizon, cost_model=parts.l2_cost_model)
            costs = TH.level2_cost_from_rollout(parts.l2_cost_model, z2, g2, out, pred_emb,
                                                **cost_kwargs)
        dist = (g2 - z2).pow(2).sum(-1).clamp_min(1.0)
        l_cost = D.relaxed_wta(costs / dist[:, None])
        l_bc = U.macro_bc(out, expert, parts.prior_std, mask)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == 0 or step + 1 == args.steps:
            net.eval()
            row = dict(step=step + 1, seconds=time.time() - t0, h=h, l_cost=float(l_cost),
                       l_bc=float(l_bc), heldout=evaluate(net, parts, held, cost_kwargs))
            log.write(json.dumps(row) + '\n')
            log.flush()
            net.train()
        if (step + 1) % 10000 == 0:
            torch.save(dict(state_dict=net.state_dict(), **meta), args.out / f'proposal_step{step + 1}.pt')
    torch.save(dict(state_dict=net.state_dict(), **meta), args.out / 'proposal.pt')
    print('TRAIN_DONE', args.out, f'{(time.time() - t0) / args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}', flush=True)


if __name__ == '__main__':
    with sdpa_kernel(SDPBackend.MATH):
        main()
