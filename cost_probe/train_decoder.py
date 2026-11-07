"""Train the level-1 ActionDecoder offline on the cached level-1 embeddings."""

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

EVAL_COST_CHUNK = 1024


def lr_at(step, total, lr, warmup=1000):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * t))


def evaluate(dec, parts, data, prior_macros, m_mode, batch=4096, seed=0):
    gen = torch.Generator(device=data.device).manual_seed(seed)
    z1, z1_seq, blocks, _ = data.sample_batch(batch, gen)
    with torch.no_grad():
        m = prior_macros if m_mode == 'prior' else parts.data_macro(z1_seq, blocks)
        z2 = parts.encode_level2(z1)
        g2 = parts.subgoal(z2, m)
        if getattr(dec, 'architecture', 'mlp') == 'mlp':
            out = dec(z1, z2, g2, m)
            costs = torch.cat([parts.level1_cost(z1[i:i + EVAL_COST_CHUNK], g2[i:i + EVAL_COST_CHUNK],
                                                 m[i:i + EVAL_COST_CHUNK], out[i:i + EVAL_COST_CHUNK])
                               for i in range(0, len(z1), EVAL_COST_CHUNK)])
            best = costs.min(1)
            usage = torch.bincount(best.indices, minlength=dec.heads).tolist()
            return dict(mean_best_cost=float(best.values.mean()),
                        bc_sse=float(D.bc_loss(out, blocks)),
                        head_usage=usage)
        outs, cost_chunks = [], []
        for i in range(0, len(z1), EVAL_COST_CHUNK):
            sl = slice(i, i + EVAL_COST_CHUNK)
            out_c, pred_c, _ = dec.generate(z1[sl], z2[sl], g2[sl], m[sl],
                                            cost_model=parts.l1_cost_model)
            outs.append(out_c)
            cost_chunks.append(TH.level1_cost_from_rollout(
                parts.l1_cost_model, z1[sl], g2[sl], m[sl], out_c, pred_c,
                action_cost_weight=parts.action_cost_weight,
                cost_last_n=parts.cost_last_n,
                intermediate_cost_weight=parts.intermediate_cost_weight))
        out = torch.cat(outs)
        costs = torch.cat(cost_chunks)
        best = costs.min(1)
        usage = torch.bincount(best.indices, minlength=dec.heads).tolist()
        return dict(mean_best_cost=float(best.values.mean()),
                    bc_sse=float(D.bc_loss(out, blocks)),
                    first_block_head_std=float(out[:, :, 0].std(dim=1).mean()),
                    head_usage=usage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--lambda-bc', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--model-seed', type=int, default=42, help='released H-JEPA checkpoint seed')
    p.add_argument('--env', default='pusht', choices=('pusht', 'fourroom', 'cube', 'ant'))
    p.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
    p.add_argument('--data-on-cpu', action='store_true',
                   help='keep cache embeddings in pinned host memory and copy each batch to the GPU')
    p.add_argument('--architecture', default='mlp', choices=TH.ARCHS,
                   help='decoder head architecture (default: mlp)')
    p.add_argument('--d-model', type=int, default=TH.DEFAULT_D_MODEL)
    p.add_argument('--layers', type=int, default=TH.DEFAULT_LAYERS)
    p.add_argument('--attention-heads', type=int, default=TH.DEFAULT_ATTN_HEADS)
    p.add_argument('--ffn-dim', type=int, default=None,
                   help='Transformer FFN dim (default: 1536 for 1-layer decoder)')
    p.add_argument('--mlp-head-dim', type=int, default=TH.DEFAULT_MLP_HEAD_DIM,
                   help='post-decoder GELU MLP hidden dim')
    p.add_argument('--tf32', action='store_true', help='TF32 matmuls during training')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    device = 'cuda'
    parts = D.PlannerParts(device, args.model_seed, args.env, args.family)
    cache = D.cache_path('train', args.model_seed, args.env, args.family)
    embed_dev = 'cpu' if args.data_on_cpu else None
    train = D.CacheData(cache, device, holdout=False, scaler=parts.action_scaler,
                        action_block=parts.action_block, embed_device=embed_dev)
    held = D.CacheData(cache, device, holdout=True, scaler=parts.action_scaler,
                       action_block=parts.action_block, embed_device=embed_dev)
    assert str(train.attrs['checkpoint_sha256']) == parts.checkpoint_sha256, 'cache built with another checkpoint'
    save_args = dict(vars(args), embed_dim=int(train.embed.shape[1]),
                     block_dim=parts.action_block * parts.action_dim)
    torch.manual_seed(args.seed)
    if args.architecture == 'mlp':
        dec = D.ActionDecoder(heads=args.heads, width=args.width, blocks=args.blocks,
                              embed_dim=save_args['embed_dim'],
                              block_dim=save_args['block_dim']).to(device)
    else:
        dec = TH.TransformerActionDecoder(
            mode=args.architecture, heads=args.heads, d_model=args.d_model,
            layers=args.layers, nhead=args.attention_heads, ffn_dim=args.ffn_dim,
            mlp_head_dim=args.mlp_head_dim,
            embed_dim=save_args['embed_dim'], block_dim=save_args['block_dim'],
            mlp_width=args.width, mlp_blocks=args.blocks).to(device)
        dec.bind_world_model(parts.l1_cost_model)
        save_args['ffn_dim'] = dec.ffn_dim
        save_args['mlp_head_dim'] = dec.mlp_head_dim
    dec.prior_mean.copy_(parts.prior_mean)
    dec.prior_std.copy_(parts.prior_std)
    n_params = TH.trainable_parameters(dec)
    mlp_target = TH.mlp_lower_param_count(heads=args.heads, width=args.width, blocks=args.blocks,
                                          embed_dim=save_args['embed_dim'],
                                          block_dim=save_args['block_dim'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = dict(save_args, git_commit=commit, cache=str(cache),
                  cache_sha256=D.sha256(cache), cache_attrs=train.attrs,
                  model_checkpoint_sha256=parts.checkpoint_sha256,
                  trainable_parameters=n_params, mlp_target_parameters=mlp_target,
                  head_init_seed=args.seed, data_sampling_seed=args.seed,
                  attention_backend='MATH',
                  n_train=len(train.valid), n_heldout=len(held.valid))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    log = (args.out / 'train_log.jsonl').open('w')
    t0 = time.time()
    is_mlp = args.architecture == 'mlp'
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, args.steps, args.lr)
        z1, z1_seq, blocks, _ = train.sample_batch(args.batch, gen)
        with torch.no_grad():
            m_data = parts.data_macro(z1_seq, blocks)
            m, keep = D.augment_macros(parts, z1_seq, blocks, m_data, gen)
            z2 = parts.encode_level2(z1)
            g2 = parts.subgoal(z2, m)
        if is_mlp:
            out = dec(z1, z2, g2, m)
            costs = parts.level1_cost(z1, g2, m, out)
        else:
            out, pred_emb, _ = dec.generate(z1, z2, g2, m, cost_model=parts.l1_cost_model)
            costs = TH.level1_cost_from_rollout(
                parts.l1_cost_model, z1, g2, m, out, pred_emb,
                action_cost_weight=parts.action_cost_weight,
                cost_last_n=parts.cost_last_n,
                intermediate_cost_weight=parts.intermediate_cost_weight)
        l_cost = D.relaxed_wta(costs)
        l_bc = D.bc_loss(out, blocks, keep)
        loss = l_cost + args.lambda_bc * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0 or step + 1 == args.steps:
            dec.eval()
            with torch.no_grad():
                row = dict(step=step + 1, seconds=time.time() - t0,
                           l_cost=float(l_cost), l_bc=float(l_bc),
                           best_cost=float(costs.min(1).values.mean()),
                           data=evaluate(dec, parts, held, None, 'data'),
                           prior=evaluate(dec, parts, held,
                                          torch.minimum(torch.maximum(
                                              parts.prior_mean + parts.prior_std * torch.randn(
                                              4096, 4, device=device,
                                              generator=torch.Generator(device=device).manual_seed(0)),
                                              parts.l2_lo), parts.l2_hi), 'prior'))
            log.write(json.dumps(row) + '\n')
            log.flush()
            dec.train()
        if (step + 1) % 10000 == 0:
            torch.save(dict(state_dict=dec.state_dict(), args=save_args, git_commit=commit,
                            cache_sha256=config['cache_sha256'], cache_attrs={k: str(v) for k, v in train.attrs.items()},
                            model_checkpoint_sha256=parts.checkpoint_sha256,
                            trainable_parameters=n_params, mlp_target_parameters=mlp_target),
                       args.out / f'decoder_step{step + 1}.pt')
    torch.save(dict(state_dict=dec.state_dict(), args=save_args, git_commit=commit,
                    cache_sha256=config['cache_sha256'], cache_attrs={k: str(v) for k, v in train.attrs.items()},
                    model_checkpoint_sha256=parts.checkpoint_sha256,
                    trainable_parameters=n_params, mlp_target_parameters=mlp_target),
               args.out / 'decoder.pt')
    print('TRAIN_DONE', args.out, f'{(time.time()-t0)/args.steps:.3f}s/step',
          f'peak_gib={torch.cuda.max_memory_allocated()/2**30:.2f}', flush=True)


if __name__ == '__main__':
    with sdpa_kernel(SDPBackend.MATH):
        main()
