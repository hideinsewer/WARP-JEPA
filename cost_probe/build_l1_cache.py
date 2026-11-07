"""Cache the released level-1 embeddings of a Push-T h5 dataset.

For every frame, computes the embedding exactly as the planner computes it for a
single current frame: the eval img_transform (ToImage -> float32/255 -> ImageNet
normalize -> Resize(img_size)) followed by the level-1 JEPA encode
(jepa.encode(info, key='pixels', chunk_temporal_inputs=False)['embed_0'][:, -1]).
The level-1 encoder is a ProjectedEncoder and consumes pixels only.

Writes cache/pusht_l1_<split>.h5 plus a verification gate JSON next to it.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'h_jepa')]
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import hdf5plugin  # registers the Blosc filter used by the train h5's pixel chunks
import h5py
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import planning_eval as evaluation
from decoder import cache_path, checkpoint_path

AUX_COLUMNS = ('episode_idx', 'ep_idx', 'step_idx', 'action', 'proprio', 'state',
               'block_ori', 'xy', 'goal_state', 'qpos', 'qvel', 'distractor0_xy',
               'min_distractor_dist')
RENAME = {'episode_idx': 'episode', 'ep_idx': 'episode', 'step_idx': 'step'}

SOURCE_FILES = {
    'pusht': {'train': 'pusht_expert_train.h5', 'val': 'pusht_expert_val.h5'},
    'fourroom': {'train': 'fourroom_tp35_d1.h5', 'val': 'fourroom_tp35_d1_val.h5'},
    'cube': {'train': 'cube_single_expert_train.h5', 'val': 'cube_single_expert_val.h5'},
    'ant': {'train': 'visual_antmaze_medium_explore_stitch_train.h5',
            'val': 'visual_antmaze_medium_stitch_val.h5'},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_name(model_seed, env='pusht', family='hjepa', levels=2):
    return str(checkpoint_path(model_seed, env, family, levels=levels).relative_to(
        ROOT / 'cost_probe/artifacts'))


def load_setup(model_seed=42, env='pusht', family='hjepa', levels=2):
    cfg = evaluation.load_eval_config(f'{env}_l{levels}')
    cfg.cache_dir = str(ROOT / 'cost_probe/artifacts')
    cfg.policy = str(ROOT / 'cost_probe/artifacts' / checkpoint_name(model_seed, env, family, levels=levels))
    model = evaluation.load_model(cfg)
    jepa = model.get_level(1)
    transform = evaluation.img_transform(cfg)
    process = evaluation._build_eval_process_from_policy_normalizer(cfg, model=model)
    return cfg, model, jepa, transform, process


def planner_embed(policy, pixels_u8, proprio):
    """One frame through the evaluation world's path: _prepare_info + _encode_all_levels."""
    info = {'pixels': pixels_u8[None, None]}
    if proprio is not None:
        info['proprio'] = proprio[None, None]
    prepared = policy._prepare_info(info)
    with sdpa_kernel(SDPBackend.MATH):  # the eval wraps planning in this backend
        out = policy.solver._encode_all_levels(prepared, key='pixels')
    return out['embed_1'][0, 0].detach().cpu()


def aligned_proprio(source, start, stop, target_dim):
    """Raw proprio rows -> the planner's proprio layout, mirroring
    policy._align_proprio_to_processor: qpos[2:]+qvel when that merges to
    target_dim, else drop the leading two dims (xy) when dim - 2 matches."""
    raw = source['proprio'][start:stop]
    if raw.shape[-1] == target_dim:
        return raw
    if 'qpos' in source and 'qvel' in source:
        merged = np.concatenate([source['qpos'][start:stop, 2:],
                                 source['qvel'][start:stop]], axis=-1)
        if merged.shape[-1] == target_dim:
            return merged
    if raw.shape[-1] - 2 == target_dim:
        return raw[..., 2:]
    raise ValueError(f'proprio dim {raw.shape[-1]} cannot align to processor dim {target_dim}')


def encode_frames(jepa, transform, pixels_u8, device, batched=False,
                  proprio=None, proprio_key='proprio'):
    """uint8 (N,H,W,C) -> embed (N,D): same transform object and encode call as the
    planner. The transform is batched (verified bit-identical); the encode runs one
    frame at a time, because the planner always encodes a single current frame and
    the ViT's batched fp32 kernels are not batch-size invariant (~2e-3). batched=True
    encodes all N frames in one call (fast, not bit-exact). proprio, when given, is
    the already-normalized proprio stream (the process['proprio'] transform applied,
    exactly as policy._prepare_info produces it)."""
    x = torch.from_numpy(np.transpose(pixels_u8, (0, 3, 1, 2)).copy()).to(device)
    info = {'pixels': x[:, None]}
    if proprio is not None:
        info[proprio_key] = proprio[:, None]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        x = transform(x)
        info['pixels'] = x[:, None]
        if batched:
            embeds = jepa.encode(info, key='pixels',
                                 chunk_temporal_inputs=False)['embed_0'][:, -1]
        else:
            embeds = torch.stack([
                jepa.encode({k: v[i:i + 1] for k, v in info.items()}, key='pixels',
                            chunk_temporal_inputs=False)['embed_0'][0, -1]
                for i in range(len(x))])
    return embeds.detach().float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'train'), required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'cost_probe/artifacts/data')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'cost_probe/artifacts/data/cache')
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--gate', type=int, default=64)
    parser.add_argument('--gate-seed', type=int, default=0)
    parser.add_argument('--model-seed', type=int, default=42, help='released H-JEPA checkpoint seed')
    parser.add_argument('--env', default='pusht', choices=tuple(SOURCE_FILES))
    parser.add_argument('--family', default='hjepa', choices=('hjepa', 'hwm'))
    parser.add_argument('--levels', type=int, default=2, choices=(2, 3, 4),
                        help='hierarchy depth of the world model checkpoint (default: 2)')
    parser.add_argument('--batched-encode', action='store_true',
                        help='encode --batch frames per call (fast, not bit-exact with the planner path)')
    parser.add_argument('--gate-only', action='store_true',
                        help='skip encoding; run the planner-parity gate on an existing cache')
    args = parser.parse_args()

    source_path = args.data_dir / SOURCE_FILES[args.env][args.split]
    assert source_path.exists(), source_path
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / cache_path(args.split, args.model_seed, args.env,
                                       args.family, levels=args.levels).name
    gate_path = out_path.with_name(out_path.stem + '_gate.json')
    if args.gate_only:
        if gate_path.exists():
            parser.error(f'{gate_path} exists; refusing to overwrite')
        if not out_path.exists():
            parser.error(f'{out_path} does not exist')
    elif out_path.exists() or gate_path.exists():
        parser.error(f'{out_path} or its gate exists; refusing to overwrite')
    encode_mode = f'batched_{args.batch}' if args.batched_encode else 'per_frame'

    cfg, model, jepa, transform, process = load_setup(args.model_seed, args.env,
                                                    args.family, levels=args.levels)
    device = next(jepa.parameters()).device
    source_sha = sha256(source_path)
    ckpt_sha = sha256(Path(cfg.policy))
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    source = h5py.File(source_path, 'r')
    n_frames = source['pixels'].shape[0]
    # A fusion-encoder level consumes the planner-normalized proprio stream along
    # with pixels (ant). The gate goes through policy._prepare_info, which applies
    # the same process['proprio'] transform, so both paths see identical inputs.
    uses_proprio = hasattr(jepa.encoder, 'proprio_encoder')
    proprio_key = getattr(jepa.encoder, 'proprio_key', 'proprio')
    proprio_scaler = process['proprio'] if uses_proprio else None
    t0 = time.time()
    embed_dim = None
    if args.gate_only:
        with h5py.File(out_path, 'r') as existing:
            n_frames = existing['embed'].shape[0]
            encode_mode = existing.attrs.get('encode_mode', encode_mode)
            print(f'GATE_ONLY {out_path} frames={n_frames} mode={encode_mode}', flush=True)
    else:
      with h5py.File(out_path, 'w') as out:
        embed_ds = None
        written = 0
        for start in range(0, n_frames, args.chunk):
            pixels = source['pixels'][start:start + args.chunk]
            for inner in range(0, pixels.shape[0], args.batch):
                proprio = None
                if proprio_scaler is not None:
                    raw = aligned_proprio(source, start + inner,
                                          start + inner + args.batch,
                                          proprio_scaler.mean_.shape[0])
                    proprio = torch.from_numpy(
                        proprio_scaler.transform(raw).astype('float32')).to(device)
                emb = encode_frames(jepa, transform, pixels[inner:inner + args.batch],
                                    device, batched=args.batched_encode,
                                    proprio=proprio, proprio_key=proprio_key)
                if embed_ds is None:
                    embed_dim = emb.shape[1]
                    embed_ds = out.create_dataset('embed', shape=(n_frames, embed_dim),
                                                  dtype='float32')
                embed_ds[written:written + emb.shape[0]] = emb.numpy()
                written += emb.shape[0]
                del emb
            done = min(start + args.chunk, n_frames)
            rate = done / (time.time() - t0)
            print(f'CACHE_PROGRESS {args.split} {done}/{n_frames} {rate:.0f} frames/s', flush=True)
        for col in AUX_COLUMNS:
            if col in source:
                out.create_dataset(RENAME.get(col, col), data=source[col][:])
        attrs = dict(checkpoint_sha256=ckpt_sha, source_h5_sha256=source_sha,
                     git_commit=git_commit, split=args.split, env=args.env,
                     family=args.family, levels=args.levels,
                     planner_config=f'{args.env}_l{args.levels}',
                     checkpoint=checkpoint_name(args.model_seed, args.env, args.family, levels=args.levels),
                     model_seed=args.model_seed,
                     uses_proprio=uses_proprio,
                     encode_mode=encode_mode, num_frames=n_frames, embed_dim=embed_dim,
                     preprocessing=('planner path: uint8 HWC -> CHW -> img_transform '
                                    '(ToImage, ToDtype float32 scale=1/255, ImageNet '
                                    'Normalize, Resize(img_size)) -> level-1 JEPA '
                                    'encode(key=pixels, chunk_temporal_inputs=False) '
                                    '-> embed_0[:, -1]; level-1 encoder is a '
                                    'ProjectedEncoder and consumes pixels only'),
                     pixels_dataset='not copied; raw frames remain in the source h5')
        for key, value in attrs.items():
            out.attrs[key] = value
    seconds = time.time() - t0
    print(f'CACHE_DONE {out_path} frames={n_frames} dim={embed_dim} '
          f'seconds={seconds:.1f} rate={n_frames / seconds:.0f} frames/s '
          f'size_bytes={out_path.stat().st_size}', flush=True)

    rng = np.random.default_rng(args.gate_seed)
    indices = np.sort(rng.choice(n_frames, size=min(args.gate, n_frames), replace=False))
    policy = evaluation._build_policy(
        cfg, process, {'pixels': transform, 'goal': evaluation.img_transform(cfg)},
        model=model)
    cache = h5py.File(out_path, 'r')['embed']
    diffs = []
    for index in indices:
        expected = planner_embed(policy, source['pixels'][int(index)],
                                 source['proprio'][int(index)]
                                 if 'proprio' in source else None)
        diffs.append(float((expected - torch.from_numpy(cache[int(index)])).abs().max()))
    gate = dict(split=args.split, model_seed=args.model_seed, env=args.env,
                family=args.family, uses_proprio=uses_proprio,
                encode_mode=encode_mode,
                frames_sampled=len(indices), seed=args.gate_seed,
                tolerance=1e-5, max_abs_diff=max(diffs),
                mean_abs_diff=float(np.mean(diffs)),
                passed=max(diffs) <= 1e-5, indices=indices.tolist())
    gate_path.write_text(json.dumps(gate, indent=2))
    print('GATE', json.dumps({k: gate[k] for k in ('split', 'frames_sampled', 'max_abs_diff',
                                                 'mean_abs_diff', 'passed')}), flush=True)
    source.close()


if __name__ == '__main__':
    main()
