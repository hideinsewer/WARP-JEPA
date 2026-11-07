"""CPU-only unit tests for the decoder helpers (no model load)."""

import unittest

import h5py
import numpy as np
import torch
import tempfile
from pathlib import Path

from decoder import ActionDecoder, CacheData, bc_loss, relaxed_wta, augment_macros, cache_path, checkpoint_path
from types import SimpleNamespace


def write_cache(path, episode, step, dim=4, adim=2):
    n = len(episode)
    with h5py.File(path, 'w') as f:
        f.create_dataset('embed', data=np.random.default_rng(0).normal(size=(n, dim)).astype('f4'))
        f.create_dataset('action', data=np.zeros((n, adim), 'f4'))
        f.create_dataset('episode', data=episode)
        f.create_dataset('step', data=step)


class DecoderTests(unittest.TestCase):
    def test_valid_index_rule_at_episode_ends(self):
        with tempfile.TemporaryDirectory() as d:
            ep = np.array([0]*20 + [1]*15)
            step = np.array(list(range(20)) + list(range(15)))
            p = Path(d)/'c.h5'
            write_cache(p, ep, step)
            c = CacheData(p, 'cpu', holdout_every=None, holdout=False)
            # valid t needs t+10 same episode: ep0 -> t 0..9, ep1 -> t 20..24
            self.assertEqual(c.valid.tolist(), list(range(10)) + list(range(20, 25)))

    def test_block_layout_row_major(self):
        with tempfile.TemporaryDirectory() as d:
            ep = np.zeros(12, dtype=np.int64)
            step = np.arange(12)
            p = Path(d)/'c.h5'
            write_cache(p, ep, step)
            c = CacheData(p, 'cpu', holdout_every=10, holdout=False)
            c.action = torch.from_numpy(np.arange(24).reshape(12, 2).astype('f4'))
            blocks = c.normalized_blocks(torch.tensor([0]))
            self.assertTrue(torch.equal(blocks[0, 0], torch.arange(10, dtype=torch.float32)))
            self.assertTrue(torch.equal(blocks[0, 1], torch.arange(10, 20, dtype=torch.float32)))

    def test_augmentation_within_bounds(self):
        parts = SimpleNamespace(prior_mean=torch.zeros(4), prior_std=torch.ones(4),
                                l2_lo=torch.full((4,), -.5), l2_hi=torch.full((4,), .5))
        g = torch.Generator().manual_seed(0)
        m_data = torch.randn(64, 4)
        m, keep = augment_macros(parts, None, None, m_data, g)
        self.assertEqual(m.shape, (64, 4))
        self.assertTrue(bool((m[~keep] <= .5).all() and (m[~keep] >= -.5).all()))
        self.assertTrue(bool((m[keep] == m_data[keep]).all()))
        self.assertTrue(bool((m[~keep] != m_data[~keep]).any()))

    def test_relaxed_wta_values(self):
        costs = torch.tensor([[1., 2., 3.], [4., 4., 4.]])
        expect = (1-.05)*((1.+4.)/2) + .05*((2.+4.)/2)
        self.assertAlmostEqual(relaxed_wta(costs, eps=.05).item(), expect, places=6)

    def test_bc_loss_min_over_heads(self):
        out = torch.zeros(2, 3, 2, 10)
        tgt = torch.ones(2, 2, 10)
        out[0, 1] = 1.; out[1, 0] = 1.
        self.assertAlmostEqual(bc_loss(out, tgt).item(), 0.)
        self.assertAlmostEqual(bc_loss(out, tgt, torch.tensor([True, False])).item(), 0.)

    def test_seed_paths_keep_seed42_names(self):
        self.assertEqual(cache_path('train').name, 'pusht_l1_train.h5')
        self.assertEqual(cache_path('val', 42).name, 'pusht_l1_val.h5')
        self.assertEqual(cache_path('train', 43).name, 'pusht_l1_train_seed43.h5')
        self.assertEqual(checkpoint_path(44).parent.name, 'seed44')
        self.assertEqual(checkpoint_path().parent.name, 'seed42')

    def test_decoder_output_shape(self):
        dec = ActionDecoder(heads=8, width=64, blocks=2, embed_dim=16)
        y = dec(torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 4))
        self.assertEqual(y.shape, (5, 8, 2, 10))

    def test_decoder_block_dim(self):
        dec = ActionDecoder(heads=8, width=64, blocks=2, embed_dim=16, block_dim=25)
        y = dec(torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 4))
        self.assertEqual(y.shape, (5, 8, 2, 25))

    def test_env_family_paths(self):
        self.assertEqual(cache_path('train', 42, 'fourroom', 'hjepa').name,
                         'fourroom_hjepa_l1_train_seed42.h5')
        self.assertEqual(cache_path('val', 43, 'ant', 'hwm').name,
                         'ant_hwm_l1_val_seed43.h5')
        self.assertEqual(cache_path('train', 42, 'cube', 'hjepa').name,
                         'cube_hjepa_l1_train_seed42.h5')
        self.assertEqual(cache_path('val', 44, 'pusht', 'hwm').name,
                         'pusht_hwm_l1_val_seed44.h5')
        self.assertEqual(cache_path('train', 42).name, 'pusht_l1_train.h5')
        self.assertEqual(cache_path('train', 43).name, 'pusht_l1_train_seed43.h5')
        self.assertEqual(checkpoint_path(42, 'cube', 'hwm').name, 'cube_hwm_l2_object.ckpt')
        self.assertEqual(checkpoint_path(42, 'cube', 'hwm').parent.name, 'seed42')
        self.assertEqual(checkpoint_path(42, 'fourroom', 'hjepa').name,
                         'fourroom_hjepa_l2_object.ckpt')
        self.assertEqual(checkpoint_path(43).name, 'pusht_hjepa_l2_object.ckpt')

    def test_cache_data_action_dim_5_and_8(self):
        for adim in (5, 8):
            with tempfile.TemporaryDirectory() as d:
                ep = np.array([0] * 20 + [1] * 15)
                step = np.array(list(range(20)) + list(range(15)))
                p = Path(d) / 'c.h5'
                write_cache(p, ep, step, adim=adim)
                c = CacheData(p, 'cpu', holdout_every=None, holdout=False,
                              action_block=5)
                self.assertEqual(c.span, 10)
                self.assertEqual(c.action_dim, adim)
                # valid t needs t+span in the same episode: ep0 -> 0..9, ep1 -> 20..24
                self.assertEqual(c.valid.tolist(), list(range(10)) + list(range(20, 25)))
                blocks = c.normalized_blocks(c.valid[:3])
                self.assertEqual(blocks.shape, (3, 2, 5 * adim))
                _, z1_seq, _, _ = c.sample_batch(4)
                self.assertEqual(z1_seq.shape[1:], (3, 4))


class EvaluateChunkTests(unittest.TestCase):
    def test_evaluate_splits_level1_cost_calls(self):
        import train_decoder
        b, k, d = 4096, 16, 8
        g = torch.Generator().manual_seed(0)
        z1, z1_seq = torch.randn(b, d, generator=g), torch.randn(b, 3, d, generator=g)
        blocks, out = torch.randn(b, 2, 10, generator=g), torch.randn(b, k, 2, 10, generator=g)
        weights = torch.randn(k, generator=g)
        calls = []

        def level1_cost(z1_, g2_, m_, out_):
            calls.append(len(z1_))
            return (out_.pow(2).sum((2, 3)) * weights + z1_.sum(1, keepdim=True)
                    + g2_.sum(1, keepdim=True) + m_.sum(1, keepdim=True))

        class Dec:
            heads = k

            def __call__(self, *args):
                return out

        parts = SimpleNamespace(data_macro=lambda s, bl: s[:, 0, :4], encode_level2=lambda x: 2 * x,
                                subgoal=lambda z2, m: z2 + m.sum(1, keepdim=True), level1_cost=level1_cost)
        data = SimpleNamespace(device='cpu', sample_batch=lambda n, gen: (z1[:n], z1_seq[:n], blocks[:n], None))
        chunked = train_decoder.evaluate(Dec(), parts, data, None, 'data')
        self.assertEqual(calls, [train_decoder.EVAL_COST_CHUNK] * (b // train_decoder.EVAL_COST_CHUNK))
        self.assertLess(train_decoder.EVAL_COST_CHUNK * k, 65536)
        calls.clear()
        chunk, train_decoder.EVAL_COST_CHUNK = train_decoder.EVAL_COST_CHUNK, b
        try:
            whole = train_decoder.evaluate(Dec(), parts, data, None, 'data')
        finally:
            train_decoder.EVAL_COST_CHUNK = chunk
        self.assertEqual(calls, [b])
        self.assertEqual(chunked, whole)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA-only data-placement test')
class CpuDataTests(unittest.TestCase):
    """--data-on-cpu paths must produce bit-identical batches to the GPU-resident path."""

    def _synthetic_cache(self, d, adim=2, eps=(0, 1, 2, 3, 4), elen=120, dim=8):
        p = Path(d) / 'c.h5'
        episode = np.repeat(np.asarray(eps), elen)
        step = np.tile(np.arange(elen), len(eps))
        write_cache(p, episode, step, dim=dim, adim=adim)
        return p

    def test_cache_data_cpu_embed_equal_batches(self):
        from decoder import CacheData as CD
        with tempfile.TemporaryDirectory() as d:
            p = self._synthetic_cache(d)
            a = CD(p, 'cuda', holdout_every=None)
            b = CD(p, 'cuda', holdout_every=None, embed_device='cpu')
            self.assertEqual(b.embed.device.type, 'cpu')
            for i in range(4):
                ga = torch.Generator('cuda').manual_seed(100 + i)
                gb = torch.Generator('cuda').manual_seed(100 + i)
                ra, rb = a.sample_batch(64, ga), b.sample_batch(64, gb)
                for x, y in zip(ra, rb):
                    self.assertTrue(torch.equal(x, y))
                self.assertEqual(rb[3].device.type, 'cuda')

    def _l2parts(self):
        g = torch.Generator().manual_seed(7)
        lin = torch.nn.Linear(8, 8).to('cuda')
        with torch.no_grad():
            lin.weight.copy_(torch.randn(8, 8, generator=g))
            lin.bias.copy_(torch.randn(8, generator=g))
        def encode(z):
            return lin(z)
        def macro(z1_seq, blocks):
            return blocks.flatten(1)[:, :4] + z1_seq[:, 2, :4]
        return SimpleNamespace(device='cuda', encode_level2=encode,
                               data_macro=macro,
                               action_mean=torch.zeros(2, device='cuda'),
                               action_std=torch.ones(2, device='cuda'))

    def test_l2data_cpu_z2_and_macro_equal(self):
        import upper_proposal as U
        with tempfile.TemporaryDirectory() as d:
            p = self._synthetic_cache(d)
            parts = self._l2parts()
            a = U.L2Data(p, parts, holdout_every=100, chunk=128)
            b = U.L2Data(p, parts, holdout_every=100, chunk=128, z2_device='cpu')
            self.assertEqual(b.z2.device.type, 'cpu')
            self.assertTrue(torch.equal(a.z2, b.z2.cuda()))
            nan_a, nan_b = torch.isnan(a.macro), torch.isnan(b.macro)
            self.assertTrue(torch.equal(nan_a, nan_b))
            fin = ~nan_a
            self.assertTrue(torch.equal(a.macro[fin], b.macro[fin]))

    def test_l2data_sample_cpu_equal(self):
        import upper_proposal as U
        with tempfile.TemporaryDirectory() as d:
            p = self._synthetic_cache(d)
            parts = self._l2parts()
            a = U.L2Data(p, parts, holdout_every=100)
            b = U.L2Data(p, parts, holdout_every=100, z2_device='cpu')
            for h in (2, 3):
                for orng in (None, (5, 40)):
                    ga = torch.Generator('cuda').manual_seed(55 + h)
                    gb = torch.Generator('cuda').manual_seed(55 + h)
                    ra = a.sample(128, h, ga, 'train', offset_range=orng)
                    rb = b.sample(128, h, gb, 'train', offset_range=orng)
                    for x, y in zip(ra, rb):
                        self.assertTrue(torch.equal(x, y))

    def test_heldout_sets_cpu_equal(self):
        import upper_proposal as U
        from train_upper import heldout_sets
        with tempfile.TemporaryDirectory() as d:
            p = self._synthetic_cache(d, eps=list(range(50)))
            parts = self._l2parts()
            a = U.L2Data(p, parts, holdout_every=10, chunk=128)
            b = U.L2Data(p, parts, holdout_every=10, chunk=128, z2_device='cpu')
            ha = heldout_sets(a, 32, max_h=4, offset_range=(5, 40))
            hb = heldout_sets(b, 32, max_h=4, offset_range=(5, 40))
            for h in ha:
                for x, y in zip(ha[h], hb[h]):
                    self.assertTrue(torch.equal(x, y))


if __name__ == '__main__':
    unittest.main()
