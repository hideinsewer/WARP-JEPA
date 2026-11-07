"""CPU-only unit tests for cost_probe/three_level.py."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

from decoder import cache_path, checkpoint_path
import three_level as TL
import transformer_heads as TH
from upper_proposal import UpperProposal


def write_cache(path, episode, step, dim=8, adim=2):
    n = len(episode)
    with h5py.File(path, 'w') as f:
        f.create_dataset('embed', data=np.random.default_rng(0).normal(size=(n, dim)).astype('f4'))
        f.create_dataset('action', data=np.random.default_rng(1).normal(size=(n, adim)).astype('f4'))
        f.create_dataset('episode', data=episode)
        f.create_dataset('step', data=step)


class ThreeLevelTests(unittest.TestCase):
    def test_three_level_paths_preserve_two_level_names(self):
        self.assertEqual(cache_path('train', 42, 'pusht', 'hjepa', levels=2).name, 'pusht_l1_train.h5')
        self.assertEqual(cache_path('train', 42, 'pusht', 'hjepa', levels=3).name,
                         'pusht_hjepa_l3_l1_train_seed42.h5')
        self.assertEqual(checkpoint_path(42, 'pusht', 'hjepa', levels=3).name,
                         'pusht_hjepa_l3_object.ckpt')
        self.assertEqual(checkpoint_path(42, 'pusht', 'hjepa', levels=3).parent.parent.name,
                         'pusht_hjepa_l3')

    def test_l2_macro_decoder_bounds_bc_and_loader(self):
        torch.manual_seed(0)
        dec = TL.L2MacroDecoder(heads=6, width=32, blocks=2, embed_dim=16, macro_dim=4)
        lo = torch.tensor([-1.5, -0.5, 0.0, -2.0])
        hi = torch.tensor([0.5, 1.5, 1.0, 2.0])
        dec.lo.copy_(lo)
        dec.hi.copy_(hi)
        z2, z3, g3 = torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 16)
        m3 = torch.randn(5, 4)
        out = dec(z2, z3, g3, m3)
        self.assertEqual(out.shape, (5, 6, 2, 4))
        self.assertTrue(bool((out > lo).all() and (out < hi).all()))

        tgt = out[:, 2].detach().clone()
        std = torch.tensor([0.5, 1.0, 2.0, 1.0])
        self.assertAlmostEqual(float(TL.l2_macro_bc_loss(out, tgt, std)), 0.0, places=6)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'l2_decoder.pt'
            torch.save(dict(state_dict=dec.state_dict(),
                            args=dict(heads=6, width=32, blocks=2, embed_dim=16, macro_dim=4),
                            git_commit='abc'), p)
            loaded, _ = TL.load_l2_decoder(p, 'cpu')
            self.assertTrue(torch.equal(loaded(z2, z3, g3, m3), out))

            # Also test TransformerL2MacroDecoder bounds, rollout parity, and loader
            from test_transformer_heads import _FakeLevelCostModel
            cm2 = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
            tdec = TL.TransformerL2MacroDecoder(mode='tf_ar_wm', heads=6, d_model=24, layers=2,
                                                nhead=4, ffn_dim=48, embed_dim=16, macro_dim=4)
            tdec.lo.copy_(lo)
            tdec.hi.copy_(hi)
            tdec.eval()
            cand_t, pred_t, _ = tdec.generate(z2, z3, g3, m3, cost_model=cm2)
            self.assertEqual(cand_t.shape, (5, 6, 2, 4))
            self.assertTrue(bool((cand_t > lo).all() and (cand_t < hi).all()))
            c_reuse = TL.level2_macro_cost(cm2, z2, g3, m3, cand_t, pred_emb=pred_t,
                                           action_cost_weight=0.0, cost_last_n=1, intermediate_cost_weight=1.0)
            c_oracle = TL.level2_macro_cost(cm2, z2, g3, m3, cand_t,
                                            action_cost_weight=0.0, cost_last_n=1, intermediate_cost_weight=1.0)
            self.assertTrue(torch.equal(c_reuse, c_oracle))
            tp = Path(d) / 'l2_tf_decoder.pt'
            torch.save(dict(state_dict=tdec.state_dict(),
                            args=dict(architecture='tf_ar_wm', heads=6, d_model=24, layers=2,
                                      attention_heads=4, ffn_dim=48, embed_dim=16, macro_dim=4),
                            git_commit='abc'), tp)
            tloaded, _ = TL.load_l2_decoder(tp, 'cpu')
            self.assertEqual(tloaded.architecture, 'tf_ar_wm')

    def test_augment_macros_l3_within_bounds(self):
        parts = SimpleNamespace(l3_prior_mean=torch.zeros(4), l3_prior_std=torch.ones(4),
                                l3_lo=torch.full((4,), -0.75), l3_hi=torch.full((4,), 0.75))
        g = torch.Generator().manual_seed(0)
        m3_data = torch.randn(64, 4)
        m3, keep = TL.augment_macros_l3(parts, m3_data, g)
        self.assertEqual(m3.shape, (64, 4))
        self.assertTrue(bool((m3[~keep] <= 0.75).all() and (m3[~keep] >= -0.75).all()))
        self.assertTrue(bool((m3[keep] == m3_data[keep]).all()))

    def test_l3_hierarchy_data_and_sampling(self):
        parts = SimpleNamespace(
            device='cpu',
            encode_level2=lambda z1: 2.0 * z1,
            encode_level3=lambda z2: 3.0 * z2,
            data_macro_l2=lambda z1_seq, blocks: blocks.flatten(1)[:, :4] + z1_seq[:, 2, :4],
            data_macro_l3=lambda z2_seq, m2_pair: m2_pair.sum(1) + z2_seq[:, 2, :4],
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.h5'
            episode = np.array([0] * 140 + [1] * 140 + [100] * 140)
            step = np.tile(np.arange(140), 3)
            write_cache(p, episode, step)
            data = TL.L3HierarchyData(p, parts, holdout_every=100)

        self.assertTrue(torch.equal(torch.isnan(data.m2).all(-1), data.remaining < 10))
        self.assertTrue(torch.equal(torch.isnan(data.m3).all(-1), data.remaining < 20))
        for t in (0, 30, 119, 140, 290):
            expect_m3 = data.m2[t] + data.m2[t + 10] + data.z2[t + 20, :4]
            self.assertTrue(torch.allclose(data.m3[t], expect_m3))

        gen = torch.Generator().manual_seed(0)
        z2, z3, m2_pair, m3_data, idx = data.sample_l2(32, gen, 'train')
        self.assertEqual(m2_pair.shape, (32, 2, 4))
        self.assertTrue(bool(torch.isfinite(m2_pair).all() and torch.isfinite(m3_data).all()))
        self.assertTrue(torch.equal(m2_pair[:, 0], data.m2[idx]))
        self.assertTrue(torch.equal(m2_pair[:, 1], data.m2[idx + 10]))

        for h3 in (2, 4):
            z3_s, g3_s, expert_m3, mask, offset, t_idx = data.sample_l3(
                64, h3, gen, 'train', match_prob=1.0)
            self.assertTrue(bool((offset == 20 * h3 - 5).all()))
            self.assertTrue(bool(mask.all()))
            self.assertEqual(expert_m3.shape, (64, h3, 4))
            self.assertTrue(bool(torch.isfinite(expert_m3).all()))
            for j in range(h3):
                self.assertTrue(torch.equal(expert_m3[:, j], data.m3[t_idx + 20 * j]))
        z2_top, g2_top, expert_m2_top, mask2, _, t2_idx = data.sample_l3(
            32, 2, gen, 'train', match_prob=1.0, top_level=2)
        self.assertEqual(expert_m2_top.shape, (32, 2, 4))
        for j in range(2):
            self.assertTrue(torch.equal(expert_m2_top[:, j], data.m2[t2_idx + 10 * j]))

    def test_l2_and_l3_solver_wrappers_and_bypass(self):
        # 1. L2 decoder wrapper
        dec2 = TL.L2MacroDecoder(heads=4, width=24, blocks=1, embed_dim=8, macro_dim=4)
        cfg2 = SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0)
        solver2 = SimpleNamespace(
            level=2, n_envs=1, _config=cfg2, solve_records=[], device=torch.device('cpu'),
            model=SimpleNamespace(upper_models=[object()],
                                  get_cost=lambda info, cand: cand.pow(2).sum((-1, -2))),
            _expand_info_dict_for_samples=lambda d, **kw: d,
            _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
        info2 = {'embed_0': torch.ones(1, 1, 8), 'goal_embed_0': torch.ones(1, 1, 8),
                 'goal_action': torch.ones(1, 1, 4), 'action_cost_weight': 0.0}
        w2 = TL.l2_decoder_solve(lambda *a, **k: 'orig2', dec2)
        with patch.object(TL, 'encode_upper', lambda m, z, lv: z), \
                patch.object(TL, 'level2_macro_cost', lambda m, z2, g3, m3, cand, **kw: m.get_cost({}, cand)):
            out2 = w2(solver2, dict(info2), planning_horizon=2)
        self.assertEqual(out2['actions'].shape, (1, 2, 4))
        self.assertEqual(w2.check['max_abs_diff'], 0.0)
        self.assertEqual(w2(solver2, dict(info2, goal_embed_0_level=2), planning_horizon=2), 'orig2')

        # 2. L3 proposal wrapper (MLP and Transformer)
        from test_transformer_heads import _FakeLevelCostModel
        cm3 = _FakeLevelCostModel(embed_dim=8, action_dim=4, level=3)
        for net3 in (UpperProposal(heads=4, width=24, blocks=1, embed_dim=8, max_h=4),
                     TH.TransformerUpperProposal(mode='tf_ar_wm', heads=4, d_model=16, layers=1,
                                                 nhead=4, ffn_dim=32, embed_dim=8, max_h=4)):
            net3.eval()
            solver3 = SimpleNamespace(
                level=3, n_envs=1, _config=cfg2, solve_records=[], device=torch.device('cpu'),
                model=cm3,
                _expand_info_dict_for_samples=lambda info, num_samples, **kw: {
                    **info,
                    'pixels': info['embed_0'].new_zeros(1, num_samples, 1),
                    'embed_0': info['embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0': info['goal_embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0_level': 3,
                    'goal_embed_0_in_upper_space': False,
                    'dense_upper_goal_projection': False,
                },
                _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
            w3 = TL.l3_proposal_solve(lambda *a, **k: 'orig3', net3)
            info3 = {'embed_0': torch.randn(1, 1, 8), 'goal_embed_0': torch.randn(1, 1, 8),
                     'action_cost_weight': 0.0}
            out3 = w3(solver3, info3, planning_horizon=3)
            self.assertEqual(out3['actions'].shape, (1, 3, 4))
            self.assertEqual(w3.check['max_abs_diff'], 0.0)

        # 3. 3-level bypass budget wrapper
        seen = []
        wrap_b = TL.bypass_budget_3level(
            lambda s, i, **kw: seen.append((s.level, s.n_steps, s.early_stop_patience)), 10)
        for lv in (1, 2, 3):
            s = SimpleNamespace(level=lv, n_steps=90, early_stop_patience=5)
            wrap_b(s, {'goal_embed_0_level': lv})  # no upper plan
            wrap_b(s, {'goal_embed_0_in_upper_space': True})  # has upper plan
        self.assertEqual(seen, [(1, 10, 0), (1, 90, 5), (2, 10, 0), (2, 90, 5), (3, 90, 5), (3, 90, 5)])


if __name__ == '__main__':
    unittest.main()
