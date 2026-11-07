"""CPU-only unit tests for cost_probe/four_level.py."""

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
import four_level as FL
import transformer_heads as TH
from upper_proposal import UpperProposal


def write_cache(path, episode, step, dim=8, adim=2):
    n = len(episode)
    with h5py.File(path, 'w') as f:
        f.create_dataset('embed', data=np.random.default_rng(0).normal(size=(n, dim)).astype('f4'))
        f.create_dataset('action', data=np.random.default_rng(1).normal(size=(n, adim)).astype('f4'))
        f.create_dataset('episode', data=episode)
        f.create_dataset('step', data=step)


class FourLevelTests(unittest.TestCase):
    def test_four_level_paths(self):
        self.assertEqual(cache_path('train', 42, 'pusht', 'hjepa', levels=4).name,
                         'pusht_hjepa_l4_l1_train_seed42.h5')
        self.assertEqual(checkpoint_path(42, 'pusht', 'hjepa', levels=4).name,
                         'pusht_hjepa_l4_object.ckpt')
        self.assertEqual(checkpoint_path(42, 'pusht', 'hjepa', levels=4).parent.parent.name,
                         'pusht_hjepa_l4')

    def test_augment_macros_l4_within_bounds(self):
        parts = SimpleNamespace(l4_prior_mean=torch.zeros(4), l4_prior_std=torch.ones(4),
                                l4_lo=torch.full((4,), -0.8), l4_hi=torch.full((4,), 0.8))
        g = torch.Generator().manual_seed(0)
        m4_data = torch.randn(64, 4)
        m4, keep = FL.augment_macros_l4(parts, m4_data, g)
        self.assertEqual(m4.shape, (64, 4))
        self.assertTrue(bool((m4[~keep] <= 0.8).all() and (m4[~keep] >= -0.8).all()))
        self.assertTrue(torch.equal(m4[keep], m4_data[keep]))

    def test_l4_hierarchy_data_spans_and_sampling(self):
        # 3 episodes: ep 0 has 130 frames, ep 100 (heldout) has 100 frames, ep 1 has 150 frames
        ep = np.concatenate([np.full(130, 0), np.full(100, 100), np.full(150, 1)]).astype('i8')
        st = np.concatenate([np.arange(130), np.arange(100), np.arange(150)]).astype('i8')
        parts = SimpleNamespace(
            device='cpu',
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
            encode_level2=lambda z1: z1 * 2.0,
            encode_level3=lambda z2: z2 * 3.0,
            encode_level4=lambda z3: z3 * 4.0,
            data_macro_l2=lambda z1_seq, blocks: blocks.sum(dim=-1)[..., :2].repeat(1, 2) + z1_seq[:, -1, :4],
            data_macro_l3=lambda z2_seq, m2_pair: m2_pair[:, 0] + m2_pair[:, 1] + z2_seq[:, -1, :4],
            data_macro_l4=lambda z3_seq, m3_pair: m3_pair[:, 0] + m3_pair[:, 1] + z3_seq[:, -1, :4],
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cache.h5'
            write_cache(p, ep, st, dim=8, adim=2)
            data = FL.L4HierarchyData(p, parts, holdout_every=100, chunk=64, action_block=5)

        self.assertEqual(data.span_l2, 10)
        self.assertEqual(data.span_l3, 20)
        self.assertEqual(data.span_l4, 40)
        self.assertTrue(torch.equal(torch.isnan(data.m2).all(-1), data.remaining < 10))
        self.assertTrue(torch.equal(torch.isnan(data.m3).all(-1), data.remaining < 20))
        self.assertTrue(torch.equal(torch.isnan(data.m4).all(-1), data.remaining < 40))
        for t in (0, 20, 80, 240, 300):
            expect_m4 = data.m3[t] + data.m3[t + 20] + data.z3[t + 40, :4]
            self.assertTrue(torch.allclose(data.m4[t], expect_m4))

        gen = torch.Generator().manual_seed(0)
        z2, z3, m2_pair, m3_data, idx2 = data.sample_l2(32, gen, 'train')
        self.assertEqual(m2_pair.shape, (32, 2, 4))
        self.assertTrue(bool(torch.isfinite(m2_pair).all() and torch.isfinite(m3_data).all()))
        self.assertTrue(torch.equal(m2_pair[:, 0], data.m2[idx2]))
        self.assertTrue(torch.equal(m2_pair[:, 1], data.m2[idx2 + 10]))

        z3_m, z4_m, m3_pair, m4_data, idx3 = data.sample_l3_middle(32, gen, 'train')
        self.assertEqual(m3_pair.shape, (32, 2, 4))
        self.assertTrue(bool(torch.isfinite(m3_pair).all() and torch.isfinite(m4_data).all()))
        self.assertTrue(torch.equal(m3_pair[:, 0], data.m3[idx3]))
        self.assertTrue(torch.equal(m3_pair[:, 1], data.m3[idx3 + 20]))

        z4_s, g4_s, expert_m4, mask, offset, t_idx = data.sample_l4(
            64, 2, gen, 'train', match_prob=1.0)
        self.assertTrue(bool((offset == 75).all()))
        self.assertTrue(bool(mask.all()))
        self.assertEqual(expert_m4.shape, (64, 2, 4))
        self.assertTrue(bool(torch.isfinite(expert_m4).all()))
        for j in range(2):
            self.assertTrue(torch.equal(expert_m4[:, j], data.m4[t_idx + 40 * j]))
        z2_top, g2_top, expert_m2_top, mask2, _, t2_idx = data.sample_l4(
            32, 2, gen, 'train', match_prob=1.0, top_level=2)
        self.assertEqual(expert_m2_top.shape, (32, 2, 4))
        for j in range(2):
            self.assertTrue(torch.equal(expert_m2_top[:, j], data.m2[t2_idx + 10 * j]))

    def test_l3_and_l4_solver_wrappers_and_bypass(self):
        # 1. L3 middle decoder wrapper (MLP and 1-layer TransformerDecoder + MLP)
        from test_transformer_heads import _FakeLevelCostModel, _FakeUpperJEPA
        cm3 = _FakeLevelCostModel(embed_dim=8, action_dim=4, level=3)
        cm3.upper_models = [_FakeUpperJEPA(embed_dim=8, macro_dim=4)]
        cfg = SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0)
        for dec3 in (FL.L3MacroDecoder(heads=4, width=24, blocks=1, embed_dim=8, macro_dim=4),
                     FL.TransformerL3MacroDecoder(mode='tf_ar_wm', heads=4, d_model=16, layers=1,
                                                  nhead=4, ffn_dim=32, mlp_head_dim=32,
                                                  embed_dim=8, macro_dim=4)):
            dec3.eval()
            solver3 = SimpleNamespace(
                level=3, n_envs=1, _config=cfg, solve_records=[], device=torch.device('cpu'),
                model=cm3,
                _expand_info_dict_for_samples=lambda info, num_samples, **kw: {
                    **info,
                    'pixels': info['embed_0'].new_zeros(1, num_samples, 1),
                    'embed_0': info['embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0': info['goal_embed_0'][:, :1][:, None].expand(1, num_samples, 1, -1),
                    'goal_action': info['goal_action'][:, :1][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0_in_upper_space': True,
                },
                _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
            w3 = FL.l3_decoder_solve(lambda *a, **k: 'orig3', dec3)
            info3 = {'embed_0': torch.ones(1, 1, 8), 'goal_embed_0': torch.ones(1, 1, 8),
                     'goal_action': torch.ones(1, 1, 4), 'action_cost_weight': 0.0}
            with patch.object(FL, 'encode_upper', lambda m, z, lv: z):
                out3 = w3(solver3, dict(info3), planning_horizon=2)
            self.assertEqual(out3['actions'].shape, (1, 2, 4))
            self.assertEqual(w3.check['max_abs_diff'], 0.0)
            self.assertEqual(w3(solver3, dict(info3, goal_embed_0_level=3), planning_horizon=2), 'orig3')

        # 2. L4 proposal wrapper (MLP and 1-layer TransformerDecoder + MLP at max_h=2)
        cm4 = _FakeLevelCostModel(embed_dim=8, action_dim=4, level=4)
        for net4 in (UpperProposal(heads=4, width=24, blocks=1, embed_dim=8, max_h=2),
                     TH.TransformerUpperProposal(mode='tf_ar_wm', heads=4, d_model=16, layers=1,
                                                 nhead=4, ffn_dim=32, mlp_head_dim=32,
                                                 embed_dim=8, max_h=2)):
            net4.eval()
            solver4 = SimpleNamespace(
                level=4, n_envs=1, _config=cfg, solve_records=[], device=torch.device('cpu'),
                model=cm4,
                _expand_info_dict_for_samples=lambda info, num_samples, **kw: {
                    **info,
                    'pixels': info['embed_0'].new_zeros(1, num_samples, 1),
                    'embed_0': info['embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0': info['goal_embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                    'goal_embed_0_level': 4,
                    'goal_embed_0_in_upper_space': False,
                    'dense_upper_goal_projection': False,
                },
                _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
            w4 = FL.l4_proposal_solve(lambda *a, **k: 'orig4', net4)
            info4 = {'embed_0': torch.randn(1, 1, 8), 'goal_embed_0': torch.randn(1, 1, 8),
                     'action_cost_weight': 0.0}
            out4 = w4(solver4, info4, planning_horizon=2)
            self.assertEqual(out4['actions'].shape, (1, 2, 4))
            self.assertEqual(w4.check['max_abs_diff'], 0.0)

        # 3. 4-level bypass budget wrapper
        seen = []
        wrap_b = FL.bypass_budget_4level(
            lambda s, i, **kw: seen.append((s.level, s.n_steps, s.early_stop_patience)), 10)
        for lv in (1, 2, 3, 4):
            s = SimpleNamespace(level=lv, n_steps=90, early_stop_patience=5)
            wrap_b(s, {'goal_embed_0_level': lv})  # no upper plan
            wrap_b(s, {'goal_embed_0_in_upper_space': True})  # has upper plan
        self.assertEqual(seen, [
            (1, 10, 0), (1, 90, 5),
            (2, 10, 0), (2, 90, 5),
            (3, 10, 0), (3, 90, 5),
            (4, 90, 5), (4, 90, 5),
        ])


if __name__ == '__main__':
    unittest.main()
