"""CPU-only unit tests for cost_probe/transformer_heads.py and its solver/loader integration."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'cost_probe')]

from decoder import ActionDecoder, level1_cost, load_decoder
from profile_cost import decoder_solve, upper_proposal_solve
import transformer_heads as TH
from upper_proposal import MAX_H, MIN_H, UpperProposal, level2_cost, load_upper_proposal


class _FakeJEPA(nn.Module):
    """Lightweight CPU stand-in for JEPA with Conv1d action_embed, predict, rollout, criterion."""

    def __init__(self, embed_dim=16, action_dim=10, level=1):
        super().__init__()
        self.level = level
        self.conv = nn.Conv1d(action_dim, 24, kernel_size=1)
        self.act_mlp = nn.Sequential(nn.SiLU(), nn.Linear(24, embed_dim))
        self.pred_s = nn.Linear(embed_dim, embed_dim)
        self.pred_a = nn.Linear(embed_dim, embed_dim)
        self.eval()
        self.requires_grad_(False)

    def action_embed(self, act):
        x = self.conv(act.transpose(1, 2)).transpose(1, 2)
        return self.act_mlp(x)

    def predict(self, emb, act_emb):
        return torch.tanh(self.pred_s(emb) + self.pred_a(act_emb))

    def rollout(self, info, action_sequence, history_size=1):
        h_init = info['pixels'].size(2)
        b, s, t_len = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [h_init, t_len - h_init], dim=2)
        emb = info['embed_0'][:, 0].unsqueeze(1).expand(b, s, -1, -1)
        emb = emb.reshape(b * s, h_init, -1).clone()
        act = act_0.reshape(b * s, h_init, -1)
        act_future = act_future.reshape(b * s, t_len - h_init, -1)
        for t in range(t_len - h_init):
            act_emb = self.action_embed(act)
            pred_emb = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t:t + 1, :]], dim=1)
        act_emb = self.action_embed(act)
        pred_emb = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)
        out = dict(info)
        out['predicted_embed_0'] = emb.view(b, s, t_len + 1, -1)
        return out

    def criterion(self, info_dict):
        pred_emb = info_dict['predicted_embed_0'][:, :, 1:, :]
        cost_last_n = max(1, int(info_dict.get('cost_last_n', 1)))
        cost_window = min(cost_last_n, pred_emb.shape[-2])
        pred_emb = pred_emb[:, :, -cost_window:, :]
        goal_emb = info_dict['goal_embed_0'].expand_as(pred_emb).detach()
        cost = F.mse_loss(pred_emb, goal_emb, reduction='none')
        w = float(info_dict.get('intermediate_cost_weight', 1.0))
        if w != 1.0 and cost.shape[2] > 1:
            weights = cost.new_ones(cost.shape[2])
            weights[:-1] = w
            cost = cost * weights.view(1, 1, -1, 1)
        return cost.sum(dim=tuple(range(2, cost.ndim)))


class _FakeUpperJEPA(nn.Module):
    def __init__(self, embed_dim=16, macro_dim=4):
        super().__init__()
        self.temporal_window_size = 1
        self.temporal_stride = 2
        self.state_enc = nn.Linear(embed_dim, embed_dim)
        self.act_enc = nn.Linear(2 * embed_dim, macro_dim)
        self.eval()
        self.requires_grad_(False)

    def encode(self, info, key='embed_1', chunk_temporal_inputs=True, temporal_stride=None):
        stride = temporal_stride or self.temporal_stride
        x = info[key][:, ::stride]
        return {'embed_0': self.state_enc(x)}


class _FakeLevelCostModel(nn.Module):
    def __init__(self, embed_dim=16, action_dim=10, level=1, macro_dim=4):
        super().__init__()
        self.level = level
        self.model = _FakeJEPA(embed_dim=embed_dim, action_dim=action_dim, level=level)
        self.upper_models = [_FakeUpperJEPA(embed_dim=embed_dim, macro_dim=macro_dim)] if level == 1 else []
        self.eval()
        self.requires_grad_(False)

    def rollout(self, info_dict, action_candidates):
        return self.model.rollout(info_dict, action_candidates)

    def _project_to_goal_space(self, pred_emb, action_candidates, *, use_upper_space, dense, goal_level):
        if not self.upper_models or not use_upper_space:
            return pred_emb, None
        b, s = pred_emb.shape[:2]
        stride = 1 if dense else self.upper_models[0].temporal_stride
        pred_for_cost = self.upper_models[0].state_enc(pred_emb[:, :, ::stride])
        action_for_cost = None
        if action_candidates is not None:
            ae = self.model.action_embed(action_candidates.reshape(b * s, -1, action_candidates.shape[-1]))
            ae = ae.view(b, s, -1, ae.shape[-1])
            pooled = self.upper_models[0].act_enc(ae.reshape(b, s, 1, -1))
            action_for_cost = pooled
        return pred_for_cost, action_for_cost

    def _action_cost(self, pred_action, info_dict):
        target = info_dict.get('goal_action')
        if pred_action is None or target is None:
            return None
        cost_last_n = max(1, int(info_dict.get('cost_last_n', 1)))
        w = min(cost_last_n, pred_action.shape[-2], target.shape[-2])
        pred_action = pred_action[..., -w:, :]
        target = target[..., -w:, :].expand_as(pred_action)
        return F.mse_loss(pred_action, target.detach(), reduction='none').sum(
            dim=tuple(range(2, pred_action.ndim)))

    def get_cost(self, info_dict, action_candidates):
        info_dict = self.rollout(dict(info_dict), action_candidates)
        return TH.cost_from_rollout(self, info_dict, action_candidates, info_dict['predicted_embed_0'])


class TransformerHeadTests(unittest.TestCase):
    def test_single_layer_decoder_with_mlp_head_and_mode_identical_params(self):
        for env, max_h, block_dim in (('pusht', 8, 10), ('fourroom', 12, 10),
                                      ('cube', 2, 20), ('ant', 9, 40)):
            mlp_up = UpperProposal(heads=32, width=1024, blocks=4, embed_dim=192, max_h=max_h)
            target_up = TH.trainable_parameters(mlp_up)
            self.assertEqual(target_up, TH.mlp_upper_param_count(max_h=max_h))
            up_counts = []
            for mode in TH.TF_MODES:
                net = TH.TransformerUpperProposal(mode=mode, heads=32, embed_dim=192, max_h=max_h)
                self.assertEqual(net.layers, 1)
                self.assertEqual(len(net.decoder.layers), 1)
                self.assertIsInstance(net.out_mlp[1], nn.GELU)
                up_counts.append(TH.trainable_parameters(net))
            self.assertEqual(len(set(up_counts)), 1, env)

            mlp_low = ActionDecoder(heads=16, width=1024, blocks=4, embed_dim=192, block_dim=block_dim)
            target_low = TH.trainable_parameters(mlp_low)
            self.assertEqual(target_low, TH.mlp_lower_param_count(block_dim=block_dim))
            low_counts = []
            for mode in TH.TF_MODES:
                dec = TH.TransformerActionDecoder(mode=mode, heads=16, embed_dim=192, block_dim=block_dim)
                self.assertEqual(dec.layers, 1)
                self.assertEqual(len(dec.decoder.layers), 1)
                self.assertIsInstance(dec.out_mlp[1], nn.GELU)
                low_counts.append(TH.trainable_parameters(dec))
            self.assertEqual(len(set(low_counts)), 1, env)

            # Explicit layers=4 still matches the 4-block MLP within 5%
            net4 = TH.TransformerUpperProposal(mode='tf_ar_wm', heads=32, layers=4,
                                               mlp_head_dim=0, embed_dim=192, max_h=max_h)
            self.assertLessEqual(abs(TH.trainable_parameters(net4) - target_up) / target_up, 0.05, env)

    def test_world_model_binding_is_not_registered_as_submodule(self):
        cm = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        net = TH.TransformerUpperProposal(mode='tf_ar_wm', heads=4, d_model=32, layers=2,
                                          nhead=4, ffn_dim=64, embed_dim=16, max_h=4)
        before_params = TH.trainable_parameters(net)
        before_keys = set(net.state_dict().keys())
        net.bind_world_model(cm)
        self.assertEqual(TH.trainable_parameters(net), before_params)
        self.assertEqual(set(net.state_dict().keys()), before_keys)
        self.assertNotIn('_cost_model', dict(net.named_modules()))

    def test_upper_shapes_bounds_horizons_and_candidate_diversity(self):
        cm = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        lo = torch.tensor([-1.0, -0.5, 0.0, -2.0])
        hi = torch.tensor([1.0, 0.5, 0.25, 2.0])
        for max_h in (8, 12, 2, 9):
            for mode in TH.TF_MODES:
                net = TH.TransformerUpperProposal(mode=mode, heads=6, d_model=24, layers=2,
                                                  nhead=4, ffn_dim=48, embed_dim=16, max_h=max_h)
                net.lo.copy_(lo)
                net.hi.copy_(hi)
                net.bind_world_model(cm)
                for b in (1, 3):
                    z2, g2 = torch.randn(b, 16), torch.randn(b, 16)
                    for h in (MIN_H, max_h):
                        h_t = torch.full((b,), h, dtype=torch.long)
                        out = net(z2, g2, h_t)
                        self.assertEqual(out.shape, (b, 6, max_h, 4))
                        self.assertTrue(bool((out > lo).all() and (out < hi).all()))
                        cand, pred, _ = net.generate(z2, g2, h_t)
                        self.assertEqual(cand.shape, (b, 6, h, 4))
                        # Candidate IDs make step-0 outputs distinct across heads
                        self.assertGreater(float(cand[:, :, 0].detach().std(dim=1).min()), 1e-5)
                        if mode == 'tf_ar_wm':
                            self.assertEqual(pred.shape, (b, 6, h + 1, 16))
                        else:
                            self.assertIsNone(pred)
                for bad_h in (1, max_h + 1):
                    with self.assertRaises(RuntimeError):
                        net(torch.randn(2, 16), torch.randn(2, 16), torch.full((2,), bad_h, dtype=torch.long))

    def test_lower_shapes_and_candidate_diversity(self):
        for block_dim in (10, 20, 40):
            cm = _FakeLevelCostModel(embed_dim=16, action_dim=block_dim, level=1)
            for mode in TH.TF_MODES:
                dec = TH.TransformerActionDecoder(mode=mode, heads=5, d_model=24, layers=2,
                                                  nhead=4, ffn_dim=48, embed_dim=16, block_dim=block_dim)
                dec.bind_world_model(cm)
                for b in (1, 3):
                    z1, z2, g2 = torch.randn(b, 16), torch.randn(b, 16), torch.randn(b, 16)
                    m = torch.randn(b, 4)
                    cand, pred, _ = dec.generate(z1, z2, g2, m)
                    self.assertEqual(cand.shape, (b, 5, 2, block_dim))
                    self.assertTrue(torch.equal(cand, dec(z1, z2, g2, m)))
                    self.assertGreater(float(cand[:, :, 0].detach().std(dim=1).min()), 1e-5)
                    if mode == 'tf_ar_wm':
                        self.assertEqual(pred.shape, (b, 5, 3, 16))
                    else:
                        self.assertIsNone(pred)

    def test_causal_mask_and_batch_invariance(self):
        torch.manual_seed(0)
        cm = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        net = TH.TransformerUpperProposal(mode='tf_ar_wm', heads=4, d_model=24, layers=2,
                                          nhead=4, ffn_dim=48, embed_dim=16, max_h=6)
        net.eval()
        # 1. Direct causal check on _decode_queries: perturbing token t >= 2 never alters outputs 0..1
        mem = torch.randn(4, 5, 24)
        q1 = torch.randn(4, 5, 24)
        q2 = q1.clone()
        q2[:, 2:] += 10.0
        y1 = net._decode_queries(q1, mem)
        y2 = net._decode_queries(q2, mem)
        self.assertTrue(torch.allclose(y1[:, :2], y2[:, :2], atol=1e-6, rtol=0.0))
        self.assertFalse(torch.allclose(y1[:, 2:], y2[:, 2:]))

        # 2. Batch invariance across B=4 vs B=1 for all 3 modes
        z2, g2 = torch.randn(4, 16), torch.randn(4, 16)
        for mode in TH.TF_MODES:
            net.mode = net.architecture = mode
            batched, _, _ = net.generate(z2, g2, torch.full((4,), 5, dtype=torch.long), cost_model=cm)
            single = torch.cat([
                net.generate(z2[i:i + 1], g2[i:i + 1], torch.full((1,), 5, dtype=torch.long), cost_model=cm)[0]
                for i in range(4)
            ], dim=0)
            self.assertLess(float((batched - single).detach().abs().max()), 1e-5, mode)

    def test_gradients_flow_through_world_model_to_earlier_actions_without_world_model_grads(self):
        torch.manual_seed(0)
        # Upper tf_ar_wm: cost only on the final predicted state (cost_last_n=1) must still
        # send non-zero gradients to step-0 macro-action m_0 via zhat_1 -> q_1 -> m_1 -> zhat_2
        cm2 = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        up = TH.TransformerUpperProposal(mode='tf_ar_wm', heads=4, d_model=24, layers=2,
                                         nhead=4, ffn_dim=48, embed_dim=16, max_h=4)
        z2, g2 = torch.randn(3, 16), torch.randn(3, 16)
        cand, pred_emb, stats = up.generate(z2, g2, torch.full((3,), 3, dtype=torch.long), cost_model=cm2)
        self.assertTrue(pred_emb.requires_grad)
        costs = TH.level2_cost_from_rollout(cm2, z2, g2, cand, pred_emb,
                                            action_cost_weight=0.0, cost_last_n=1,
                                            intermediate_cost_weight=1.0)
        grad_steps = torch.autograd.grad(costs.sum(), stats['step_actions'], retain_graph=True)
        # Step 0 and step 1 actions affect the final state at step 3 through F2 and subsequent queries
        self.assertGreater(float(grad_steps[0].abs().max()), 0.0)
        self.assertGreater(float(grad_steps[1].abs().max()), 0.0)
        self.assertGreater(float(grad_steps[2].abs().max()), 0.0)
        costs.sum().backward()
        self.assertTrue(all(p.grad is None for p in cm2.parameters()))
        self.assertGreater(float(up.prev_action_proj.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(up.step_state_proj.weight.grad.abs().max()), 0.0)

        # Lower tf_ar_wm: terminal cost_last_n=1 sends non-zero gradient to block 0
        cm1 = _FakeLevelCostModel(embed_dim=16, action_dim=10, level=1)
        dec = TH.TransformerActionDecoder(mode='tf_ar_wm', heads=4, d_model=24, layers=2,
                                          nhead=4, ffn_dim=48, embed_dim=16, block_dim=10)
        z1, z2_l, g2_l, m = torch.randn(3, 16), torch.randn(3, 16), torch.randn(3, 16), torch.randn(3, 4)
        cand_l, pred_l, stats_l = dec.generate(z1, z2_l, g2_l, m, cost_model=cm1)
        costs_l = TH.level1_cost_from_rollout(cm1, z1, g2_l, m, cand_l, pred_l,
                                              action_cost_weight=0.2, cost_last_n=1,
                                              intermediate_cost_weight=1.0)
        grad_l = torch.autograd.grad(costs_l.sum(), stats_l['step_actions'], retain_graph=True)
        self.assertGreater(float(grad_l[0].abs().max()), 0.0)
        self.assertGreater(float(grad_l[1].abs().max()), 0.0)
        costs_l.sum().backward()
        self.assertTrue(all(p.grad is None for p in cm1.parameters()))
        self.assertGreater(float(dec.prev_action_proj.weight.grad.abs().max()), 0.0)

    def test_rollout_and_cost_oracle_exact_zero_diff(self):
        torch.manual_seed(1)
        # Level 2 with and without warm-start extra_actions
        cm2 = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        up = TH.TransformerUpperProposal(mode='tf_ar_wm', heads=6, d_model=24, layers=2,
                                         nhead=4, ffn_dim=48, embed_dim=16, max_h=6)
        up.eval()
        z2, g2 = torch.randn(2, 16), torch.randn(2, 16)
        warm = torch.randn(2, 1, 4, 4)
        for extra in (None, warm):
            with torch.no_grad():
                cand, pred_emb, _ = up.generate(z2, g2, torch.full((2,), 4, dtype=torch.long),
                                                cost_model=cm2, extra_actions=extra)
                info = {'pixels': z2.new_zeros(2, cand.shape[1], 1),
                        'embed_0': z2[:, None, None].expand(2, cand.shape[1], 1, -1)}
                oracle_pred = cm2.rollout(info, cand)['predicted_embed_0']
                self.assertTrue(torch.equal(pred_emb, oracle_pred))
                for last_n, iw in ((1, 1.0), (2, 0.7)):
                    kw = dict(action_cost_weight=0.0, cost_last_n=last_n, intermediate_cost_weight=iw)
                    c_reuse = TH.level2_cost_from_rollout(cm2, z2, g2, cand, pred_emb, **kw)
                    c_oracle = level2_cost(cm2, z2, g2, cand, **kw)
                    self.assertTrue(torch.equal(c_reuse, c_oracle))
                    self.assertTrue(torch.equal(c_reuse.argmin(dim=1), c_oracle.argmin(dim=1)))

        # Level 1 with Conv1d action_embed and action_cost_weight in (0.0, 0.2)
        cm1 = _FakeLevelCostModel(embed_dim=16, action_dim=10, level=1)
        dec = TH.TransformerActionDecoder(mode='tf_ar_wm', heads=6, d_model=24, layers=2,
                                          nhead=4, ffn_dim=48, embed_dim=16, block_dim=10)
        dec.eval()
        z1, z2_l, g2_l, m = torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 4)
        with torch.no_grad():
            cand_l, pred_l, _ = dec.generate(z1, z2_l, g2_l, m, cost_model=cm1)
            info_l = {'pixels': z1.new_zeros(2, 6, 1),
                      'embed_0': z1[:, None, None].expand(2, 6, 1, -1)}
            oracle_pred_l = cm1.rollout(info_l, cand_l)['predicted_embed_0']
            self.assertTrue(torch.equal(pred_l, oracle_pred_l))
            for aw in (0.0, 0.2):
                kw = dict(action_cost_weight=aw, cost_last_n=2, intermediate_cost_weight=1.0)
                c_reuse = TH.level1_cost_from_rollout(cm1, z1, g2_l, m, cand_l, pred_l, **kw)
                c_oracle = level1_cost(cm1, z1, g2_l, m, cand_l, **kw)
                self.assertTrue(torch.equal(c_reuse, c_oracle))

    def test_checkpoint_loaders_and_solver_wrappers(self):
        torch.manual_seed(2)
        cm2 = _FakeLevelCostModel(embed_dim=16, action_dim=4, level=2)
        cm1 = _FakeLevelCostModel(embed_dim=16, action_dim=10, level=1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for mode in TH.TF_MODES:
                up = TH.TransformerUpperProposal(mode=mode, heads=4, d_model=24, layers=2,
                                                 nhead=4, ffn_dim=48, embed_dim=16, max_h=4)
                up_path = d / f'up_{mode}.pt'
                torch.save(dict(state_dict=up.state_dict(),
                                args=dict(architecture=mode, heads=4, d_model=24, layers=2,
                                          attention_heads=4, ffn_dim=48, embed_dim=16, max_h=4),
                                git_commit='abc'), up_path)
                up_loaded, _ = load_upper_proposal(up_path, 'cpu')
                self.assertEqual(up_loaded.architecture, mode)

                solver2 = SimpleNamespace(
                    level=2, n_envs=1, device=torch.device('cpu'), solve_records=[],
                    _config=SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0),
                    model=cm2,
                    _expand_info_dict_for_samples=lambda info, num_samples, **kw: {
                        **info,
                        'pixels': info['embed_0'].new_zeros(1, num_samples, 1),
                        'embed_0': info['embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                        'goal_embed_0': info['goal_embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                        'goal_embed_0_level': 2,
                        'goal_embed_0_in_upper_space': False,
                        'dense_upper_goal_projection': False,
                    },
                    _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
                w2 = upper_proposal_solve(lambda *a, **k: 'orig', up_loaded)
                info2 = {'embed_0': torch.randn(1, 1, 16), 'goal_embed_0': torch.randn(1, 1, 16),
                         'action_cost_weight': 0.0}
                out2 = w2(solver2, info2, planning_horizon=3, init_action=torch.randn(1, 3, 4))
                self.assertEqual(out2['actions'].shape, (1, 3, 4))
                self.assertEqual(w2.check['max_abs_diff'], 0.0)
                self.assertEqual(solver2.solve_records[-1]['proposal_architecture'], mode)

                dec = TH.TransformerActionDecoder(mode=mode, heads=4, d_model=24, layers=2,
                                                  nhead=4, ffn_dim=48, embed_dim=16, block_dim=10)
                dec_path = d / f'dec_{mode}.pt'
                torch.save(dict(state_dict=dec.state_dict(),
                                args=dict(architecture=mode, heads=4, d_model=24, layers=2,
                                          attention_heads=4, ffn_dim=48, embed_dim=16, block_dim=10),
                                git_commit='abc'), dec_path)
                dec_loaded, _ = load_decoder(dec_path, 'cpu')
                self.assertEqual(dec_loaded.architecture, mode)

                solver1 = SimpleNamespace(
                    level=1, n_envs=1, device=torch.device('cpu'), solve_records=[],
                    _config=SimpleNamespace(cost_last_n=2, intermediate_cost_weight=1.0),
                    model=cm1,
                    _expand_info_dict_for_samples=lambda info, num_samples, **kw: {
                        **info,
                        'pixels': info['embed_0'].new_zeros(1, num_samples, 1),
                        'embed_0': info['embed_0'][:, -1:][:, None].expand(1, num_samples, 1, -1),
                        'goal_embed_0': info['goal_embed_0'][:, 0:1][:, None].expand(1, num_samples, 1, -1),
                        'goal_action': info['goal_action'][:, 0:1][:, None].expand(1, num_samples, 1, -1),
                        'goal_embed_0_in_upper_space': True,
                    },
                    _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})
                w1 = decoder_solve(lambda *a, **k: 'orig', dec_loaded)
                info1 = {'embed_0': torch.randn(1, 1, 16), 'goal_embed_0': torch.randn(1, 1, 16),
                         'goal_action': torch.randn(1, 1, 4), 'action_cost_weight': 0.2}
                out1 = w1(solver1, info1, planning_horizon=2)
                self.assertEqual(out1['actions'].shape, (1, 2, 10))
                self.assertEqual(w1.check['max_abs_diff'], 0.0)
                self.assertEqual(solver1.solve_records[-1]['decoder_architecture'], mode)


if __name__ == '__main__':
    unittest.main()
