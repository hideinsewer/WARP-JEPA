"""CPU-only unit tests for the level-2 proposal helpers (no model load)."""

import unittest

import h5py
import numpy as np
import torch
import tempfile
from pathlib import Path

from upper_proposal import (JITTER, L2Data, MAX_H, MIN_H, STEPS_PER_MACRO, UpperProposal,
                            level2_cost, macro_bc, run_remaining)
from types import SimpleNamespace


def write_cache(path, episode, step, dim=8, adim=2):
    n = len(episode)
    with h5py.File(path, 'w') as f:
        f.create_dataset('embed', data=np.random.default_rng(0).normal(size=(n, dim)).astype('f4'))
        f.create_dataset('action', data=np.random.default_rng(1).normal(size=(n, adim)).astype('f4'))
        f.create_dataset('episode', data=episode)
        f.create_dataset('step', data=step)


def fake_parts():
    return SimpleNamespace(
        device='cpu',
        encode_level2=lambda z: 2 * z,
        data_macro=lambda z1_seq, blocks: blocks.flatten(1)[:, :4] + z1_seq[:, 2, :4],
        action_mean=torch.zeros(2),
        action_std=torch.ones(2))


def three_episode_cache(path):
    episode = np.array([0] * 120 + [1] * 120 + [100] * 120)
    step = np.tile(np.arange(120), 3)
    write_cache(path, episode, step)
    return episode


class RunRemainingTests(unittest.TestCase):
    def test_runs_split_on_episode_and_step_gaps(self):
        episode = torch.tensor([0] * 10 + [0] * 5 + [1] * 5)
        step = torch.tensor(list(range(10)) + list(range(20, 25)) + list(range(5)))
        expect = list(range(9, -1, -1)) + list(range(4, -1, -1)) + list(range(4, -1, -1))
        self.assertEqual(run_remaining(episode, step).tolist(), expect)


class L2DataTests(unittest.TestCase):
    def test_z2_macro_and_splits(self):
        parts = fake_parts()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.h5'
            episode = three_episode_cache(p)
            with h5py.File(p, 'r') as f:
                embed = torch.from_numpy(f['embed'][:])
                action = torch.from_numpy(f['action'][:])
            data = L2Data(p, parts, holdout_every=100)
        self.assertTrue(torch.equal(data.z2, 2 * embed))
        # macro is NaN exactly where fewer than 10 frames remain in the run
        self.assertTrue(torch.equal(torch.isnan(data.macro).all(-1), data.remaining < 10))
        for t in (0, 50, 109, 120, 270):
            blocks = action[t:t + STEPS_PER_MACRO].reshape(1, 2, STEPS_PER_MACRO)
            z1_seq = embed[t + torch.tensor([0, 5, 10])][None]
            expect = parts.data_macro(z1_seq, blocks)[0]
            self.assertTrue(torch.equal(data.macro[t], expect))
        # splits by episode % 100: episodes 0 and 100 are held out
        train = torch.nonzero(torch.from_numpy(episode) % 100 != 0).flatten()
        heldout = torch.nonzero(torch.from_numpy(episode) % 100 == 0).flatten()
        self.assertTrue(torch.equal(data.splits['train'][0].sort().values, train))
        self.assertTrue(torch.equal(data.splits['heldout'][0].sort().values, heldout))
        for split, mask in (('train', episode % 100 != 0), ('heldout', episode % 100 == 0)):
            mask = torch.from_numpy(mask)
            for need in (1, 10, 50, 119, 120, 121):
                expect = torch.nonzero(mask & (data.remaining >= need)).flatten()
                self.assertTrue(torch.equal(data.frames_with(need, split), expect))

    def test_holdout_every_none_gives_all_split(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.h5'
            three_episode_cache(p)
            data = L2Data(p, fake_parts(), holdout_every=None)
        self.assertEqual(list(data.splits), ['all'])
        self.assertEqual(len(data.splits['all'][0]), 360)

    def test_sample_matched_and_jittered(self):
        parts = fake_parts()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.h5'
            episode = three_episode_cache(p)
            data = L2Data(p, parts, holdout_every=100)
        episode = torch.from_numpy(episode)
        for h in (2, 8):
            gen = torch.Generator().manual_seed(h)
            z2, g2, expert, mask, offset, t = data.sample(128, h, gen, 'train', match_prob=1.0)
            self.assertTrue(bool((offset == STEPS_PER_MACRO * h - 5).all()))
            self.assertTrue(bool(mask.all()))
            self.assertTrue(bool((data.remaining[t] >= STEPS_PER_MACRO * h).all()))
            for j in range(h):
                self.assertTrue(torch.equal(expert[:, j], data.macro[t + STEPS_PER_MACRO * j]))
            self.assertTrue(bool(torch.isfinite(expert).all()))
            self.assertTrue(torch.equal(g2, data.z2[t + offset]))
            self.assertTrue(bool((episode[t] % 100 != 0).all()))

            gen = torch.Generator().manual_seed(h)
            z2, g2, expert, mask, offset, t = data.sample(
                128, h, gen, 'heldout', match_prob=0.0)
            lo, hi = max(5, STEPS_PER_MACRO * h - 5 + JITTER[0]), STEPS_PER_MACRO * h - 5 + JITTER[1]
            self.assertTrue(bool((offset >= lo).all() and (offset <= hi).all()))
            self.assertTrue(bool((data.remaining[t] >= offset).all()))
            self.assertTrue(not bool(mask.any()))
            self.assertTrue(bool((expert == 0).all()) and bool(torch.isfinite(expert).all()))
            self.assertTrue(torch.equal(g2, data.z2[t + offset]))
            self.assertTrue(bool((episode[t] % 100 == 0).all()))


class MacroBCTests(unittest.TestCase):
    def test_min_over_heads_and_mask(self):
        torch.manual_seed(0)
        out = torch.randn(4, 3, 5, 4, requires_grad=True)
        expert = torch.randn(4, 5, 4)
        prior_std = torch.tensor([1., 2., .5, 1.])
        mask = torch.tensor([True, False, True, False])
        expert[~mask] = 0.
        out.data[0, 2] = expert[0]
        err = ((out.detach() - expert[:, None]) / prior_std).pow(2).sum(-1).mean(-1)
        expect = err.min(dim=1).values[mask].mean()
        value = macro_bc(out, expert, prior_std, mask)
        self.assertAlmostEqual(value.item(), expect.item(), places=6)
        self.assertEqual(err.min(dim=1).values[0].item(), 0.)
        value.backward()
        self.assertTrue(bool(torch.isfinite(out.grad).all()))
        self.assertTrue(bool((out.grad[~mask] == 0).all()))

    def test_empty_mask_returns_zero_with_graph(self):
        out = torch.randn(2, 3, 4, 4, requires_grad=True)
        expert = torch.randn(2, 4, 4)
        value = macro_bc(out, expert, torch.ones(4), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(value.item(), 0.)
        self.assertIsNotNone(value.grad_fn)
        value.backward()
        self.assertTrue(bool((out.grad == 0).all()))


class UpperProposalTests(unittest.TestCase):
    def test_output_shape_box_and_horizon(self):
        torch.manual_seed(0)
        net = UpperProposal(heads=4, width=32, blocks=1, embed_dim=16)
        lo = torch.tensor([-1., -.5, 0., -2.])
        hi = torch.tensor([1., .5, .25, 2.])
        net.lo.copy_(lo)
        net.hi.copy_(hi)
        z2, g2 = torch.randn(5, 16), torch.randn(5, 16)
        out = net(z2, g2, torch.randint(MIN_H, MAX_H + 1, (5,)))
        self.assertEqual(out.shape, (5, 4, MAX_H, 4))
        self.assertTrue(bool((out > lo).all() and (out < hi).all()))
        a = net(z2, g2, torch.full((5,), 2))
        b = net(z2, g2, torch.full((5,), 7))
        self.assertFalse(torch.equal(a, b))
        for h in (MIN_H - 1, MAX_H + 1):
            with self.assertRaises(RuntimeError):
                net(z2, g2, torch.full((5,), h))


class Level2CostTests(unittest.TestCase):
    def test_info_dict_layout(self):
        b, k, h, d = 2, 3, 6, 8
        z2, g2, cand = torch.randn(b, d), torch.randn(b, d), torch.randn(b, k, h, 4)
        seen = {}

        def get_cost(info, actions):
            seen.update(info)
            return torch.zeros(b, k)

        model = SimpleNamespace(get_cost=get_cost)
        costs = level2_cost(model, z2, g2, cand, action_cost_weight=0.2,
                            cost_last_n=3, intermediate_cost_weight=0.7)
        self.assertEqual(costs.shape, (b, k))
        self.assertEqual(seen['embed_0'].shape, (b, k, 1, d))
        self.assertTrue(torch.equal(seen['embed_0'], z2[:, None, None].expand(b, k, 1, d)))
        self.assertTrue(torch.equal(seen['goal_embed_0'], g2[:, None, None].expand(b, k, 1, d)))
        self.assertEqual(seen['pixels'].shape, (b, k, 1))
        self.assertTrue(bool((seen['pixels'] == 0).all()))
        self.assertEqual(seen['goal_embed_0_level'], 2)
        self.assertIs(seen['goal_embed_0_in_upper_space'], False)
        self.assertIs(seen['dense_upper_goal_projection'], False)
        self.assertEqual(seen['action_cost_weight'], 0.2)
        self.assertEqual(seen['cost_last_n'], 3)
        self.assertEqual(seen['intermediate_cost_weight'], 0.7)


class GenericShapeTests(unittest.TestCase):
    def test_upper_proposal_max_h_12_and_2(self):
        for max_h in (12, 2):
            net = UpperProposal(heads=4, width=64, blocks=2, embed_dim=16, max_h=max_h)
            net.lo.copy_(torch.full((4,), -0.5))
            net.hi.copy_(torch.full((4,), 0.5))
            for h in range(MIN_H, max_h + 1):
                out = net(torch.randn(6, 16), torch.randn(6, 16),
                          torch.full((6,), h, dtype=torch.long))
                self.assertEqual(out.shape, (6, 4, max_h, 4))
                self.assertTrue(float(out.min()) >= -0.5 and float(out.max()) <= 0.5)

    def test_l2_data_action_dim_5_and_8(self):
        for adim in (5, 8):
            parts = SimpleNamespace(
                device='cpu', encode_level2=lambda z: 2 * z,
                data_macro=lambda z1_seq, blocks: blocks.flatten(1)[:, :4] + z1_seq[:, 2, :4],
                action_mean=torch.zeros(adim), action_std=torch.ones(adim))
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'c.h5'
                episode = np.array([0] * 120 + [1] * 120 + [100] * 120)
                step = np.tile(np.arange(120), 3)
                write_cache(p, episode, step, adim=adim)
                with h5py.File(p, 'r') as f:
                    action = torch.from_numpy(f['action'][:])
                data = L2Data(p, parts, holdout_every=100, action_block=5)
            self.assertEqual(data.span, 10)
            self.assertEqual(data.macro.shape[1], 4)
            finite = torch.isfinite(data.macro).all(-1)
            expect = data.remaining >= data.span
            self.assertTrue(torch.equal(finite, expect))

    def test_offset_range_sampling(self):
        parts = SimpleNamespace(
            device='cpu', encode_level2=lambda z: 2 * z,
            data_macro=lambda z1_seq, blocks: blocks.flatten(1)[:, :4] + z1_seq[:, 2, :4],
            action_mean=torch.zeros(2), action_std=torch.ones(2))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.h5'
            episode = np.array([0] * 400 + [1] * 400)
            step = np.tile(np.arange(400), 2)
            write_cache(p, episode, step)
            data = L2Data(p, parts, holdout_every=100)
        gen = torch.Generator().manual_seed(0)
        h = 3
        z2, g2, expert, mask, offset, t = data.sample(
            256, h, gen, 'train', offset_range=(5, 95))
        self.assertTrue(bool(((offset >= 5) & (offset <= 95)).all()))
        # mask = offset >= span*h - 5 = 25
        self.assertTrue(torch.equal(mask, offset >= 25))
        # expert rows are zeros exactly where the mask is False
        zero = (expert.abs().sum((-1, -2)) == 0) | ~torch.isfinite(expert).all((-1, -2))
        self.assertTrue(torch.equal(zero, ~mask))
        # need rule: masked rows need max(offset, span*h); unmasked need offset
        need = torch.where(mask, torch.maximum(offset, torch.full_like(offset, 30)),
                           offset)
        self.assertTrue(bool((data.remaining[t] >= need).all()))
        # goal latents sit at t + offset
        self.assertTrue(torch.equal(g2, data.z2[t + offset]))


if __name__ == '__main__':
    unittest.main()
