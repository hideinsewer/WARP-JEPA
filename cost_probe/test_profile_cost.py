"""Check the accounting used to decide the maximum possible Decoder benefit."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from profile_cost import (Timing, action_grads_only, apply_upper_overrides, capture_planning_context,
                          coupled_costs, decoder_solve, macro_budget, set_grad_flags, summarize,
                          upper_proposal_solve, warm_start_level2)
from analyze_cost import analyze


class CostAccountingTests(unittest.TestCase):
    def test_planning_and_end_to_end_denominators(self):
        timing = Timing()
        timing.plans = [dict(seconds=10.0, level1_seconds=3.0,
                             level2_seconds=6.0, encoding_seconds=1.0)]
        timing.levels = [dict(level=1, n_iters=[30]), dict(level=2, n_iters=[40])]
        result = summarize(timing, 12.0)
        self.assertEqual(result['lower_fraction_planning'], .3)
        self.assertEqual(result['lower_fraction_evaluation'], .25)
        self.assertAlmostEqual(result['zero_lower_cost_max_speedup'], 4 / 3)
        self.assertEqual(result['level_iterations'], {'1': [30], '2': [40]})

    def test_empty_measurements_fail(self):
        with self.assertRaises(AssertionError):
            summarize(Timing(), 1.0)

    def test_decoder_bound_excludes_calls_without_macro_actions(self):
        report = {'planning': {'evaluation_seconds': 20.0, 'lower_seconds': 7.0}}
        calls = [dict(planner_seed=42, seconds=10., level1_seconds=3., level2_seconds=6.),
                 dict(planner_seed=42, seconds=5., level1_seconds=4., level2_seconds=0.)]
        result = analyze(report, calls)
        self.assertEqual(result['macro_lower_seconds'], 3.)
        self.assertEqual(result['no_macro_lower_seconds'], 4.)
        self.assertAlmostEqual(result['zero_macro_lower_cost_max_speedup'], 20 / 17)

    def test_decoder_solve_routes_macro_calls(self):
        import decoder as decoder_lib

        class FakeDec:
            heads = 2
            def __call__(self, z1, z2, g2, m):
                return torch.zeros(1, 2, 2, 10)

        cfg = SimpleNamespace(cost_last_n=2, intermediate_cost_weight=1.0)
        solver = SimpleNamespace(
            level=1, n_envs=1, _config=cfg, solve_records=[], device=torch.device('cpu'),
            model=SimpleNamespace(upper_models=[object()],
                                  get_cost=lambda info, cand: torch.tensor([[2., 1.]])),
            _expand_info_dict_for_samples=lambda d, **kw: d,
            _rollout_final_actions=lambda info, a: {'predicted': a})
        info = {'embed_0': torch.ones(1, 1, 192), 'goal_embed_0': torch.ones(1, 1, 192),
                'goal_action': torch.ones(1, 1, 4), 'action_cost_weight': 0.2}
        calls = []
        wrapped = decoder_solve(lambda s, i, **kw: calls.append(i) or 'orig', FakeDec())
        with patch.object(decoder_lib, 'encode_level2', lambda model, z: z), \
                patch.object(decoder_lib, 'level1_cost',
                             lambda model, z1, g2, m, cand, **kw: torch.tensor([[2., 1.]])):
            out = wrapped(solver, dict(info), planning_horizon=2)
        self.assertEqual(len(calls), 0)
        self.assertEqual(out['actions'].shape, (1, 2, 10))
        self.assertEqual(solver.solve_records[-1]['n_iters'], [0])
        self.assertTrue(solver.solve_records[-1]['decoder'])
        self.assertEqual(wrapped.check['max_abs_diff'], 0.0)
        bypass = dict(info, goal_embed_0_level=2)
        self.assertEqual(wrapped(solver, bypass, planning_horizon=2), 'orig')
        self.assertEqual(wrapped(solver, dict(info), planning_horizon=1), 'orig')

    def test_macro_budget_only_changes_level1_solves_with_macro_actions(self):
        seen = []

        def original(solver, info_dict, **kwargs):
            seen.append((solver.n_steps, solver.early_stop_patience))
            if info_dict.get('fail'):
                raise RuntimeError

        solve = macro_budget(original, 7)
        low, high = (SimpleNamespace(level=level, n_steps=90, early_stop_patience=5) for level in (1, 2))
        macro = {'goal_action': 'upper plan', 'goal_embed_0_in_upper_space': True}
        no_macro = {'goal_action': 'dataset goal frame', 'goal_embed_0_level': 1}
        solve(low, info_dict=macro, planning_horizon=2)
        solve(low, no_macro)
        solve(high, info_dict={'goal_action': 'dataset goal frame', 'goal_embed_0_level': 2})
        with self.assertRaises(RuntimeError):
            solve(low, info_dict={**macro, 'fail': True})
        self.assertEqual(seen, [(7, 0), (90, 5), (90, 5), (7, 0)])
        self.assertEqual((low.n_steps, low.early_stop_patience), (90, 5))

    def test_macro_and_bypass_budgets_compose(self):
        seen = []

        def original(solver, info_dict, **kwargs):
            seen.append((solver.n_steps, solver.early_stop_patience))
            if info_dict.get('fail'):
                raise RuntimeError

        solve = macro_budget(original, 7, 3)
        low = SimpleNamespace(level=1, n_steps=90, early_stop_patience=5)
        high = SimpleNamespace(level=2, n_steps=90, early_stop_patience=5)
        macro = {'goal_action': 'upper plan', 'goal_embed_0_in_upper_space': True}
        bypass = {'goal_embed_0_level': 1}
        solve(low, macro)
        solve(low, bypass)
        solve(high, bypass)
        with self.assertRaises(RuntimeError):
            solve(low, {**bypass, 'fail': True})
        self.assertEqual(seen, [(7, 0), (3, 0), (90, 5), (3, 0)])
        self.assertEqual((low.n_steps, low.early_stop_patience), (90, 5))
        self.assertEqual((high.n_steps, high.early_stop_patience), (90, 5))

    def test_warm_start_sequence(self):
        calls = []

        def original(solver, info_dict, **kwargs):
            calls.append(kwargs.copy())
            solver.solve_records.append({'level': solver.level})
            return {'actions': solver.next_actions}

        solve = warm_start_level2(original, 10)
        up = SimpleNamespace(level=2, n_envs=1, solve_records=[], next_actions=None)
        low = SimpleNamespace(level=1, n_envs=1, solve_records=[], next_actions=torch.ones(1, 3, 2))
        up.next_actions = torch.arange(8., dtype=torch.float32).view(1, 4, 2)
        solve(up, {}, steps_taken=0)
        solve(low, {}, steps_taken=0)
        up.next_actions = torch.ones(1, 4, 2)
        solve(up, {}, steps_taken=10)
        solve(up, {}, steps_taken=30)
        solve(up, {}, steps_taken=0)
        self.assertNotIn('init_action', calls[0])
        self.assertNotIn('init_action', calls[1])
        self.assertTrue(torch.equal(calls[2]['init_action'], torch.arange(8.).view(1, 4, 2)[:, 1:]))
        self.assertNotIn('init_action', calls[3])
        self.assertNotIn('init_action', calls[4])
        self.assertEqual([r['warm_start'] for r in up.solve_records], [False, True, False, False])
        self.assertEqual([r['warm_start'] for r in low.solve_records], [False])

    def test_upper_overrides_only_change_level2(self):
        cfg = OmegaConf.create({'solver': {'solvers': {
            'level1': {'n_steps': 90, 'early_stop_patience': 5, 'num_samples': 50},
            'level2': {'n_steps': 90, 'num_samples': 400}}}})
        apply_upper_overrides(cfg, n_steps=5, num_samples=100)
        level2 = cfg.solver.solvers.level2
        self.assertEqual((level2.n_steps, level2.early_stop_patience, level2.num_samples), (5, 0, 100))
        level1 = cfg.solver.solvers.level1
        self.assertEqual((level1.n_steps, level1.early_stop_patience, level1.num_samples), (90, 5, 50))

    def test_grad_flags_restore_and_clear_gradients(self):
        model = torch.nn.Linear(2, 1)
        flags = [p.requires_grad for p in model.parameters()]
        model(torch.ones(1, 2)).sum().backward()
        set_grad_flags(model, [False] * len(flags))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        set_grad_flags(model, flags)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)

    def test_action_grads_only_restricts_backward_inside_solve(self):
        model = torch.nn.Linear(2, 1)

        def original(solver, info_dict):
            x = torch.ones(1, 2, requires_grad=True)
            solver.optimizer_cls([x], lr=0.1)
            model(x).sum().backward()
            return x

        state, solve, backward = action_grads_only(original, torch.Tensor.backward)
        solver = SimpleNamespace(optimizer_cls=torch.optim.SGD)
        with patch.object(torch.Tensor, 'backward', backward):
            x = solve(solver, {})
            self.assertTrue(torch.equal(x.grad, model.weight.detach()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertIs(solver.optimizer_cls, torch.optim.SGD)
            self.assertEqual(state.targets, [])
            state.enabled = False
            solve(solver, {})
            self.assertTrue(all(p.grad is not None for p in model.parameters()))


class UpperProposalSolveTests(unittest.TestCase):
    """Level-2 solves route to the proposal network's best-of-K under the lean cost."""

    def _solver(self):
        return SimpleNamespace(
            level=2, n_envs=1, device=torch.device('cpu'), n_steps=90,
            early_stop_patience=5, num_samples=400, solve_records=[],
            _config=SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0),
            model=SimpleNamespace(
                get_cost=lambda info, cand: cand.pow(2).sum((-1, -2))),
            _expand_info_dict_for_samples=lambda d, **kw: d,
            _rollout_final_actions=lambda info, a: {'predicted_embed_0': a})

    def _info(self):
        return {'embed_0': torch.ones(1, 1, 192),
                'goal_embed_0': torch.full((1, 1, 192), 2.),
                'action_cost_weight': 0.2}

    def _net_and_patch(self):
        import upper_proposal as up_lib
        plans = torch.zeros(1, 3, 8, 4)
        plans[0, 0], plans[0, 1], plans[0, 2] = 3., 1., 2.
        fake_cost = lambda model, z2, g2, cand, **kw: model.get_cost({}, cand)
        return plans, lambda z2, g2, h: plans, patch.object(up_lib, 'level2_cost', fake_cost)

    def test_routes_level2_to_best_of_k(self):
        plans, net, fake = self._net_and_patch()
        solver = self._solver()
        calls = []
        wrapped = upper_proposal_solve(lambda s, i, **kw: calls.append(kw.copy()) or 'orig', net)
        with fake:
            out = wrapped(solver, self._info(), planning_horizon=4)
        self.assertEqual(calls, [])
        self.assertEqual(out['actions'].shape, (1, 4, 4))
        self.assertEqual(out['actions'].device.type, 'cpu')
        self.assertTrue(torch.equal(out['actions'], plans[0, 1, :4][None]))
        record = solver.solve_records[-1]
        self.assertEqual(record['n_iters'], [0])
        self.assertTrue(record['proposal'])
        self.assertEqual(record['proposal_candidates'], 3)
        self.assertIs(record['proposal_warm_candidate'], False)
        self.assertEqual(record['proposal_chosen'], [1])
        self.assertEqual(record['proposal_chosen_warm'], [False])
        self.assertEqual(wrapped.check['max_abs_diff'], 0.0)

    def test_level1_calls_pass_through_untouched(self):
        _, net, _ = self._net_and_patch()
        calls = []
        wrapped = upper_proposal_solve(lambda s, i, **kw: calls.append(kw.copy()) or 'orig', net)
        low = SimpleNamespace(level=1)
        init = torch.ones(1, 3, 4)
        out = wrapped(low, self._info(), planning_horizon=2, init_action=init)
        self.assertEqual(out, 'orig')
        self.assertIs(calls[0]['init_action'], init)

    def test_warm_init_appended_as_last_candidate(self):
        plans, net, fake = self._net_and_patch()
        solver = self._solver()
        wrapped = upper_proposal_solve(lambda s, i, **kw: 'orig', net)
        init = torch.zeros(1, 4, 4)
        with fake:
            out = wrapped(solver, self._info(), planning_horizon=4, init_action=init)
        record = solver.solve_records[-1]
        self.assertEqual(record['proposal_candidates'], 4)
        self.assertIs(record['proposal_warm_candidate'], True)
        self.assertEqual(record['proposal_chosen_warm'], [True])
        self.assertTrue(torch.equal(out['actions'], init))
        init = torch.full((1, 4, 4), 10.)
        with fake:
            out = wrapped(solver, self._info(), planning_horizon=4, init_action=init)
        record = solver.solve_records[-1]
        self.assertEqual(record['proposal_chosen'], [1])
        self.assertEqual(record['proposal_chosen_warm'], [False])
        self.assertTrue(torch.equal(out['actions'], plans[0, 1, :4][None]))

    def test_init_action_with_wrong_horizon_raises(self):
        _, net, fake = self._net_and_patch()
        wrapped = upper_proposal_solve(lambda s, i, **kw: 'orig', net)
        with fake, self.assertRaises(AssertionError):
            wrapped(self._solver(), self._info(), planning_horizon=4,
                    init_action=torch.zeros(1, 5, 4))

    def test_refine_iters_run_original_from_candidates(self):
        plans, net, fake = self._net_and_patch()
        solver = self._solver()
        init = torch.zeros(1, 4, 4)
        expected = torch.cat([plans[:, :, :4], init[:, None]], dim=1)
        seen = []

        def original(s, i, **kw):
            seen.append(dict(kw, n_steps=s.n_steps, patience=s.early_stop_patience,
                             num_samples=s.num_samples))
            s.init_action()
            self.assertTrue(torch.equal(s.init.data, expected))
            s.solve_records.append({'level': 2, 'n_iters': [s.n_steps]})
            return {'actions': torch.zeros(1, kw['planning_horizon'], 4)}

        wrapped = upper_proposal_solve(original, net, refine_iters=2)
        with fake:
            out = wrapped(solver, self._info(), planning_horizon=4, init_action=init)
        self.assertEqual(len(seen), 1)
        self.assertNotIn('init_action', seen[0])
        self.assertEqual((seen[0]['n_steps'], seen[0]['patience'], seen[0]['num_samples']),
                         (2, 0, 4))
        self.assertEqual((solver.n_steps, solver.early_stop_patience, solver.num_samples),
                         (90, 5, 400))
        self.assertNotIn('init_action', vars(solver))
        record = solver.solve_records[-1]
        self.assertEqual(record['n_iters'], [2])
        self.assertTrue(record['proposal'])
        self.assertEqual(record['proposal_candidates'], 4)
        self.assertEqual(record['proposal_chosen_warm'], [True])
        self.assertEqual(out['actions'].shape, (1, 4, 4))

    def test_couple_and_refine_are_exclusive(self):
        _, net, _ = self._net_and_patch()
        with self.assertRaises(AssertionError):
            upper_proposal_solve(lambda s, i, **kw: 'orig', net,
                                 refine_iters=2, couple=(object(), {}))


class CapturePlanningContextTests(unittest.TestCase):
    def test_stores_solver_and_embed1_on_pixels_key(self):
        context = {}
        solver, other = object(), object()
        result = {'embed_1': 'E1', 'embed_2': 'E2'}
        wrapped = capture_planning_context(lambda s, *a, **kw: result, context)
        # keyword key
        self.assertIs(wrapped(solver, {}, key='pixels'), result)
        self.assertIs(context['hsolver'], solver)
        self.assertEqual(context['embed_1'], 'E1')
        # positional key
        self.assertIs(wrapped(other, {}, 'pixels'), result)
        self.assertIs(context['hsolver'], other)
        # default key counts as pixels
        self.assertIs(wrapped(solver, {}), result)
        self.assertIs(context['hsolver'], solver)
        # key='goal' leaves the stored values unchanged
        self.assertIs(wrapped(other, {}, key='goal'), result)
        self.assertIs(context['hsolver'], solver)
        self.assertEqual(context['embed_1'], 'E1')
        self.assertIs(wrapped(other, {}, 'goal'), result)
        self.assertIs(context['hsolver'], solver)


class CoupledCostsTests(unittest.TestCase):
    """The coupled cost: decode the first macro to a level-1 plan, roll it out, and cost the
    remaining macros from the end latent toward the level-2 goal."""

    def test_wiring(self):
        import profile_cost as pc
        b, c, h, heads = 2, 3, 4, 16
        d1, d2 = 6, 8
        torch.manual_seed(0)
        z2 = torch.randn(b, d2)
        g2 = torch.randn(b, d2)
        cand = torch.randn(b, c, h, 4)
        embed_1 = torch.randn(b, 3, d1)

        # level-2 rollout: subgoal (index 1) encodes the candidate's first macro
        def l2_rollout(info, actions):
            pred = torch.zeros(actions.shape[0], actions.shape[1], 3, d2)
            pred[:, :, 1, :4] = actions[:, :, 0]
            pred[:, :, 1, 4:] = actions[:, :, 0].sum(-1, keepdim=True)
            return {'predicted_embed_0': pred}

        l2_model = SimpleNamespace(rollout=l2_rollout)
        expected_sub = l2_rollout(None, cand)['predicted_embed_0'][:, :, 1]

        # level-1 decoder returns 16 distinct plans per flattened row
        plans = (torch.arange(b * c * heads * 20, dtype=torch.float32)
                 .reshape(b * c, heads, 2, 10) * 1e-4)
        seen_dec = {}

        def fake_dec(z1, z2_, sub, m):
            seen_dec.update(z1=z1, z2=z2_, sub=sub, m=m)
            return plans

        # level-1 cost: argmin plan index differs per row in a known way
        rows = torch.arange(b * c)
        argmin = (3 * rows + 1) % heads
        seen_l1 = {}

        def fake_level1_cost(model, z1, g2_, m, plans_, **kw):
            seen_l1.update(model=model, z1=z1, g2=g2_, m=m, plans=plans_, kw=kw)
            costs = torch.ones(b * c, heads)
            costs[rows, argmin] = 0.25 + 0.01 * rows
            return costs

        # level-1 rollout: end latent [:, 0, -1] encodes the plan it received
        def l1_rollout(info, actions):
            end = actions[:, 0].sum((-1, -2))
            pred = torch.zeros(actions.shape[0], 1, 3, d1)
            pred[:, 0, -1] = end[:, None]
            return {'predicted_embed_0': pred}

        l1_model = SimpleNamespace(rollout=l1_rollout)
        low_solver = SimpleNamespace(
            model=l1_model,
            _config=SimpleNamespace(cost_last_n=2, intermediate_cost_weight=0.7))
        hsolver = SimpleNamespace(
            level_solvers={1: low_solver},
            level_models=[None, 'level2_jepa'],
            _level_plan_config=lambda level: SimpleNamespace(action_cost_weight=0.35))
        context = dict(hsolver=hsolver, embed_1=embed_1)

        seen_l2 = {}

        def fake_level2_cost(model, z2_, g2_, rest_, **kw):
            seen_l2.update(model=model, z2=z2_, g2=g2_, cand=rest_, kw=kw)
            return (rest_[:, 0].sum((-1, -2)) + z2_.mean(-1))[:, None]

        with patch.object(pc.decoder, 'level1_cost', fake_level1_cost), \
                patch.object(pc.decoder, 'encode_level2', lambda jepa, z: z), \
                patch.object(pc.upper_proposal, 'level2_cost', fake_level2_cost):
            rest, low_costs = coupled_costs(fake_dec, context, l2_model, z2, g2, cand,
                                            dict(action_cost_weight=0.2, cost_last_n=1,
                                                 intermediate_cost_weight=1.0))

        self.assertEqual(rest.shape, (b, c))
        self.assertEqual(low_costs.shape, (b, c))
        # decoder inputs per flattened row r = b*c + col
        self.assertTrue(torch.equal(seen_dec['z1'].view(b, c, d1),
                                    embed_1[:, -1][:, None].expand(b, c, d1)))
        self.assertTrue(torch.equal(seen_dec['z2'].view(b, c, d2),
                                    z2[:, None].expand(b, c, d2)))
        self.assertTrue(torch.equal(seen_dec['sub'].view(b, c, d2), expected_sub))
        self.assertTrue(torch.equal(seen_dec['m'].view(b, c, 4), cand[:, :, 0]))
        # level-1 cost kwargs come from the level-1 plan config / solver config
        self.assertEqual(seen_l1['kw']['action_cost_weight'], 0.35)
        self.assertEqual(seen_l1['kw']['cost_last_n'], 2)
        self.assertEqual(seen_l1['kw']['intermediate_cost_weight'], 0.7)
        self.assertIs(seen_l1['model'], l1_model)
        # level-1 costs returned are the row minima
        self.assertTrue(torch.equal(low_costs, (0.25 + 0.01 * rows).view(b, c)))
        # each level-2 row gets exactly cand[b, c, 1:], that row's end latent, and g2[b]
        self.assertTrue(torch.equal(seen_l2['cand'][:, 0].view(b, c, h - 1, 4),
                                    cand[:, :, 1:]))
        expected_end = plans[rows, argmin].sum((-1, -2))
        self.assertTrue(torch.equal(seen_l2['z2'], expected_end[:, None].expand(b * c, d1)))
        self.assertTrue(torch.equal(seen_l2['g2'].view(b, c, d2),
                                    g2[:, None].expand(b, c, d2)))
        expected_rest = cand[:, :, 1:].sum((-1, -2)) + expected_end.view(b, c) * 1.0
        self.assertTrue(torch.allclose(rest, expected_rest))
        self.assertEqual(seen_l2['kw']['action_cost_weight'], 0.2)


class CoupledProposalSolveTests(unittest.TestCase):
    def test_coupled_pick_overrides_level2_argmin(self):
        import profile_cost as pc
        plans = torch.zeros(1, 3, 8, 4)
        plans[0, 0], plans[0, 1], plans[0, 2] = 3., 1., 2.
        net = lambda z2, g2, h: plans
        embed_0_last = torch.full((1, 192), 0.5)
        info = {'embed_0': embed_0_last[:, None], 'goal_embed_0': torch.full((1, 1, 192), 2.),
                'action_cost_weight': 0.2}
        solver = SimpleNamespace(
            level=2, n_envs=1, device=torch.device('cpu'), solve_records=[],
            _config=SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0),
            model=SimpleNamespace(
                get_cost=lambda info_, cand: cand.pow(2).sum((-1, -2))),
            _expand_info_dict_for_samples=lambda d, **kw: d,
            _rollout_final_actions=lambda info_, a: {'predicted_embed_0': a})
        # level-2 costs: argmin is candidate 1; coupled rest argmin is candidate 2
        rest = torch.tensor([[5., 4., 0.5]])
        low_costs = torch.tensor([[0.3, 0.1, 0.7]])
        context = dict(hsolver=SimpleNamespace(level_models=[None, None]),
                       embed_1=torch.zeros(1, 1, 8))
        wrapped = upper_proposal_solve(lambda s, i, **kw: 'orig', net,
                                       couple=(lambda *a: None, context))
        with patch.object(pc.upper_proposal, 'level2_cost',
                          lambda model, z2, g2, cand, **kw: model.get_cost({}, cand)), \
                patch.object(pc, 'coupled_costs',
                             lambda dec, ctx, model, z2, g2, cand, kw: (rest, low_costs)), \
                patch.object(pc.decoder, 'encode_level2',
                             lambda jepa, z1: embed_0_last):
            out = wrapped(solver, info, planning_horizon=4)
        self.assertTrue(torch.equal(out['actions'], plans[0, 2, :4][None]))
        record = solver.solve_records[-1]
        self.assertEqual(record['proposal_chosen'], [1])
        self.assertEqual(record['coupled_chosen'], [2])
        self.assertEqual(record['coupled_differs'], [True])
        self.assertAlmostEqual(record['coupled_rest_cost'][0], 0.5)
        self.assertAlmostEqual(record['coupled_level2_cost'][0], 4.0 * 16)
        self.assertAlmostEqual(record['coupled_level1_cost'][0], 0.7)
        self.assertAlmostEqual(record['proposal_rest_cost'][0], 4.0)
        self.assertAlmostEqual(record['proposal_level1_cost'][0], 0.1)
        self.assertEqual(wrapped.check['max_abs_diff'], 0.0)
        self.assertEqual(wrapped.check['context_z2_max_abs_diff'], 0.0)


if __name__ == '__main__':
    unittest.main()
