"""Check the statistics and call pairing used to compare probe runs."""

import unittest

import math

from compare_runs import compare, group_stats, label_calls, mcnemar_exact, newcombe_paired_interval, wilson_interval


class CompareRunsTests(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = wilson_interval(3, 5)
        self.assertAlmostEqual(low, 0.2307, places=4)
        self.assertAlmostEqual(high, 0.8824, places=4)

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(mcnemar_exact(4, 1), 0.375)

    def test_newcombe_paired_interval_matches_contingencytables(self):
        # Fagerland, Lydersen & Laake examples: bentur_2009 (psi = 0), cavo_2012 (corrected psi > 0).
        for table, expected in (((1, 1, 7, 12), (-0.5069, -0.0256)), ((59, 6, 16, 80), (-0.1186, -0.0046))):
            for value, reference in zip(newcombe_paired_interval(*table), expected):
                self.assertAlmostEqual(value, reference, places=4)
        low, high = newcombe_paired_interval(10, 4, 4, 32)
        self.assertAlmostEqual(low, -high)

    def test_label_calls_pairs_levels_with_and_without_macro_flags(self):
        calls = [dict(level1_seconds=2., level2_seconds=5.), dict(level1_seconds=1., level2_seconds=0.)]
        old = [dict(level=2, seconds=5.), dict(level=1, seconds=2.), dict(level=1, seconds=1.)]
        new = [dict(level=2, seconds=5.), dict(level=1, macro=True, seconds=2.),
               dict(level=1, macro=False, seconds=1.)]
        for levels in (old, new):
            labelled = label_calls(levels, calls)
            self.assertIs(labelled[0][1], levels[0])
            self.assertIsNone(labelled[1][1])
        with self.assertRaises(AssertionError):
            label_calls([dict(level=2, seconds=5.), dict(level=1, macro=False, seconds=2.),
                         dict(level=1, macro=False, seconds=1.)], calls)
        with self.assertRaises(AssertionError):
            label_calls(old + [dict(level=1, seconds=1.)], calls)

    def test_compare_rejects_different_task_files(self):
        run = dict(tasks=[0, 1], checkpoint_sha256='c', successes=[True, False], success_rate=0.5,
                   planning_seconds_by_seed={42: 10.0, 43: 8.0}, evaluation_seconds=20.0,
                   planning_seconds=18.0, iteration_sequence=[], gpu_visible='g')
        same = compare(dict(run, tasks_sha256='aaa'), dict(run, tasks_sha256='aaa'))
        self.assertEqual(same['both_success'], 1)
        with self.assertRaises(AssertionError):
            compare(dict(run, tasks_sha256='aaa'), dict(run, tasks_sha256='bbb'))
        missing = dict(run)
        self.assertEqual(compare(missing, dict(run, tasks_sha256='aaa'))['both_success'], 1)

    def test_group_stats_zero_iteration_calls(self):
        rows = [dict(n_iters=[0], seconds=0.01), dict(n_iters=[0], seconds=0.02)]
        stats = group_stats(rows)
        self.assertEqual(stats['calls'], 2)
        self.assertEqual(stats['iterations'], 0)
        self.assertTrue(math.isnan(stats['seconds_per_iteration']))
        self.assertNotIn('fit_seconds_per_iteration', stats)


if __name__ == '__main__':
    unittest.main()
