"""Compare cost-probe runs: paired task success and per-level planner time."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(k, n, z=Z95):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [centre - half, centre + half]


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def newcombe_paired_interval(n11, n12, n21, n22, z=Z95):
    """Newcombe (1998) square-and-add CI for p_first - p_second of a paired 2x2 table
    (n12: first-only successes), as in R contingencytables::Newcombe_square_and_add_CI_paired_2x2."""
    n = n11 + n12 + n21 + n22
    rows, cols = (n11 + n12, n21 + n22), (n11 + n21, n12 + n22)
    p1, p2 = rows[0] / n, cols[0] / n
    l1, u1 = wilson_interval(rows[0], n, z)
    l2, u2 = wilson_interval(cols[0], n, z)
    a, psi = n11 * n22 - n12 * n21, 0.0
    if min(rows + cols) > 0 and (a > n / 2 or a < 0):
        psi = (a - n / 2 if a > 0 else a) / math.sqrt(math.prod(rows + cols))
    low = math.sqrt(max(0., (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * psi * (p1 - l1) * (u2 - p2)))
    high = math.sqrt(max(0., (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * psi * (p2 - l2) * (u1 - p1)))
    return [p1 - p2 - low, p1 - p2 + high]


def label_calls(levels, calls):
    """Pair level records with planning calls; a level-1 solve is macro iff level 2 planned."""
    records, labelled = iter(levels), []
    for call in calls:
        macro = call['level2_seconds'] > 0
        upper = next(records) if macro else None
        lower = next(records)
        assert lower['level'] == 1 and lower.get('macro', macro) == macro, (call, lower)
        assert math.isclose(lower['seconds'], call['level1_seconds'])
        assert upper is None or (upper['level'] == 2
                                 and math.isclose(upper['seconds'], call['level2_seconds']))
        labelled.append((call, upper, lower))
    assert next(records, None) is None, 'Unpaired level records.'
    return labelled


def group_stats(rows):
    if not rows:
        return dict(calls=0)
    assert all(len(r['n_iters']) == 1 for r in rows)
    iters = np.array([r['n_iters'][0] for r in rows], float)
    secs = np.array([r['seconds'] for r in rows], float)
    out = dict(calls=len(rows), iterations=int(iters.sum()), seconds=float(secs.sum()),
               seconds_per_iteration=float(secs.sum() / iters.sum()) if iters.sum() else math.nan,
               median_call_seconds=float(np.median(secs)), median_iterations=float(np.median(iters)),
               min_iterations=int(iters.min()), max_iterations=int(iters.max()))
    if len(np.unique(iters)) >= 3:
        slope, intercept = np.polyfit(iters, secs, 1)
        out.update(fit_seconds_per_iteration=float(slope), fit_fixed_seconds=float(intercept))
    return out


def summarize_run(run):
    report = json.loads((run / 'report.json').read_text())
    levels = json.loads((run / 'level_calls.json').read_text())
    calls = json.loads((run / 'planning_calls.json').read_text())
    meta, plan, metrics = report['metadata'], report['planning'], report['evaluation_metrics']
    if 'macro_rule' not in meta:
        # Runs before the goal_embed_0_level rule tagged macro by goal_action, which every solve has.
        levels = [{k: v for k, v in r.items() if k != 'macro'} for r in levels]
    labelled = label_calls(levels, calls)
    success = [bool(s) for s in metrics['episode_successes']]
    n, k = len(success), sum(success)
    assert n == len(meta['tasks'])
    upper = [u for _, u, _ in labelled if u is not None]
    macro = [low for _, u, low in labelled if u is not None]
    bypass = [low for _, u, low in labelled if u is None]
    total = plan['evaluation_seconds']
    macro_seconds = sum(r['seconds'] for r in macro)
    bypass_seconds = sum(r['seconds'] for r in bypass)
    assert math.isclose(sum(r['seconds'] for r in upper), plan['upper_seconds'])
    assert math.isclose(macro_seconds + bypass_seconds, plan['lower_seconds'])
    latencies = [c['seconds'] for c in calls]
    steps = [s for s in metrics['steps_to_success'] if s is not None and math.isfinite(s)]
    by_seed = {}
    for c in calls:
        by_seed[c['planner_seed']] = by_seed.get(c['planner_seed'], 0.0) + c['seconds']
    assert len(by_seed) == n
    parity = {key: ({k: v for k, v in value.items() if k != 'solve_iterations'}
                    if isinstance(value, dict) else value)
              for key, value in report['parity'].items()}
    mode = ('frozen' if meta.get('freeze_model_params') else
            'action-grads' if meta.get('action_grads_only') else 'released')
    return dict(
        run=run.name, mode=mode, lower_macro_iters=meta.get('lower_macro_iters'), tasks=meta['tasks'],
        gpu_visible=meta['gpu_visible'], git_commit=meta['git_commit'], git_dirty=meta.get('git_dirty'),
        checkpoint_sha256=meta['checkpoint_sha256'], tasks_sha256=meta.get('tasks_sha256'),
        successes=success, success_count=k, task_count=n, success_rate=k / n,
        success_wilson95=wilson_interval(k, n),
        median_steps_to_success=float(np.median(steps)) if steps else None,
        evaluation_seconds=total, evaluation_seconds_per_task=total / n,
        planning_seconds=plan['planning_seconds'], upper_seconds=plan['upper_seconds'],
        lower_macro_seconds=macro_seconds, lower_no_macro_seconds=bypass_seconds,
        upper_fraction=plan['upper_seconds'] / total, lower_macro_fraction=macro_seconds / total,
        lower_no_macro_fraction=bypass_seconds / total,
        zero_macro_lower_cost_max_speedup=total / (total - macro_seconds),
        median_plan_seconds=float(np.median(latencies)),
        p95_plan_seconds=float(np.quantile(latencies, .95)),
        median_macro_plan_seconds=float(np.median([c['seconds'] for c, u, _ in labelled if u is not None])),
        groups=dict(upper=group_stats(upper), lower_macro=group_stats(macro),
                    lower_no_macro=group_stats(bypass)),
        planning_seconds_by_seed=by_seed,
        iteration_sequence=[[r['level'], r['n_iters'][0]] for r in levels],
        components={name: value['median_ms'] for name, value in report['components'].items()
                    if isinstance(value, dict) and 'median_ms' in value},
        parity=parity, peak_allocated_gib=plan['peak_allocated_gib'])


def compare(reference, row):
    assert row['tasks'] == reference['tasks'], 'Different task lists.'
    assert row['checkpoint_sha256'] == reference['checkpoint_sha256'], 'Different checkpoints.'
    if reference.get('tasks_sha256') and row.get('tasks_sha256'):
        assert row['tasks_sha256'] == reference['tasks_sha256'], 'Different task files.'
    pairs = list(zip(reference['successes'], row['successes']))
    both, neither = sum(a and b for a, b in pairs), sum(not a and not b for a, b in pairs)
    reference_only = sum(a and not b for a, b in pairs)
    run_only = sum(b and not a for a, b in pairs)
    ratios = [reference['planning_seconds_by_seed'][seed] / row['planning_seconds_by_seed'][seed]
              for seed in reference['planning_seconds_by_seed']]
    return dict(both_success=both, reference_only=reference_only,
                run_only=run_only, neither=neither,
                mcnemar_exact_p=mcnemar_exact(reference_only, run_only),
                success_difference=row['success_rate'] - reference['success_rate'],
                success_difference_newcombe95=newcombe_paired_interval(
                    both, run_only, reference_only, neither),
                evaluation_speedup=reference['evaluation_seconds'] / row['evaluation_seconds'],
                planning_speedup=reference['planning_seconds'] / row['planning_seconds'],
                median_task_planning_speedup=float(np.median(ratios)),
                iteration_sequence_identical=row['iteration_sequence'] == reference['iteration_sequence'],
                same_gpu=row['gpu_visible'] == reference['gpu_visible'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    reference = summarize_run(args.reference)
    rows = [{**row, 'paired_vs_reference': compare(reference, row)}
            for row in map(summarize_run, args.runs)]
    with args.out.open('x') as stream:
        json.dump(dict(reference=reference, runs=rows), stream, indent=2)
    print(f"{'run':30} {'mode':>4} {'K':>3} {'succ':>6} {'wilson95':>11} {'ref/run only':>12} {'p':>6} "
          f"{'diff pp (95%)':>15} "
          f"{'eval_s':>8} {'speedup':>7} {'up%':>5} {'Lmac%':>5} {'Lbyp%':>5} {'L_s/it':>7} {'U_s/it':>7}")
    for row in [reference, *rows]:
        paired = row.get('paired_vs_reference')
        low, high = row['success_wilson95']
        discordant = f"{paired['reference_only']}/{paired['run_only']}" if paired else 'ref'
        p_value = f"{paired['mcnemar_exact_p']:.3f}" if paired else '-'
        difference = '-'
        if paired:
            d_low, d_high = paired['success_difference_newcombe95']
            difference = (f"{100 * paired['success_difference']:+.0f} "
                          f"[{100 * d_low:+.0f},{100 * d_high:+.0f}]")
        speedup = paired['evaluation_speedup'] if paired else 1.0
        print(f"{row['run'][:30]:30} {row['mode'][:4]:>4} "
              f"{str(row['lower_macro_iters'] or '-'):>3} {row['success_count']:>2}/{row['task_count']:<3} "
              f"{low:>5.2f}-{high:<5.2f} {discordant:>12} {p_value:>6} {difference:>15} "
              f"{row['evaluation_seconds']:>8.1f} {speedup:>7.3f} "
              f"{100 * row['upper_fraction']:>5.1f} {100 * row['lower_macro_fraction']:>5.1f} "
              f"{100 * row['lower_no_macro_fraction']:>5.1f} "
              f"{row['groups']['lower_macro'].get('seconds_per_iteration', math.nan):>7.4f} "
              f"{row['groups']['upper'].get('seconds_per_iteration', math.nan):>7.4f}")


if __name__ == '__main__':
    main()
