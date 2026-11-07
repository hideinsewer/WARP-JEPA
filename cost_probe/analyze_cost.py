"""Separate all lower-level cost from cost reachable by a macro-action decoder."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(report, calls):
    total = report['planning']['evaluation_seconds']
    active = [row for row in calls if row['level2_seconds'] > 0]
    bypassed = [row for row in calls if row['level2_seconds'] == 0]
    eligible = sum(row['level1_seconds'] for row in active)
    ineligible = sum(row['level1_seconds'] for row in bypassed)
    np.testing.assert_allclose(eligible + ineligible, report['planning']['lower_seconds'])
    assert 0 <= eligible < total
    per_task = []
    for seed in sorted({row['planner_seed'] for row in calls}):
        rows = [row for row in calls if row['planner_seed'] == seed]
        plan = sum(row['seconds'] for row in rows)
        low = sum(row['level1_seconds'] for row in rows)
        macro_low = sum(row['level1_seconds'] for row in rows if row['level2_seconds'] > 0)
        per_task.append(dict(planner_seed=seed, planning_seconds=plan,
                             lower_seconds=low, macro_lower_seconds=macro_low,
                             lower_fraction_planning=low / plan,
                             macro_lower_fraction_planning=macro_low / plan))
    return dict(active_macro_calls=len(active), bypassed_high_level_calls=len(bypassed),
                macro_lower_seconds=eligible, no_macro_lower_seconds=ineligible,
                macro_lower_fraction_evaluation=eligible / total,
                zero_macro_lower_cost_max_speedup=1 / (1 - eligible / total),
                median_lower_seconds=float(np.median([r['level1_seconds'] for r in calls])),
                median_active_lower_seconds=float(np.median([r['level1_seconds'] for r in active])) if active else None,
                per_task=per_task,
                interpretation='The bound removes only the lower solve when an upper macro-action exists; all other work stays fixed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads((args.run / 'report.json').read_text()),
                     json.loads((args.run / 'planning_calls.json').read_text()))
    with (args.run / 'decoder_scope.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
