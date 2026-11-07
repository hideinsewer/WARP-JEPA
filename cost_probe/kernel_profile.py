"""Where planner solve time goes: wall time versus GPU kernel time, and time per GD step
versus candidate count and level-2 horizon.

Captures the first planning input of official Push-T task 0, then times fixed-step level-2 and
level-1 solves (no early stopping, action gradients only, MATH attention as in evaluation) and
profiles one solve per level with torch.profiler. Wall times come from unprofiled repeats.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import open_dict
from torch.autograd import DeviceType
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

import profile_cost as pc

evaluation, GradientSolver = pc.evaluation, pc.GradientSolver
TASKS_SHA256 = '588e8db4a6be13e41f00b83332906ac91bc6fda8e3e211a06c8a31431b0d0742'


def union_us(intervals):
    total, start, end = 0.0, None, None
    for begin, finish in sorted(intervals):
        if end is None or begin > end:
            total += 0.0 if end is None else end - start
            start, end = begin, finish
        else:
            end = max(end, finish)
    return total + (0.0 if end is None else end - start)


def analyze(prof, iterations):
    """GPU work = kernels and memory copies/sets; ranges such as Optimizer.step are mirrored on the
    GPU timeline as annotations that share a CPU event's name, so they are excluded."""
    events = prof.events()
    cpu_names = {e.name for e in events if e.device_type == DeviceType.CPU}
    device = [e for e in events if e.device_type == DeviceType.CUDA and e.name not in cpu_names]
    names = {}
    for e in device:
        entry = names.setdefault(e.name, [0, 0.0])
        entry[0] += 1
        entry[1] += e.time_range.elapsed_us()
    busy = union_us([(e.time_range.start, e.time_range.end) for e in device])
    return dict(device_events=len(device),
                device_busy_ms=busy / 1000,
                device_busy_ms_per_iteration=busy / 1000 / iterations,
                device_events_per_iteration=len(device) / iterations,
                cpu_aten_ops_per_iteration=sum(e.device_type == DeviceType.CPU and e.name.startswith('aten::')
                                               for e in events) / iterations,
                top_device=[dict(name=name[:160], count=count, total_ms=total / 1000)
                            for name, (count, total) in sorted(names.items(), key=lambda kv: -kv[1][1])[:15]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--upper-samples', type=int, nargs='+', default=[25, 100, 400, 1600, 6400])
    parser.add_argument('--lower-samples', type=int, nargs='+', default=[50, 200, 800, 3200])
    parser.add_argument('--steps-taken', type=int, nargs='+', default=[0, 30, 60])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    artifacts = pc.ROOT / 'cost_probe/artifacts'
    cfg = evaluation.load_eval_config('pusht_l2')
    cfg.cache_dir = str(artifacts)
    cfg.policy = str(artifacts / 'models/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt')
    task_path = artifacts / 'data/eval_trajs/pusht/goal_offset_75_val.pt'
    assert pc.sha256(task_path) == TASKS_SHA256, 'Task artifact checksum mismatch.'
    cfg.load_eval_trajs_path = str(task_path)
    cfg.eval.num_eval = 1
    cfg.eval.chunk_size = 1
    defaults = {}
    for level in (1, 2):
        with open_dict(cfg.solver.solvers[f'level{level}']) as node:
            node.n_steps, node.early_stop_patience = args.iters, 0
            defaults[level] = int(node.num_samples)
    model = evaluation.load_model(cfg)

    _, solve, backward = pc.action_grads_only(GradientSolver.solve, torch.Tensor.backward)
    patches = [patch.object(GradientSolver, 'solve', solve), patch.object(torch.Tensor, 'backward', backward)]
    for item in patches:
        item.start()
    timing = pc.Timing()
    for item in timing.install():
        item.start()
        patches.append(item)

    records, profiles, state = [], {}, dict(profile_level=None)
    inner = GradientSolver.solve

    def timed(solver, *a, **k):
        with ExitStack() as stack:
            prof = None
            if state['profile_level'] == solver.level:
                prof = stack.enter_context(profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = inner(solver, *a, **k)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
        record = solver.solve_records[-1]
        records.append(dict(level=solver.level, seconds=seconds, iterations=record['n_iters'][0],
                            horizon=record['horizon'], num_samples=record['num_samples']))
        if prof is not None:
            profiles[solver.level] = (prof, records[-1])
        return result

    patches.append(patch.object(GradientSolver, 'solve', timed))
    patches[-1].start()
    try:
        evaluation.run_planning_eval(cfg, model=model, results_dir=args.out / 'warmup')
        info, steps, budget, action_space, plan_config = timing.first_plan
        assert steps == 0

        def solve_once(upper, lower, steps_taken):
            for level, value in ((2, upper), (1, lower)):
                with open_dict(cfg.solver.solvers[f'level{level}']) as node:
                    node.num_samples = value
            solver = evaluation.build_solver(cfg, model)
            solver.configure(action_space=action_space, n_envs=1, config=plan_config)
            records.clear()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                solver.solve(pc.detached(info), steps_taken, budget)
            return {r['level']: r for r in records}

        def measure(upper, lower, steps_taken=0):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                solve_once(upper, lower, steps_taken)
                runs = [solve_once(upper, lower, steps_taken) for _ in range(args.repeats)]
            except torch.OutOfMemoryError:
                return dict(upper_samples=upper, lower_samples=lower, steps_taken=steps_taken, oom=True)
            row = dict(upper_samples=upper, lower_samples=lower, steps_taken=steps_taken,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
            for level in (2, 1):
                if level not in runs[0]:
                    continue
                assert all(r[level]['iterations'] == args.iters for r in runs)
                seconds = [r[level]['seconds'] for r in runs]
                row[f'level{level}'] = dict(horizon=runs[0][level]['horizon'],
                                            median_seconds=float(np.median(seconds)),
                                            median_seconds_per_iteration=float(np.median(seconds)) / args.iters,
                                            seconds=seconds)
            print('MEASURE', json.dumps(row), flush=True)
            return row

        sweep = dict(upper=[measure(n, defaults[1]) for n in args.upper_samples],
                     lower=[measure(defaults[2], n) for n in args.lower_samples],
                     horizon=[measure(defaults[2], defaults[1], s) for s in args.steps_taken])
        reference = measure(defaults[2], defaults[1])
        profiled = {}
        for level in (2, 1):
            state['profile_level'] = level
            solve_once(defaults[2], defaults[1], 0)
            state['profile_level'] = None
            prof, record = profiles[level]
            wall = reference[f'level{level}']['median_seconds']
            summary = analyze(prof, record['iterations'])
            summary.update(profiled_wall_seconds=record['seconds'], unprofiled_median_wall_seconds=wall,
                           gpu_busy_fraction_of_unprofiled_wall=summary['device_busy_ms'] / 1000 / wall,
                           horizon=record['horizon'], num_samples=record['num_samples'],
                           iterations=record['iterations'])
            profiled[f'level{level}'] = summary
            print('PROFILE', f'level{level}', json.dumps({k: v for k, v in summary.items() if k != 'top_device'}),
                  flush=True)
            prof.export_chrome_trace(str(args.out / f'level{level}_trace.json'))
            (args.out / f'level{level}_key_averages.txt').write_text('\n\n'.join(
                prof.key_averages().table(sort_by=key, row_limit=40)
                for key in ('self_device_time_total', 'self_cpu_time_total')))
        gpu_apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader'], capture_output=True, text=True).stdout
        result = dict(
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pc.ROOT, text=True).strip(),
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_compute_apps_at_end=gpu_apps,
            iterations=args.iters, repeats=args.repeats, default_samples=defaults,
            setting='fixed GD steps without early stopping, action gradients only, MATH SDPA; '
                    'first planning input of official task 0 (level-1 solve is macro-conditioned)',
            sweep=sweep, reference=reference, profiled=profiled)
        (args.out / 'kernel_profile.json').write_text(json.dumps(result, indent=2))
        print('KERNEL_PROFILE_COMPLETE', args.out / 'kernel_profile.json', flush=True)
    finally:
        for item in reversed(patches):
            item.stop()


if __name__ == '__main__':
    main()
