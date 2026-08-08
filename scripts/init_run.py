#!/usr/bin/env python3
"""Create a run directory seeded with the benchmark tasks.

A run directory mirrors `benchmark/tasks/`: one `NNN/` folder per task holding
`task_NNN.txt` (the natural-language goal) and `env_NNN.json` (the initial
world state). Planners write `plan_NNN.txt` into these folders and the
evaluator writes `result_NNN.txt` alongside, so each run stays self-contained.

Usage:
    python scripts/init_run.py gpt5_baseline
    python scripts/init_run.py gpt5_baseline --end_idx 10   # smoke-test subset
"""

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simmer.paths import RUNS_DIR, TASKS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', help='Run name, created under runs/')
    parser.add_argument('--runs_dir', type=str, default=str(RUNS_DIR),
                        help='Parent directory for runs (default: runs/)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Only seed tasks with index >= start_idx')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Only seed tasks with index <= end_idx')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite the run directory if it already exists')
    args = parser.parse_args()

    dest = pathlib.Path(args.runs_dir) / args.name
    if dest.exists():
        if not args.force:
            sys.exit(f'Run directory already exists: {dest}\n'
                     f'Re-run with --force to overwrite, or pick another name.')
        shutil.rmtree(dest)

    task_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())
    seeded = 0
    for task_dir in task_dirs:
        idx = int(task_dir.name)
        if args.start_idx is not None and idx < args.start_idx:
            continue
        if args.end_idx is not None and idx > args.end_idx:
            continue
        target = dest / task_dir.name
        target.mkdir(parents=True)
        for name in (f'task_{task_dir.name}.txt', f'env_{task_dir.name}.json'):
            shutil.copy2(task_dir / name, target / name)
        seeded += 1

    if seeded == 0:
        sys.exit('No tasks matched the given index range.')

    print(f'Seeded {seeded} task(s) into {dest}')
    print(f'Next: python scripts/generate_plan.py --base_dir {dest} --model_family gpt')


if __name__ == '__main__':
    main()
