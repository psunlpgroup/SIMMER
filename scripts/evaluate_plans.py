#!/usr/bin/env python3
"""
Evaluate generated plans using the Kitchen State Machine v2.

Runs each plan through the state machine, collects immediate and latent
failures, and writes annotated result files (result_XXX.txt) alongside
each plan.

Usage:
    python evaluate_plans.py                          # evaluate all plans in ./results/v1
    python evaluate_plans.py --base_dir ./results/v1  # specify base directory
    python evaluate_plans.py --end_idx 20             # only evaluate plans 000-020
    python evaluate_plans.py --start_idx 10 --end_idx 20  # evaluate plans 010-020
"""

import argparse
import json
import os
import re
import sys
import pathlib
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.state_machine import KitchenStateMachine, FailureType


def evaluate_single_plan(folder: str, base_dir: str, action_defs_path: str) -> dict:
    """Evaluate a single plan and write its annotated result file.

    Returns a summary dict with failure counts.
    """
    env_path = f'{base_dir}/{folder}/env_{folder}.json'
    plan_path = f'{base_dir}/{folder}/plan_{folder}.txt'
    result_path = f'{base_dir}/{folder}/result_{folder}.txt'
    task_path = f'{base_dir}/{folder}/task_{folder}.txt'

    if not os.path.exists(env_path) or not os.path.exists(plan_path):
        return None

    # Read plan lines
    with open(plan_path) as f:
        plan_lines = [l.rstrip() for l in f.readlines()]

    # Read task description
    task_line = ''
    if os.path.exists(task_path):
        with open(task_path) as f:
            first = f.readline().strip()
            if first.startswith('Task:'):
                task_line = first

    # Run state machine
    sm = KitchenStateMachine(env_path, action_defs_path)
    sm.execute_plan_file(plan_path)

    # Group failures by step
    step_failures = defaultdict(list)
    post_failures = []
    for fail in sm.failures:
        if fail.step == 0:
            post_failures.append(fail)
        else:
            step_failures[fail.step].append(fail)

    # Deduplicated summary
    deduped = sm.deduplicated_failures()
    imm_deduped = [d for d in deduped if d['failure_type'] != FailureType.LATENT]
    latent_deduped = [d for d in deduped if d['failure_type'] == FailureType.LATENT]
    total_imm = sum(1 for f in sm.failures if f.failure_type != FailureType.LATENT and f.step != 0)
    total_latent = sum(1 for f in sm.failures if f.failure_type == FailureType.LATENT)
    total_irreversible = sum(1 for f in sm.failures if not f.reversible)
    total_reversible = sum(1 for f in sm.failures if f.reversible)

    def _format_tag(d):
        timing = d['failure_type'].value.upper()
        rev = 'REVERSIBLE' if d['reversible'] else 'IRREVERSIBLE'
        return f'{timing}/{rev}'

    # Write annotated result
    with open(result_path, 'w') as out:
        out.write(f'# {task_line}\n')
        out.write(f'# Plan: {plan_path}\n')
        out.write(f'# Immediate failures: {total_imm} raw, {len(imm_deduped)} unique (all reversible)\n')
        out.write(f'# Latent failures: {total_latent} ({total_irreversible} irreversible, {total_latent - total_irreversible} reversible)\n')
        out.write(f'#\n')

        # Summary section
        if imm_deduped or latent_deduped:
            out.write(f'# === Failure Summary ===\n')
            for d in sorted(imm_deduped, key=lambda x: -x['count']):
                steps = ','.join(str(s) for s in d['steps'])
                tag = _format_tag(d)
                if d['count'] == 1:
                    out.write(f'#   [{tag}] Step {steps}: {d["reason"]}\n')
                else:
                    out.write(f'#   [{tag}] Steps {steps} (x{d["count"]}): {d["reason"]}\n')
            for d in sorted(latent_deduped, key=lambda x: -x['count']):
                steps = ','.join(str(s) for s in d['steps'])
                tag = _format_tag(d)
                if d['count'] == 1:
                    out.write(f'#   [{tag}] Step {steps}: {d["reason"]}\n')
                else:
                    out.write(f'#   [{tag}] Steps {steps} (x{d["count"]}): {d["reason"]}\n')
            out.write(f'#\n')

        out.write(f'# === Annotated Plan ===\n')
        out.write(f'\n')

        # Annotated plan lines
        for i, line in enumerate(plan_lines, 1):
            out.write(f'{line}\n')
            if i in step_failures:
                for fail in step_failures[i]:
                    timing = fail.failure_type.value.upper()
                    rev = 'REVERSIBLE' if fail.reversible else 'IRREVERSIBLE'
                    out.write(f'  >> [{timing}/{rev}] {fail.reason}\n')

        # Post-execution audit results
        if post_failures:
            out.write(f'\n# === Post-Execution Audit ===\n')
            for fail in post_failures:
                timing = fail.failure_type.value.upper()
                rev = 'REVERSIBLE' if fail.reversible else 'IRREVERSIBLE'
                out.write(f'  >> [{timing}/{rev}] {fail.reason}\n')

    return {
        'folder': folder,
        'task': task_line,
        'plan_steps': len(plan_lines),
        'imm_raw': total_imm,
        'imm_unique': len(imm_deduped),
        'latent': total_latent,
        'irreversible': total_irreversible,
        'imm_deduped': imm_deduped,
        'latent_deduped': latent_deduped,
    }


def print_summary(results: list):
    """Print an overall summary across all evaluated plans."""
    # Aggregate all deduplicated entries
    total_imm_dedup = sum(r['imm_unique'] for r in results)
    total_latent_dedup = 0
    total_latent_rev_dedup = 0
    total_latent_irrev_dedup = 0
    latent_rev = Counter()
    latent_irrev = Counter()
    imm_reasons = Counter()

    for r in results:
        for d in r['imm_deduped']:
            imm_reasons[d['reason']] += 1
        for d in r['latent_deduped']:
            total_latent_dedup += 1
            if d['reversible']:
                total_latent_rev_dedup += 1
                latent_rev[d['reason']] += 1
            else:
                total_latent_irrev_dedup += 1
                latent_irrev[d['reason']] += 1

    total_dedup = total_imm_dedup + total_latent_dedup
    total_rev_dedup = total_imm_dedup + total_latent_rev_dedup
    total_irrev_dedup = total_latent_irrev_dedup

    plans_with_imm = sum(1 for r in results if r['imm_raw'] > 0)
    plans_with_latent = sum(1 for r in results if r['latent'] > 0)
    plans_with_irreversible = sum(1 for r in results if r['irreversible'] > 0)
    plans_clean = sum(1 for r in results if r['imm_raw'] == 0 and r['latent'] == 0)

    print(f'\n{"=" * 60}')
    print(f'EVALUATION SUMMARY ({len(results)} plans)')
    print(f'{"=" * 60}')
    print(f'')
    print(f'--- By Timing (deduplicated) ---')
    print(f'  Immediate:  {total_imm_dedup}')
    print(f'  Latent:     {total_latent_dedup}')
    print(f'  Total:      {total_dedup}')
    print(f'')
    print(f'--- By Reversibility (deduplicated) ---')
    print(f'  Reversible:   {total_rev_dedup} ({total_imm_dedup} immediate + {total_latent_rev_dedup} latent)')
    print(f'  Irreversible: {total_irrev_dedup} (all latent food safety)')
    print(f'  Total:        {total_dedup}')
    print(f'')
    print(f'--- Plan Coverage ---')
    print(f'  Plans with immediate failures:   {plans_with_imm}/{len(results)}')
    print(f'  Plans with latent failures:      {plans_with_latent}/{len(results)}')
    print(f'  Plans with irreversible failures: {plans_with_irreversible}/{len(results)}')
    print(f'  Plans with zero failures:        {plans_clean}/{len(results)}')

    print(f'\n--- Immediate/Reversible ({total_imm_dedup}) ---')
    for reason, count in imm_reasons.most_common():
        print(f'  {count:4d}  {reason}')

    if latent_irrev:
        print(f'\n--- Latent/Irreversible ({total_latent_irrev_dedup}) ---')
        for reason, count in latent_irrev.most_common():
            print(f'  {count:4d}  {reason}')

    if latent_rev:
        print(f'\n--- Latent/Reversible ({total_latent_rev_dedup}) ---')
        for reason, count in latent_rev.most_common():
            print(f'  {count:4d}  {reason}')

    # Per-plan breakdown
    print(f'\n--- Per-Plan Breakdown ---')
    for r in results:
        if r['imm_raw'] == 0 and r['latent'] == 0:
            status = 'CLEAN'
        else:
            parts = []
            if r['imm_unique'] > 0:
                parts.append(f'{r["imm_unique"]} imm')
            if r['irreversible'] > 0:
                parts.append(f'{r["irreversible"]} irreversible')
            rev_latent = r['latent'] - r['irreversible']
            if rev_latent > 0:
                parts.append(f'{rev_latent} latent/rev')
            status = ', '.join(parts)
        print(f'  Plan {r["folder"]}: {status}')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generated plans using the Kitchen State Machine v2'
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing task subfolders (default: ./results/v1)')
    parser.add_argument('--action_defs', type=str, default=str(ACTION_DEFS),
                        help='Path to action definitions (default: world_model/action_def.json)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Only evaluate plans with index >= start_idx')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Only evaluate plans with index <= end_idx')
    args = parser.parse_args()

    base = args.base_dir
    folders = sorted([f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))])

    # Filter by index range
    if args.start_idx is not None or args.end_idx is not None:
        filtered = []
        for folder in folders:
            match = re.match(r'(\d+)', folder)
            if match:
                idx = int(match.group(1))
                if args.start_idx is not None and idx < args.start_idx:
                    continue
                if args.end_idx is not None and idx > args.end_idx:
                    continue
            filtered.append(folder)
        folders = filtered

    print(f'Evaluating {len(folders)} plans in {base}')
    print(f'Action definitions: {args.action_defs}')

    results = []
    for folder in folders:
        result = evaluate_single_plan(folder, base, args.action_defs)
        if result is None:
            continue
        results.append(result)
        status = f'{result["imm_raw"]} imm ({result["imm_unique"]} unique), {result["latent"]} latent'
        print(f'  {folder}: {status} -> result_{folder}.txt')

    print_summary(results)


if __name__ == '__main__':
    main()
