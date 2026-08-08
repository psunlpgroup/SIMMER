#!/usr/bin/env python3
"""
Detect immediate+irreversible failures in generated plans.

These are cases where an irreversible action (cook, blend, combine, etc.)
has precondition failures BUT the state machine still applies the effects,
permanently producing wrong results. For example:
  - BAKE bread_dough when bread_dough is not in the oven
  - FRY fish when fish is not in the pan
  - BLEND almonds when almonds are not in the blender
  - COMBINE a container that's missing key ingredients

Usage:
    python detect_irreversible.py                          # all plans
    python detect_irreversible.py --start_idx 0 --end_idx 20
"""

import argparse
import json
import os
import re
import sys
import pathlib
from collections import defaultdict, Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.state_machine import KitchenStateMachine, FailureType


# Actions that permanently transform ingredients
COOKING_ACTIONS = {
    'fry', 'bake', 'roast', 'grill', 'toast', 'steam', 'blanch',
    'boil', 'simmer', 'heat', 'brew',
}
CUTTING_ACTIONS = {
    'cut', 'chop', 'slice', 'mince', 'dice',
}
MIXING_ACTIONS = {
    'mix', 'blend', 'whisk', 'beat', 'stir',
}
TRANSFORM_ACTIONS = {
    'break', 'peel', 'crush', 'mash', 'press', 'grate',
}
COMBINE_ACTIONS = {
    'combine',
}

ALL_IRREVERSIBLE = (
    COOKING_ACTIONS | CUTTING_ACTIONS | MIXING_ACTIONS |
    TRANSFORM_ACTIONS | COMBINE_ACTIONS
)


def detect_in_plan(folder, base_dir, action_defs_path):
    """Run a plan and detect immediate+irreversible failures."""
    env_path = f'{base_dir}/{folder}/env_{folder}.json'
    plan_path = f'{base_dir}/{folder}/plan_{folder}.txt'
    task_path = f'{base_dir}/{folder}/task_{folder}.txt'

    if not os.path.exists(env_path) or not os.path.exists(plan_path):
        return None

    task_line = ''
    if os.path.exists(task_path):
        with open(task_path) as f:
            first = f.readline().strip()
            if first.startswith('Task:'):
                task_line = first[5:].strip()

    sm = KitchenStateMachine(env_path, action_defs_path)
    plan = sm.parse_plan_file(plan_path)

    issues = []

    for i, (action_name, args) in enumerate(plan, 1):
        sm.current_step = i
        raw_args = ' '.join(f'<{n}> ({d})' for n, d in args)
        sm.current_action = f'[{action_name.upper()}] {raw_args}'

        if action_name not in ALL_IRREVERSIBLE:
            sm._execute_step(action_name, args)
            continue

        # Snapshot failure count before execution
        failures_before = len(sm.failures)

        # Execute the step
        sm._execute_step(action_name, args)

        # Check if new failures were recorded
        new_failures = sm.failures[failures_before:]
        precondition_failures = [
            f for f in new_failures
            if f.failure_type == FailureType.IMMEDIATE
            and 'Precondition failed' in f.reason
        ]

        if not precondition_failures:
            continue  # Action succeeded cleanly or had only syntax errors

        # This irreversible action had precondition failures but effects
        # were still applied. Detect what went wrong.
        bound = sm._bind_args(action_name, args)
        fail_reasons = [f.reason for f in precondition_failures]

        if action_name in COOKING_ACTIONS:
            # Check: did the object get "cooked" despite not being in the vessel?
            obj = bound.get('object')
            vessel = bound.get('receptacle') or bound.get('pot') or bound.get('appliance')
            obj_data = sm.get_object(obj) if obj else None
            vessel_data = sm.get_object(vessel) if vessel else None

            cooked_states = {'cooked', 'fried', 'baked', 'roasted', 'grilled',
                             'toasted', 'steamed', 'blanched', 'hot'}
            if obj_data and (cooked_states & set(obj_data.get('states', []))):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{obj} marked as cooked despite: {fail_reasons}',
                    'category': 'cooking_wrong_state',
                })
            elif vessel_data and ('boiling' in vessel_data.get('states', [])
                                  or 'hot' in vessel_data.get('states', [])):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{vessel} heated despite: {fail_reasons}',
                    'category': 'heating_wrong_state',
                })

        elif action_name in MIXING_ACTIONS:
            vessel = bound.get('receptacle') or bound.get('appliance')
            obj = bound.get('object')
            obj_data = sm.get_object(obj) if obj else None
            vessel_data = sm.get_object(vessel) if vessel else None

            mixed_states = {'mixed', 'whisked', 'stirred', 'beaten', 'blended', 'smooth'}
            if obj_data and (mixed_states & set(obj_data.get('states', []))):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{obj} marked as {action_name}ed despite: {fail_reasons}',
                    'category': 'mixing_wrong_state',
                })
            elif vessel_data and (mixed_states & set(vessel_data.get('states', []))):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{vessel} marked as {action_name}ed despite: {fail_reasons}',
                    'category': 'mixing_wrong_state',
                })

        elif action_name in COMBINE_ACTIONS:
            container = bound.get('container')
            product_id = bound.get('product_name')
            product = sm.get_object(product_id) if product_id else None
            if product:
                contents = sm.get_contents(container)
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{product_id} created from {container} '
                              f'({len(contents)} items) despite: {fail_reasons}',
                    'category': 'combine_wrong_state',
                })

        elif action_name in CUTTING_ACTIONS:
            obj = bound.get('object')
            obj_data = sm.get_object(obj) if obj else None
            cut_states = {'cut', 'chopped', 'sliced', 'minced', 'diced'}
            if obj_data and (cut_states & set(obj_data.get('states', []))):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{obj} marked as {action_name} despite: {fail_reasons}',
                    'category': 'cutting_wrong_state',
                })

        elif action_name in TRANSFORM_ACTIONS:
            obj = bound.get('object')
            obj_data = sm.get_object(obj) if obj else None
            transform_states = {'broken', 'peeled', 'crushed', 'mashed', 'pressed', 'grated'}
            if obj_data and (transform_states & set(obj_data.get('states', []))):
                issues.append({
                    'step': i,
                    'action': action_name,
                    'detail': f'{obj} marked as {action_name}ed despite: {fail_reasons}',
                    'category': 'transform_wrong_state',
                })

    return {
        'folder': folder,
        'task': task_line,
        'issues': issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Detect immediate+irreversible failures in cooking plans'
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Run directory containing NNN/ task folders (create one with scripts/init_run.py)')
    parser.add_argument('--action_defs', type=str, default=str(ACTION_DEFS))
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    base = args.base_dir
    folders = sorted([f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))])

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

    print(f'Scanning {len(folders)} plans for immediate+irreversible failures...\n')

    all_results = []
    category_counts = Counter()
    plans_with_issues = 0
    total_issues = 0

    for folder in folders:
        result = detect_in_plan(folder, base, args.action_defs)
        if result is None:
            continue
        all_results.append(result)
        if result['issues']:
            plans_with_issues += 1
            total_issues += len(result['issues'])
            print(f'Plan {folder} ({result["task"]}):')
            for issue in result['issues']:
                category_counts[issue['category']] += 1
                print(f'  Step {issue["step"]} [{issue["action"].upper()}] '
                      f'({issue["category"]}): {issue["detail"]}')
            print()

    print(f'{"=" * 60}')
    print(f'IMMEDIATE+IRREVERSIBLE FAILURE SUMMARY')
    print(f'{"=" * 60}')
    print(f'Plans scanned:      {len(all_results)}')
    print(f'Plans with issues:  {plans_with_issues}')
    print(f'Total issues:       {total_issues}')
    print()
    print(f'By category:')
    for cat, count in category_counts.most_common():
        print(f'  {count:4d}  {cat}')


if __name__ == '__main__':
    main()
