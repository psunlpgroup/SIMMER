#!/usr/bin/env python3
"""
Inner Monologue baseline: rich environment feedback after each action.

The model outputs only actions. The system provides detailed observations
including success/failure, agent state, object state changes, and scene
descriptions after each step.

Usage:
    python generate_plan_inner_monologue.py --model_family gpt --base_dir ./results/gpt5_inner_monologue
"""

import argparse
import json
import os
import re
import sys
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.llm_client import LLMClient
from simmer.state_machine import KitchenStateMachine


ACTION_DEFS_PATH = str(ACTION_DEFS)


def load_task_goal(task_path: str) -> str:
    with open(task_path, 'r') as f:
        content = f.read()
    m = re.match(r'Task:\s*(.+)', content)
    return m.group(1).strip() if m else "Unknown task"


def build_initial_prompt(env_content: str, task_goal: str) -> str:
    return f"""You are a planning agent in a simulated kitchen environment. You will accomplish a cooking task by issuing ONE action at a time. After each action, the environment will provide detailed feedback about what happened.

## Task Goal
{task_goal}

## Environment Specification
{env_content}

## Key Rules
- The agent starts in the kitchen holding nothing.
- To interact with an object, the agent must be at the object's location (use [WALK]).
- Containers (cabinet, fridge, pantry, drawer) must be opened before taking objects out.
- The agent can only hold one object at a time — put it down before grabbing another.
- When ingredients are combined in a container, use [COMBINE] <container> (id) <product_name> (1).
- The final action must be [SERVE] to complete the task.

## Action Format
Output ONLY your next action using this exact syntax:
- `[ACTION] <class_name> (id)` for single-argument actions
- `[ACTION] <class_name> (id) <class_name> (id)` for two-argument actions
- `[ACTION] <class_name> (id) <class_name> (id) <class_name> (id)` for three-argument actions
- For value arguments: `[WAIT] <5_minutes>`, `[PREHEAT] <oven> (1) <375F>`

Do NOT include any reasoning, explanations, or numbering. Output ONLY the action line.

Begin now. What is your first action?"""


def snapshot_states(sm: KitchenStateMachine) -> Dict[str, List[str]]:
    """Capture current states of all objects for diff."""
    snap = {}
    for oid, obj in {**sm.objects, **sm.created_objects}.items():
        snap[oid] = list(obj.get('states', []))
    return snap


def diff_states(before: Dict[str, List[str]], after: Dict[str, List[str]]) -> List[str]:
    """Return human-readable descriptions of state changes."""
    changes = []
    all_ids = set(before.keys()) | set(after.keys())
    for oid in sorted(all_ids):
        old = set(before.get(oid, []))
        new = set(after.get(oid, []))
        added = new - old
        removed = old - new
        name = oid.rsplit('_', 1)[0] if '_' in oid else oid
        if added:
            changes.append(f"  {oid} is now: {', '.join(sorted(added))}")
        if removed:
            changes.append(f"  {oid} no longer: {', '.join(sorted(removed))}")
    # New objects created
    for oid in sorted(set(after.keys()) - set(before.keys())):
        states = after[oid]
        changes.append(f"  {oid} was created (states: {', '.join(states)})")
    return changes


def describe_scene(sm: KitchenStateMachine) -> str:
    """Generate a brief scene description."""
    parts = []

    # What's on the counter
    counter_items = []
    for oid, obj in {**sm.objects, **sm.created_objects}.items():
        if obj.get('location') in ('counter', 'counter_1'):
            counter_items.append(oid)
    if counter_items:
        parts.append(f"On counter: {', '.join(counter_items[:8])}" +
                     (f" (+{len(counter_items)-8} more)" if len(counter_items) > 8 else ""))

    # Open containers
    open_containers = []
    for oid, obj in sm.objects.items():
        if 'open' in obj.get('states', []) and 'openable' in obj.get('properties', []):
            open_containers.append(oid)
    if open_containers:
        parts.append(f"Open: {', '.join(open_containers)}")

    # Appliances that are on
    on_appliances = []
    for oid, obj in sm.objects.items():
        if 'on' in obj.get('states', []) and 'switchable' in obj.get('properties', []):
            on_appliances.append(oid)
    if on_appliances:
        parts.append(f"Active: {', '.join(on_appliances)}")

    return '. '.join(parts) + '.' if parts else "Kitchen is in its initial state."


def build_observation(sm: KitchenStateMachine, action_line: str,
                      failures_before: int, states_before: Dict) -> str:
    """Execute one action and return a rich observation."""
    action_name, args = sm.parse_plan_line(action_line)
    if not action_name:
        return ("Environment: Could not parse your action. "
                "Please use the format: [ACTION] <object> (id). "
                "Output ONLY the action, nothing else.")

    sm.current_step += 1
    raw_args = ' '.join(f"<{n}> ({i})" if i else f"<{n}>" for n, i in args)
    sm.current_action = f"[{action_name.upper()}] {raw_args}"
    sm._execute_step(action_name, args)

    new_failures = sm.failures[failures_before:]
    holding = sm.agent['holding'] or 'nothing'
    location = sm.agent['location']

    states_after = snapshot_states(sm)
    changes = diff_states(states_before, states_after)
    scene = describe_scene(sm)

    obs_parts = []

    # 1. Success Detection
    if new_failures:
        reasons = '; '.join(f.reason for f in new_failures)
        obs_parts.append(f"Result: FAILED — {reasons}")
    else:
        obs_parts.append(f"Result: Success")

    # 2. Agent State
    obs_parts.append(f"Agent: at {location}, holding {holding}")

    # 3. State Changes
    if changes:
        obs_parts.append("Changes:\n" + '\n'.join(changes))

    # 4. Scene
    obs_parts.append(f"Scene: {scene}")

    return "Environment:\n" + '\n'.join(obs_parts)


def extract_action(response: str) -> Optional[str]:
    """Extract an action line from the model's response."""
    for line in response.strip().split('\n'):
        line = line.strip()
        # Remove leading numbering if present
        line = re.sub(r'^\d+\.\s*', '', line)
        if re.search(r'\[\w+\]', line):
            return line
    return None


def process_single_task(folder: str, base_dir: str, llm: LLMClient,
                        max_steps: int) -> Optional[Dict]:
    env_path = os.path.join(base_dir, folder, f'env_{folder}.json')
    task_path = os.path.join(base_dir, folder, f'task_{folder}.txt')
    plan_path = os.path.join(base_dir, folder, f'plan_{folder}.txt')
    trace_path = os.path.join(base_dir, folder, f'trace_{folder}.txt')

    if not os.path.exists(env_path) or not os.path.exists(task_path):
        return None

    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        print(f"  Skipping {folder} (plan already exists)")
        return None

    task_goal = load_task_goal(task_path)
    print(f"  Task {folder}: {task_goal}")

    with open(env_path, 'r') as f:
        env_content = json.dumps(json.load(f), indent=2)

    sm = KitchenStateMachine(env_path, ACTION_DEFS_PATH)
    conversation = build_initial_prompt(env_content, task_goal)

    actions = []
    trace_lines = []
    parse_failures = 0

    for step in range(1, max_steps + 1):
        try:
            response = llm.generate_response(conversation)
        except Exception as e:
            trace_lines.append(f"\n[ERROR at step {step}] {e}")
            break

        action_line = extract_action(response)
        trace_lines.append(f"\n--- Step {step} ---")
        trace_lines.append(f"Model output: {response.strip()}")

        if not action_line:
            parse_failures += 1
            feedback = ("Environment: Could not parse a valid action from your response. "
                        "Please output ONLY one action in the format: [ACTION] <object> (id)")
            trace_lines.append(feedback)
            conversation += f"\n{response}\n{feedback}\n"
            if parse_failures >= 5:
                trace_lines.append("\n[ABORT] Too many consecutive parse failures")
                break
            continue

        parse_failures = 0
        trace_lines.append(f"Action: {action_line}")

        failures_before = len(sm.failures)
        states_before = snapshot_states(sm)
        obs = build_observation(sm, action_line, failures_before, states_before)
        trace_lines.append(obs)

        # Only include successfully executed actions in the final plan
        new_failure_count = len(sm.failures) - failures_before
        if new_failure_count == 0:
            actions.append(action_line)

        conversation += f"\n{response}\n{obs}\n"

        if re.search(r'\[SERVE\]', action_line, re.IGNORECASE):
            break

    # Save plan
    with open(plan_path, 'w') as f:
        for i, a in enumerate(actions, 1):
            if not re.match(r'^\d+\.', a):
                f.write(f"{i}. {a}\n")
            else:
                f.write(f"{a}\n")
    print(f"  ✓ Saved: {plan_path} ({len(actions)} steps)")

    # Save trace
    with open(trace_path, 'w') as f:
        f.write(f"Task: {task_goal}\n")
        f.write(f"Total steps: {len(actions)}\n")
        f.write('\n'.join(trace_lines))
    print(f"  ✓ Saved: {trace_path}")

    return {
        'folder': folder,
        'task': task_goal,
        'steps': len(actions),
        'status': 'success',
    }


def main():
    parser = argparse.ArgumentParser(description='Inner Monologue baseline for kitchen planning')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Run directory containing NNN/ task folders (create one with scripts/init_run.py)')
    parser.add_argument('--model_family', type=str, default='gpt',
                        choices=['gpt', 'gemini', 'llama', 'deepseek', 'claude', 'qwen'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--vllm_url', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=80)
    args = parser.parse_args()

    llm = LLMClient(model_family=args.model_family, model=args.model,
                     base_url=args.vllm_url)

    base = args.base_dir
    folders = sorted([f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))])

    if args.start_idx is not None or args.end_idx is not None:
        filtered = []
        for folder in folders:
            m = re.match(r'(\d+)', folder)
            if m:
                idx = int(m.group(1))
                if args.start_idx is not None and idx < args.start_idx:
                    continue
                if args.end_idx is not None and idx > args.end_idx:
                    continue
            filtered.append(folder)
        folders = filtered

    print(f"Inner Monologue baseline: {len(folders)} tasks in {base}")
    print(f"Model: {args.model_family}, max_steps: {args.max_steps}")

    results = []
    for folder in folders:
        result = process_single_task(folder, base, llm, args.max_steps)
        if result:
            results.append(result)

    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary: {successful}/{len(results)} successful")


if __name__ == '__main__':
    main()
