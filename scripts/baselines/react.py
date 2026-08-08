#!/usr/bin/env python3
"""
ReAct baseline: Reason + Act with step-by-step state machine feedback.

The model generates Thought + Action each turn, receives an Observation
from the state machine, and continues until [SERVE] or max steps.

Usage:
    python generate_plan_react.py --model_family gpt --base_dir ./results/gpt5_react
    python generate_plan_react.py --model_family deepseek --base_dir ./results/deepseek_react --max_steps 100
"""

import argparse
import json
import os
import re
import sys
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    return f"""You are a planning agent in a simulated kitchen environment. You will interact with the environment step by step using the ReAct framework: Thought, then Action. After each Action, you will receive an Observation from the environment.

## Task Goal
{task_goal}

## Environment Specification
{env_content}

## Key Rules
- The agent starts in the kitchen holding nothing.
- To interact with an object, the agent must be at the object's location (use [WALK]).
- Containers (cabinet, fridge, pantry, drawer) must be opened before taking objects out.
- The agent can only hold one object at a time — put it down before grabbing another.
- When ingredients are combined in a container, use [COMBINE] <container> (id) <product_name> (1) to declare the result.
- The final action must be [SERVE] to complete the task.

## Format
Each turn, output exactly:
Thought: <your reasoning about the current state and what to do next>
Action: [ACTION] <class_name> (id)

For multi-argument actions:
Action: [CUT] <carrot> (1) <cutting_board> (1) <knife> (1)

For value arguments (duration, temperature):
Action: [WAIT] <5_minutes>
Action: [PREHEAT] <oven> (1) <375F>

Output ONLY one Thought and one Action per turn. Wait for the Observation before continuing.

Begin planning now."""


def parse_action_from_response(response: str) -> Optional[str]:
    """Extract the Action line from the model's response."""
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith('Action:'):
            action_part = line[len('Action:'):].strip()
            return action_part
    # Fallback: look for any line with [ACTION] pattern
    for line in response.strip().split('\n'):
        if re.search(r'\[\w+\]', line.strip()):
            return line.strip()
    return None


def extract_thought(response: str) -> str:
    """Extract the Thought from the model's response."""
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith('Thought:'):
            return line[len('Thought:'):].strip()
    return ""


def build_observation(sm: KitchenStateMachine, action_line: str,
                      failures_before: int) -> str:
    """Execute one action and return an observation string."""
    action_name, args = sm.parse_plan_line(action_line)
    if not action_name:
        return "Observation: Could not parse your action. Please use the format: [ACTION] <object> (id)"

    sm.current_step += 1
    raw_args = ' '.join(f"<{n}> ({i})" if i else f"<{n}>" for n, i in args)
    sm.current_action = f"[{action_name.upper()}] {raw_args}"
    sm._execute_step(action_name, args)

    new_failures = sm.failures[failures_before:]
    holding = sm.agent['holding'] or 'nothing'
    location = sm.agent['location']

    if new_failures:
        reasons = '; '.join(f.reason for f in new_failures)
        obs = f"Observation: Action FAILED. Reason: {reasons}. "
    else:
        obs = "Observation: Action succeeded. "

    obs += f"Agent is at {location}, holding {holding}."
    return obs


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
    initial_prompt = build_initial_prompt(env_content, task_goal)

    history = []
    actions = []
    trace_lines = []

    # Build conversation as a single growing prompt
    conversation = initial_prompt

    for step in range(1, max_steps + 1):
        try:
            response = llm.generate_response(conversation)
        except Exception as e:
            trace_lines.append(f"\n[ERROR at step {step}] {e}")
            break

        thought = extract_thought(response)
        action_line = parse_action_from_response(response)

        trace_lines.append(f"\n--- Step {step} ---")
        trace_lines.append(f"Thought: {thought}")

        if not action_line:
            obs = "Observation: No valid action found in your response. Please output: Thought: ... then Action: [ACTION] <object> (id)"
            trace_lines.append(f"Action: (none)")
            trace_lines.append(obs)
            conversation += f"\n{response}\n{obs}\n"
            continue

        trace_lines.append(f"Action: {action_line}")

        failures_before = len(sm.failures)
        obs = build_observation(sm, action_line, failures_before)
        trace_lines.append(obs)

        # Only include successfully executed actions in the final plan
        new_failure_count = len(sm.failures) - failures_before
        if new_failure_count == 0:
            actions.append(action_line)

        conversation += f"\n{response}\n{obs}\n"

        # Check for SERVE action — task complete
        if re.search(r'\[SERVE\]', action_line, re.IGNORECASE):
            break

    # Save plan (numbered action lines)
    with open(plan_path, 'w') as f:
        for i, a in enumerate(actions, 1):
            # Ensure line has step number
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
    parser = argparse.ArgumentParser(description='ReAct baseline for kitchen planning')
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

    print(f"ReAct baseline: {len(folders)} tasks in {base}")
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
