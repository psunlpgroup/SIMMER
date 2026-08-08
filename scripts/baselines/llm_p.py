#!/usr/bin/env python3
"""
LLM+P baseline: LLM translates to PDDL, classical planner solves, LLM translates back.

Pipeline:
1. LLM generates PDDL domain from action definitions
2. LLM generates PDDL problem from initial state + goal
3. Classical planner (pyperplan) finds a plan
4. LLM translates PDDL plan back to standard format

Usage:
    python generate_plan_llm_p.py --model_family gpt --base_dir ./results/gpt5_llm_p
"""

import argparse
import json
import os
import re
import subprocess
import sys
import pathlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.llm_client import LLMClient


ACTION_DEFS_PATH = str(ACTION_DEFS)


def load_task_goal(task_path: str) -> str:
    with open(task_path, 'r') as f:
        content = f.read()
    m = re.match(r'Task:\s*(.+)', content)
    return m.group(1).strip() if m else "Unknown task"


def generate_pddl_domain(llm: LLMClient, env_data: Dict, task_goal: str) -> str:
    """Use LLM to translate action definitions to PDDL domain."""
    action_defs = env_data.get('action_definitions', {})
    actions_desc = json.dumps(action_defs, indent=2)

    # Collect all properties and states for predicate generation
    all_props = set()
    all_states = set()
    for obj in env_data.get('initial_objects', []):
        all_props.update(obj.get('properties', []))
        all_states.update(obj.get('states', []))

    prompt = f"""Convert the following kitchen environment action definitions into a PDDL domain file.

## Task Context
{task_goal}

## Known Properties (use as predicates)
{sorted(all_props)}

## Known States (use as predicates)
{sorted(all_states)}

## Action Definitions (JSON)
{actions_desc}

## Instructions
Generate a valid PDDL domain file with:
1. Domain name: kitchen
2. Requirements: :strips :typing
3. Types: object
4. Predicates: one for each property, state, and spatial relationship (at, holding, in, on, empty-hands, etc.)
5. Actions: one for each action definition, with:
   - :parameters matching the action's args
   - :precondition from the action's preconditions list
   - :effect from the action's effects list

Important:
- Use simple predicate names (lowercase, hyphens): (at ?agent ?location), (holding ?agent ?obj), (is-open ?obj), etc.
- Every precondition/effect must map to a declared predicate
- Use (not ...) for negative effects
- Keep it simple — focus on the most important preconditions and effects

Output ONLY the PDDL domain file, starting with (define (domain kitchen) ...). No explanations."""

    return llm.generate_response(prompt)


def generate_pddl_problem(llm: LLMClient, env_data: Dict, task_goal: str,
                           domain_pddl: str) -> str:
    """Use LLM to translate initial state + goal to PDDL problem."""
    objects = env_data.get('initial_objects', [])
    objects_desc = json.dumps(objects, indent=2)

    prompt = f"""Given the following PDDL domain and kitchen environment, generate a PDDL problem file.

## PDDL Domain
{domain_pddl}

## Task Goal
{task_goal}

## Initial Objects and States (JSON)
{objects_desc}

## Agent Initial State
- Agent is at: kitchen
- Agent is holding: nothing (empty hands)

## Instructions
Generate a valid PDDL problem file with:
1. Problem name: kitchen-task
2. Domain: kitchen (matching the domain above)
3. Objects: list all objects from the JSON using the types from the domain
   - Use format: obj_name_id - object (e.g., cabinet_1 - object, bowl_1 - object)
4. Init: translate all object states, properties, and locations into predicates
   - Include (at agent kitchen) and (empty-hands agent) for the agent
5. Goal: translate the task goal into PDDL predicates
   - The goal should represent the final desired state (e.g., something is served)

Use ONLY predicates that are declared in the domain file above.
Output ONLY the PDDL problem file, starting with (define (problem kitchen-task) ...). No explanations."""

    return llm.generate_response(prompt)


def run_planner(domain_path: str, problem_path: str) -> Optional[str]:
    """Run pyperplan and return the solution plan, or None if failed."""
    try:
        result = subprocess.run(
            ['pyperplan', domain_path, problem_path],
            capture_output=True, text=True, timeout=60
        )

        # pyperplan writes solution to problem_file.soln
        soln_path = problem_path + '.soln'
        if os.path.exists(soln_path):
            with open(soln_path, 'r') as f:
                plan = f.read()
            os.remove(soln_path)
            return plan

        # Check stderr for errors
        if result.returncode != 0:
            return None

        return None
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # Try as Python module
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pyperplan', domain_path, problem_path],
                capture_output=True, text=True, timeout=60
            )
            soln_path = problem_path + '.soln'
            if os.path.exists(soln_path):
                with open(soln_path, 'r') as f:
                    plan = f.read()
                os.remove(soln_path)
                return plan
            return None
        except Exception:
            return None


def translate_plan_back(llm: LLMClient, pddl_plan: str, env_data: Dict,
                         task_goal: str) -> str:
    """Use LLM to translate PDDL plan back to standard format."""
    objects = env_data.get('initial_objects', [])
    objects_summary = []
    for obj in objects:
        objects_summary.append(f"  {obj['class_name']} (id={obj['id']}), location={obj.get('location', 'kitchen')}")
    objects_str = '\n'.join(objects_summary)

    prompt = f"""Translate the following PDDL plan back into kitchen environment action format.

## Task Goal
{task_goal}

## Available Objects
{objects_str}

## PDDL Plan
{pddl_plan}

## Output Format
Convert each PDDL action into the format:
- [ACTION] <class_name> (id) for single-argument actions
- [ACTION] <class_name> (id) <class_name> (id) for two-argument actions
- [ACTION] <class_name> (id) <class_name> (id) <class_name> (id) for three-argument actions

Rules:
- Map PDDL action names to uppercase: (walk cabinet_1) → [WALK] <cabinet> (1)
- Split object IDs: cabinet_1 → <cabinet> (1), bowl_2 → <bowl> (2)
- Preserve action order
- Number each step

Output ONLY the numbered plan. No explanations."""

    return llm.generate_response(prompt)


def extract_pddl(response: str) -> str:
    """Extract PDDL content from LLM response."""
    # Try markdown code fence
    fence_match = re.search(r'```(?:pddl|lisp)?\s*\n(.*?)```', response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Look for (define ...)
    define_match = re.search(r'(\(define\s.*)', response, re.DOTALL)
    if define_match:
        text = define_match.group(1)
        # Balance parentheses
        depth = 0
        for i, ch in enumerate(text):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return text[:i+1]
        return text

    return response.strip()


def process_single_task(folder: str, base_dir: str, llm: LLMClient) -> Optional[Dict]:
    env_path = os.path.join(base_dir, folder, f'env_{folder}.json')
    task_path = os.path.join(base_dir, folder, f'task_{folder}.txt')
    plan_path = os.path.join(base_dir, folder, f'plan_{folder}.txt')
    domain_path = os.path.join(base_dir, folder, f'domain_{folder}.pddl')
    problem_path = os.path.join(base_dir, folder, f'problem_{folder}.pddl')
    pddl_plan_path = os.path.join(base_dir, folder, f'pddl_plan_{folder}.txt')

    if not os.path.exists(env_path) or not os.path.exists(task_path):
        return None

    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        print(f"  Skipping {folder} (plan already exists)")
        return None

    task_goal = load_task_goal(task_path)
    print(f"  Task {folder}: {task_goal}")

    with open(env_path, 'r') as f:
        env_data = json.load(f)

    llm_calls = 0

    # Step 1: Generate PDDL domain
    print(f"    Step 1: Generating PDDL domain...")
    try:
        domain_response = generate_pddl_domain(llm, env_data, task_goal)
        llm_calls += 1
        domain_pddl = extract_pddl(domain_response)
        with open(domain_path, 'w') as f:
            f.write(domain_pddl)
    except Exception as e:
        print(f"  ✗ Domain generation failed: {e}")
        with open(plan_path, 'w') as f:
            f.write(f"Error: Domain generation failed: {e}\n")
        return {'folder': folder, 'task': task_goal, 'steps': 0,
                'llm_calls': llm_calls, 'status': 'error', 'error_stage': 'domain'}

    # Step 2: Generate PDDL problem
    print(f"    Step 2: Generating PDDL problem...")
    try:
        problem_response = generate_pddl_problem(llm, env_data, task_goal, domain_pddl)
        llm_calls += 1
        problem_pddl = extract_pddl(problem_response)
        with open(problem_path, 'w') as f:
            f.write(problem_pddl)
    except Exception as e:
        print(f"  ✗ Problem generation failed: {e}")
        with open(plan_path, 'w') as f:
            f.write(f"Error: Problem generation failed: {e}\n")
        return {'folder': folder, 'task': task_goal, 'steps': 0,
                'llm_calls': llm_calls, 'status': 'error', 'error_stage': 'problem'}

    # Step 3: Run classical planner
    print(f"    Step 3: Running planner...")
    pddl_plan = run_planner(domain_path, problem_path)

    if pddl_plan is None:
        print(f"  ✗ Planner failed to find a solution")
        with open(plan_path, 'w') as f:
            f.write("Error: Planner failed to find a solution\n")
        return {'folder': folder, 'task': task_goal, 'steps': 0,
                'llm_calls': llm_calls, 'status': 'error', 'error_stage': 'planner'}

    with open(pddl_plan_path, 'w') as f:
        f.write(pddl_plan)

    # Step 4: Translate plan back
    print(f"    Step 4: Translating plan back...")
    try:
        plan_response = translate_plan_back(llm, pddl_plan, env_data, task_goal)
        llm_calls += 1
    except Exception as e:
        print(f"  ✗ Plan translation failed: {e}")
        with open(plan_path, 'w') as f:
            f.write(f"Error: Plan translation failed: {e}\n")
        return {'folder': folder, 'task': task_goal, 'steps': 0,
                'llm_calls': llm_calls, 'status': 'error', 'error_stage': 'translation'}

    # Extract plan lines
    plan_lines = []
    for line in plan_response.strip().split('\n'):
        line = line.strip()
        if re.search(r'\[\w+\]', line):
            if not re.match(r'^\d+\.', line):
                plan_lines.append(f"{len(plan_lines)+1}. {line}")
            else:
                plan_lines.append(line)

    with open(plan_path, 'w') as f:
        f.write('\n'.join(plan_lines))

    print(f"  ✓ Saved: {plan_path} ({len(plan_lines)} steps, {llm_calls} LLM calls)")

    return {
        'folder': folder,
        'task': task_goal,
        'steps': len(plan_lines),
        'llm_calls': llm_calls,
        'status': 'success',
    }


def main():
    parser = argparse.ArgumentParser(description='LLM+P baseline for kitchen planning')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Run directory containing NNN/ task folders (create one with scripts/init_run.py)')
    parser.add_argument('--model_family', type=str, default='gpt',
                        choices=['gpt', 'gemini', 'llama', 'deepseek', 'claude', 'qwen'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--vllm_url', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
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

    print(f"LLM+P baseline: {len(folders)} tasks in {base}")
    print(f"Model: {args.model_family}")

    results = []
    total_llm_calls = 0
    for folder in folders:
        result = process_single_task(folder, base, llm)
        if result:
            results.append(result)
            total_llm_calls += result.get('llm_calls', 0)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed_stages = {}
    for r in results:
        if r['status'] == 'error':
            stage = r.get('error_stage', 'unknown')
            failed_stages[stage] = failed_stages.get(stage, 0) + 1

    print(f"\nSummary: {successful}/{len(results)} successful")
    print(f"Total LLM calls: {total_llm_calls}")
    if failed_stages:
        print(f"Failures by stage: {failed_stages}")


if __name__ == '__main__':
    main()
