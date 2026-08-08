#!/usr/bin/env python3
"""
ProgPrompt baseline: generate plans as Python-like programs.

The model generates a Python function using action primitives,
which is then parsed back into the standard plan format.

Usage:
    python generate_plan_progprompt.py --model_family gpt --base_dir ./results/gpt5_progprompt
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


ACTION_DEFS_PATH = str(ACTION_DEFS)


def load_task_goal(task_path: str) -> str:
    with open(task_path, 'r') as f:
        content = f.read()
    m = re.match(r'Task:\s*(.+)', content)
    return m.group(1).strip() if m else "Unknown task"


def build_action_signatures(env_data: Dict) -> str:
    """Generate Python function signatures from action definitions."""
    action_defs = env_data.get('action_definitions', {})
    signatures = []

    for action_name, action_data in sorted(action_defs.items()):
        if 'definition' in action_data:
            defn = action_data['definition']
        elif 'definitions' in action_data:
            defn = action_data['definitions'][0]['definition']
        else:
            continue

        args = defn.get('args', [])
        preconditions = defn.get('preconditions', [])
        effects = defn.get('effects', [])

        # Build parameter list
        params = []
        for arg in args:
            clean = arg.rstrip('?')
            optional = arg.endswith('?')
            if clean in ('duration', 'temperature', 'product_name'):
                params.append(f'{clean}=""')
            elif optional:
                params.append(f'{clean}_name=None, {clean}_id=None')
            else:
                params.append(f'{clean}_name, {clean}_id')

        param_str = ', '.join(params)

        # Build docstring
        preconds_str = '; '.join(preconditions[:3]) if preconditions else 'none'
        effects_str = '; '.join(effects[:3]) if effects else 'none'

        sig = f'def {action_name}({param_str}):\n'
        sig += f'    """Preconditions: {preconds_str}. Effects: {effects_str}."""\n'
        signatures.append(sig)

    return '\n'.join(signatures)


def build_objects_listing(env_data: Dict) -> str:
    """List objects grouped by location as Python-style declarations."""
    objects = env_data.get('initial_objects', [])
    by_location = {}
    for obj in objects:
        loc = obj.get('location', 'kitchen')
        by_location.setdefault(loc, []).append(obj)

    lines = []
    for loc in sorted(by_location.keys()):
        lines.append(f"# Location: {loc}")
        for obj in by_location[loc]:
            name = obj['class_name']
            oid = obj['id']
            props = ', '.join(obj.get('properties', []))
            states = ', '.join(obj.get('states', []))
            lines.append(f"{name}_{oid} = Object(name='{name}', id={oid}, properties=[{props}], states=[{states}])")
        lines.append("")

    return '\n'.join(lines)


def build_progprompt(env_content_str: str, env_data: Dict, task_goal: str) -> str:
    action_sigs = build_action_signatures(env_data)
    objects_listing = build_objects_listing(env_data)

    return f"""You are a planning agent in a simulated kitchen environment. Generate a Python function that accomplishes the given task using ONLY the action primitives and objects defined below.

## Task Goal
{task_goal}

## Available Action Primitives
```python
{action_sigs}
```

## Available Objects
```python
{objects_listing}
```

## Action Call Format
- For object arguments, pass the class_name and id separately: `walk("cabinet", 1)`
- For two-argument actions: `put_on("bowl", 1, "counter", 1)`
- For three-argument actions: `cut("carrot", 1, "cutting_board", 1, "knife", 1)`
- For value arguments (duration, temperature), pass as strings: `wait("5_minutes")`, `preheat("oven", 1, "375F")`
- For combine: `combine("pot", 1, "tea", 1)` — second pair is the product name and id

## Key Rules
- The agent starts in the kitchen holding nothing.
- Must walk() to an object's location before interacting with it.
- Containers (cabinet, fridge, pantry, drawer) must be open() before take_out().
- Agent can hold only one object — put_on() or put_in() before grab() another.
- Final action must be serve() to complete the task.

## Example
```python
def make_tea():
    # Boil water
    walk("kettle", 1)
    grab("kettle", 1)
    walk("sink", 1)
    switch_on("water", 1)
    fill("kettle", 1, "water", 1)
    switch_off("water", 1)
    walk("stove", 1)
    put_on("kettle", 1, "stove", 1)
    switch_on("stove", 1)
    boil("water", 1, "kettle", 1, "stove", 1)
    switch_off("stove", 1)
    # Brew tea
    walk("pantry", 1)
    open("pantry", 1)
    take_out("tea_bag", 1, "pantry", 1)
    walk("counter", 1)
    put_on("tea_bag", 1, "counter", 1)
    grab("cup", 1)
    walk("stove", 1)
    fill("cup", 1, "kettle", 1)
    walk("counter", 1)
    put_on("cup", 1, "counter", 1)
    grab("tea_bag", 1)
    put_in("tea_bag", 1, "cup", 1)
    wait("3_minutes")
    combine("cup", 1, "tea", 1)
    serve("tea", 1, "cup", 1)
```

## Your Task
Write a Python function that achieves: {task_goal}
Use ONLY the actions and objects listed above. Output ONLY the Python function, no explanations.

```python
"""

    return prompt.strip()


def extract_program(response: str) -> str:
    """Extract Python function from model response."""
    # Try to find code in markdown fences
    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Look for def statement
    lines = response.split('\n')
    in_function = False
    func_lines = []
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
        if in_function:
            func_lines.append(line)

    if func_lines:
        return '\n'.join(func_lines)

    return response.strip()


def parse_program_to_plan(program: str) -> List[str]:
    """Parse Python function calls into standard plan format."""
    plan_lines = []
    step = 0

    for line in program.split('\n'):
        line = line.strip()
        # Skip comments, empty lines, def, return, pass
        if not line or line.startswith('#') or line.startswith('def ') or \
           line.startswith('return') or line == 'pass':
            continue

        # Match function call: action_name(args...)
        m = re.match(r'(\w+)\((.*)\)\s*$', line)
        if not m:
            continue

        action_name = m.group(1)
        args_str = m.group(2)

        # Parse arguments
        args = parse_arguments(args_str)
        if args is None:
            continue

        # Convert to plan format
        plan_line = convert_to_plan_format(action_name, args)
        if plan_line:
            step += 1
            plan_lines.append(f"{step}. {plan_line}")

    return plan_lines


def parse_arguments(args_str: str) -> Optional[List[str]]:
    """Parse comma-separated arguments respecting quotes."""
    args = []
    current = ''
    depth = 0
    in_quote = False
    quote_char = None

    for ch in args_str:
        if ch in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = ch
            current += ch
        elif ch == quote_char and in_quote:
            in_quote = False
            current += ch
            quote_char = None
        elif ch == '(' and not in_quote:
            depth += 1
            current += ch
        elif ch == ')' and not in_quote:
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0 and not in_quote:
            args.append(current.strip())
            current = ''
        else:
            current += ch

    if current.strip():
        args.append(current.strip())

    return args


def convert_to_plan_format(action_name: str, args: List[str]) -> Optional[str]:
    """Convert action_name + parsed args into [ACTION] <obj> (id) format."""
    action_upper = action_name.upper()
    plan_parts = [f"[{action_upper}]"]

    i = 0
    while i < len(args):
        arg = args[i].strip().strip('"').strip("'")

        # Check if this is a value argument (quoted string that's not followed by an id)
        is_value = False
        if args[i].strip().startswith('"') or args[i].strip().startswith("'"):
            # It's a quoted string — could be object name or value
            # If next arg is a number, treat current as object name
            if i + 1 < len(args) and re.match(r'^\d+$', args[i + 1].strip()):
                # Object name + id pair
                obj_name = arg
                obj_id = args[i + 1].strip()
                plan_parts.append(f"<{obj_name}> ({obj_id})")
                i += 2
                continue
            else:
                # Standalone value (duration, temperature, product_name)
                is_value = True

        if is_value:
            plan_parts.append(f"<{arg}>")
            i += 1
        elif re.match(r'^\d+$', arg):
            # Bare number — shouldn't happen at start, skip
            i += 1
        else:
            # Object name — check if next is id
            if i + 1 < len(args) and re.match(r'^\d+$', args[i + 1].strip()):
                obj_name = arg
                obj_id = args[i + 1].strip()
                plan_parts.append(f"<{obj_name}> ({obj_id})")
                i += 2
            else:
                # Bare name without id (value argument)
                plan_parts.append(f"<{arg}>")
                i += 1

    return ' '.join(plan_parts)


def process_single_task(folder: str, base_dir: str, llm: LLMClient) -> Optional[Dict]:
    env_path = os.path.join(base_dir, folder, f'env_{folder}.json')
    task_path = os.path.join(base_dir, folder, f'task_{folder}.txt')
    plan_path = os.path.join(base_dir, folder, f'plan_{folder}.txt')
    program_path = os.path.join(base_dir, folder, f'program_{folder}.txt')

    if not os.path.exists(env_path) or not os.path.exists(task_path):
        return None

    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        print(f"  Skipping {folder} (plan already exists)")
        return None

    task_goal = load_task_goal(task_path)
    print(f"  Task {folder}: {task_goal}")

    with open(env_path, 'r') as f:
        env_data = json.load(f)
    env_content_str = json.dumps(env_data, indent=2)

    prompt = build_progprompt(env_content_str, env_data, task_goal)

    try:
        response = llm.generate_response(prompt)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {'folder': folder, 'task': task_goal, 'steps': 0, 'status': 'error'}

    program = extract_program(response)
    plan_lines = parse_program_to_plan(program)

    # Save program
    with open(program_path, 'w') as f:
        f.write(program)

    # Save plan
    with open(plan_path, 'w') as f:
        f.write('\n'.join(plan_lines))
    print(f"  ✓ Saved: {plan_path} ({len(plan_lines)} steps)")

    return {
        'folder': folder,
        'task': task_goal,
        'steps': len(plan_lines),
        'status': 'success',
    }


def main():
    parser = argparse.ArgumentParser(description='ProgPrompt baseline for kitchen planning')
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

    print(f"ProgPrompt baseline: {len(folders)} tasks in {base}")
    print(f"Model: {args.model_family}")

    results = []
    for folder in folders:
        result = process_single_task(folder, base, llm)
        if result:
            results.append(result)

    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary: {successful}/{len(results)} successful")


if __name__ == '__main__':
    main()
