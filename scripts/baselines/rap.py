#!/usr/bin/env python3
"""
RAP (Reasoning via Planning) baseline: MCTS with LLM as world model.

Uses Monte Carlo Tree Search where:
- Action generation: LLM proposes candidate actions
- World model: State machine simulates effects
- Value estimation: LLM scores states for goal progress

Usage:
    python generate_plan_rap.py --model_family gpt --base_dir ./results/gpt5_rap
    python generate_plan_rap.py --model_family gpt --base_dir ./results/gpt5_rap --num_iterations 15
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from simmer.paths import ACTION_DEFS, RUNS_DIR
from simmer.llm_client import LLMClient
from simmer.state_machine import KitchenStateMachine


ACTION_DEFS_PATH = str(ACTION_DEFS)


@dataclass
class MCTSNode:
    state: Optional[object] = None  # KitchenStateMachine snapshot
    action: str = ""  # action that led to this node
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False


class RAPPlanner:
    def __init__(self, llm: LLMClient, env_path: str, task_goal: str,
                 env_content: str, num_iterations: int = 10,
                 num_candidates: int = 5, max_depth: int = 40,
                 exploration_weight: float = 1.4):
        self.llm = llm
        self.env_path = env_path
        self.task_goal = task_goal
        self.env_content = env_content
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.llm_calls = 0

    def _get_state_description(self, sm: KitchenStateMachine, actions_so_far: List[str]) -> str:
        holding = sm.agent['holding'] or 'nothing'
        location = sm.agent['location']
        recent = actions_so_far[-5:] if actions_so_far else []
        recent_str = '\n'.join(recent) if recent else '(none yet)'
        return f"Agent at: {location}, holding: {holding}\nRecent actions:\n{recent_str}"

    def _generate_candidates(self, sm: KitchenStateMachine,
                             actions_so_far: List[str]) -> List[str]:
        """Use LLM to propose candidate next actions."""
        state_desc = self._get_state_description(sm, actions_so_far)

        prompt = f"""You are a planning agent in a kitchen environment.

## Task Goal
{self.task_goal}

## Current State
{state_desc}

## Environment Specification
{self.env_content}

## Instructions
Propose exactly {self.num_candidates} different candidate next actions that could help achieve the goal.
Each action should be on its own line in the format: [ACTION] <class_name> (id)
For multi-argument actions: [ACTION] <class_name> (id) <class_name> (id)

Consider the current state carefully:
- If holding nothing, you might grab/take_out something
- If holding something, you might put it down, use it, or walk somewhere
- Progress toward the goal step by step

Output ONLY the {self.num_candidates} action lines, nothing else."""

        self.llm_calls += 1
        try:
            response = self.llm.generate_response(prompt)
        except Exception:
            return []

        candidates = []
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if re.search(r'\[\w+\]', line):
                candidates.append(line)

        return candidates[:self.num_candidates]

    def _evaluate_state(self, sm: KitchenStateMachine,
                        actions_so_far: List[str]) -> float:
        """Use LLM to score how close the state is to the goal (0-1)."""
        state_desc = self._get_state_description(sm, actions_so_far)
        num_failures = len(sm.failures)

        prompt = f"""You are evaluating progress toward a cooking task.

## Task Goal
{self.task_goal}

## Current State
{state_desc}

## Steps taken so far: {len(actions_so_far)}
## Execution errors so far: {num_failures}

Rate the progress toward completing the goal on a scale from 0.0 to 1.0:
- 0.0 = no progress at all
- 0.5 = roughly halfway done
- 1.0 = task is complete (SERVE action done)

Output ONLY a single number between 0.0 and 1.0."""

        self.llm_calls += 1
        try:
            response = self.llm.generate_response(prompt)
            # Extract float from response
            match = re.search(r'(0\.\d+|1\.0|0|1)', response.strip())
            if match:
                return float(match.group(1))
            return 0.1
        except Exception:
            return 0.1

    def _simulate_action(self, sm: KitchenStateMachine, action_line: str) -> Tuple[bool, KitchenStateMachine]:
        """Apply an action to a copy of the state machine. Returns (success, new_sm)."""
        new_sm = copy.deepcopy(sm)
        action_name, args = new_sm.parse_plan_line(f"1. {action_line}")
        if not action_name:
            return False, sm

        failures_before = len(new_sm.failures)
        new_sm.current_step += 1
        raw_args = ' '.join(f"<{n}> ({i})" if i else f"<{n}>" for n, i in args)
        new_sm.current_action = f"[{action_name.upper()}] {raw_args}"
        new_sm._execute_step(action_name, args)

        success = len(new_sm.failures) == failures_before
        return success, new_sm

    def _ucb1(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return float('inf')
        exploitation = node.value / node.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )
        return exploitation + exploration

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1."""
        while node.children and not node.is_terminal:
            node = max(node.children, key=self._ucb1)
        return node

    def _get_path_actions(self, node: MCTSNode) -> List[str]:
        """Get all actions from root to this node."""
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.action)
            current = current.parent
        actions.reverse()
        return actions

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by generating candidate actions."""
        actions_so_far = self._get_path_actions(node)

        if len(actions_so_far) >= self.max_depth:
            node.is_terminal = True
            return node

        candidates = self._generate_candidates(node.state, actions_so_far)
        if not candidates:
            node.is_terminal = True
            return node

        for action in candidates:
            success, new_sm = self._simulate_action(node.state, action)
            child = MCTSNode(
                state=new_sm,
                action=action,
                parent=node,
                is_terminal=bool(re.search(r'\[SERVE\]', action, re.IGNORECASE)),
            )
            node.children.append(child)

        # Return first unvisited child
        for child in node.children:
            if child.visits == 0:
                return child
        return node.children[0]

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def search(self) -> Tuple[List[str], str]:
        """Run MCTS and return (best_plan, tree_summary)."""
        # Initialize root
        root_sm = KitchenStateMachine(self.env_path, ACTION_DEFS_PATH)
        root = MCTSNode(state=root_sm)

        tree_lines = [f"RAP MCTS Search: {self.num_iterations} iterations, "
                      f"{self.num_candidates} candidates, max_depth={self.max_depth}\n"]

        for iteration in range(self.num_iterations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal and node.visits > 0:
                node = self._expand(node)
            elif node.visits == 0 and not node.children:
                # First visit to root — expand it
                if node == root:
                    node = self._expand(node)

            # Evaluation
            actions_so_far = self._get_path_actions(node)
            value = self._evaluate_state(node.state, actions_so_far)

            # Bonus for terminal (SERVE reached)
            if node.is_terminal and node.action:
                if re.search(r'\[SERVE\]', node.action, re.IGNORECASE):
                    value = 1.0

            # Backpropagation
            self._backpropagate(node, value)

            tree_lines.append(f"Iteration {iteration+1}: depth={len(actions_so_far)}, "
                              f"value={value:.2f}, action={node.action}")

        # Extract best path: greedily follow highest-value children
        best_actions = []
        current = root
        while current.children:
            best_child = max(current.children, key=lambda c: c.value / max(c.visits, 1))
            best_actions.append(best_child.action)
            current = best_child
            if current.is_terminal:
                break

        tree_lines.append(f"\nBest path: {len(best_actions)} actions")
        tree_lines.append(f"Total LLM calls: {self.llm_calls}")

        return best_actions, '\n'.join(tree_lines)


def load_task_goal(task_path: str) -> str:
    with open(task_path, 'r') as f:
        content = f.read()
    m = re.match(r'Task:\s*(.+)', content)
    return m.group(1).strip() if m else "Unknown task"


def process_single_task(folder: str, base_dir: str, llm: LLMClient,
                        num_iterations: int, num_candidates: int,
                        max_depth: int, exploration_weight: float) -> Optional[Dict]:
    env_path = os.path.join(base_dir, folder, f'env_{folder}.json')
    task_path = os.path.join(base_dir, folder, f'task_{folder}.txt')
    plan_path = os.path.join(base_dir, folder, f'plan_{folder}.txt')
    tree_path = os.path.join(base_dir, folder, f'tree_{folder}.txt')

    if not os.path.exists(env_path) or not os.path.exists(task_path):
        return None

    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        print(f"  Skipping {folder} (plan already exists)")
        return None

    task_goal = load_task_goal(task_path)
    print(f"  Task {folder}: {task_goal}")

    with open(env_path, 'r') as f:
        env_content = json.dumps(json.load(f), indent=2)

    planner = RAPPlanner(
        llm=llm,
        env_path=env_path,
        task_goal=task_goal,
        env_content=env_content,
        num_iterations=num_iterations,
        num_candidates=num_candidates,
        max_depth=max_depth,
        exploration_weight=exploration_weight,
    )

    try:
        actions, tree_summary = planner.search()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        with open(plan_path, 'w') as f:
            f.write(f"Error: {e}\n")
        return {'folder': folder, 'task': task_goal, 'steps': 0,
                'llm_calls': planner.llm_calls, 'status': 'error'}

    # Save plan
    with open(plan_path, 'w') as f:
        for i, a in enumerate(actions, 1):
            if not re.match(r'^\d+\.', a):
                f.write(f"{i}. {a}\n")
            else:
                f.write(f"{a}\n")

    # Save tree summary
    with open(tree_path, 'w') as f:
        f.write(f"Task: {task_goal}\n")
        f.write(tree_summary)

    print(f"  ✓ Saved: {plan_path} ({len(actions)} steps, {planner.llm_calls} LLM calls)")

    return {
        'folder': folder,
        'task': task_goal,
        'steps': len(actions),
        'llm_calls': planner.llm_calls,
        'status': 'success',
    }


def main():
    parser = argparse.ArgumentParser(description='RAP (MCTS) baseline for kitchen planning')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Run directory containing NNN/ task folders (create one with scripts/init_run.py)')
    parser.add_argument('--model_family', type=str, default='gpt',
                        choices=['gpt', 'gemini', 'llama', 'deepseek', 'claude', 'qwen'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--vllm_url', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=40)
    parser.add_argument('--exploration_weight', type=float, default=1.4)
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

    print(f"RAP baseline: {len(folders)} tasks in {base}")
    print(f"Model: {args.model_family}, iterations: {args.num_iterations}, "
          f"candidates: {args.num_candidates}, max_depth: {args.max_depth}")

    results = []
    total_llm_calls = 0
    for folder in folders:
        result = process_single_task(folder, base, llm, args.num_iterations,
                                     args.num_candidates, args.max_depth,
                                     args.exploration_weight)
        if result:
            results.append(result)
            total_llm_calls += result.get('llm_calls', 0)

    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary: {successful}/{len(results)} successful")
    print(f"Total LLM calls: {total_llm_calls}")
    if results:
        avg_calls = total_llm_calls / len(results)
        print(f"Average LLM calls per task: {avg_calls:.1f}")


if __name__ == '__main__':
    main()
