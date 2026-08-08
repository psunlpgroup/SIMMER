import os
import sys
import json
import pathlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simmer.llm_client import LLMClient


class EnvFileProcessor:
    """
    Process environment JSON files to generate plans using LLM (supports multiple model families).
    """

    def __init__(self, api_key: Optional[str] = None, model_family: str = "gpt", model: str = None,
                 base_dir: str = None, start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None, base_url: Optional[str] = None,
                 use_foresight: bool = False, use_self_refine: bool = False,
                 foresight_group: Optional[str] = None):
        """
        Initialize the file processor.

        Args:
            api_key (str, optional): API key for the selected model family
            model_family (str): Model family to use. Options: "gpt", "gemini", "llama", "deepseek", "claude".
            model (str, optional): Specific model name. If None, uses default for the family.
            base_dir (str): Base directory containing task subfolders (e.g., ./results/v1/000/, ./results/v1/001/)
            start_idx (int, optional): Only process tasks with index >= start_idx
            end_idx (int, optional): Only process tasks with index <= end_idx
            base_url (str, optional): vLLM server URL for llama/deepseek
            use_foresight (bool): Use counterfactual foresight simulation prompt
            use_self_refine (bool): Use self-refine (generate then critique and revise)
            foresight_group (str, optional): Ablation group: "reasoning" (state/precondition/effect checks) or "safety" (safety/appliance checks)
        """
        self.llm_client = LLMClient(model_family=model_family, api_key=api_key, model=model, base_url=base_url)
        self.model_family = model_family
        self.use_foresight = use_foresight
        self.use_self_refine = use_self_refine
        self.foresight_group = foresight_group
        self.results = []
        self.base_dir = base_dir
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.task_goals = self._load_task_goals()

    def _load_task_goals(self) -> Dict[int, str]:
        """
        Load task goals by scanning task_*.txt files in base_dir subfolders.

        Returns:
            Dict[int, str]: Mapping of task number to goal
        """
        import re

        task_goals = {}
        try:
            base_path = Path(self.base_dir)
            if not base_path.exists():
                print(f"Warning: Base directory {self.base_dir} does not exist")
                return {}

            # Find all task_*.txt files in subfolders (e.g., 000/task_000.txt)
            task_files = sorted(base_path.glob("*/task_*.txt"))

            for task_file in task_files:
                # Extract task number from filename
                match = re.search(r'task_(\d+)\.txt', task_file.name)
                if not match:
                    continue

                task_num = int(match.group(1))

                # Filter by start_idx / end_idx if specified
                if self.start_idx is not None and task_num < self.start_idx:
                    continue
                if self.end_idx is not None and task_num > self.end_idx:
                    continue

                # Read file and extract goal
                with open(task_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract goal from "Task: How to make XXX" line
                task_match = re.match(r'Task:\s*(.+)', content)
                if task_match:
                    goal = task_match.group(1).strip()
                    task_goals[task_num] = goal

            print(f"Loaded {len(task_goals)} task goals from {self.base_dir}" +
                  (f" (end_idx={self.end_idx})" if self.end_idx is not None else ""))
            return task_goals

        except Exception as e:
            print(f"Warning: Could not load task goals from {self.base_dir}: {e}")
            return {}

    def get_json_files(self) -> List[Path]:
        """
        Retrieve all env_*.json files from subfolders in base_dir.

        Returns:
            List[Path]: List of Path objects for json files, sorted by name
        """
        base_path = Path(self.base_dir)
        if not base_path.exists():
            raise ValueError(f"Directory {self.base_dir} does not exist")

        json_files = sorted(base_path.glob("*/env_*.json"))
        # Filter out empty files and error files
        json_files = [f for f in json_files if f.stat().st_size > 0 and '_error' not in f.name]

        # Filter by start_idx / end_idx if specified
        if self.start_idx is not None or self.end_idx is not None:
            import re
            filtered = []
            for f in json_files:
                match = re.search(r'env_(\d+)', f.name)
                if match:
                    idx = int(match.group(1))
                    if self.start_idx is not None and idx < self.start_idx:
                        continue
                    if self.end_idx is not None and idx > self.end_idx:
                        continue
                    filtered.append(f)
            json_files = filtered

        return json_files

    def read_file_content(self, file_path: Path) -> str:
        """
        Read the content of a JSON file and format it as a string.

        Args:
            file_path (Path): Path to the file

        Returns:
            str: JSON content formatted as a string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                # Return formatted JSON string
                return json.dumps(json_content, indent=2)
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def create_prompt(self, file_content: str, task_goal: str) -> str:
        if self.use_foresight:
            if self.foresight_group == "reasoning":
                return self._create_prompt_foresight_reasoning(file_content, task_goal)
            elif self.foresight_group == "safety":
                return self._create_prompt_foresight_safety(file_content, task_goal)
            return self._create_prompt_foresight(file_content, task_goal)
        return self._create_prompt_base(file_content, task_goal)

    def _create_prompt_base(self, file_content: str, task_goal: str) -> str:
        """Base prompt without counterfactual foresight simulation."""

        prompt = f"""You are a planning agent in a simulated kitchen environment. Your task is to generate a step-by-step action plan to achieve a given goal using ONLY the objects and actions available in the environment specification below.

## Task Goal
{task_goal}

## Environment Specification
{file_content}

## Instructions

1. **Understand the environment**: Study the objects (their IDs, properties, states, and locations), the available actions (their arguments, preconditions, and effects), and the agent's starting state.

2. **Track state carefully**: The agent starts in the kitchen holding nothing. To interact with an object, the agent must first be at the object's location. Containers (cabinet, fridge, pantry, drawer) must be opened before objects inside them can be taken out. The agent can only hold one object at a time — after grabbing or taking out an object, the agent must put it down before grabbing another.

3. **Output format**: Write a numbered list of actions using this exact syntax:
   - `[ACTION] <class_name> (id)` for single-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id)` for two-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id) <class_name> (id)` for three-argument actions
   - Use the object IDs from the environment specification. When multiple instances of an object exist (e.g., bowl id=1 and bowl id=2), pick the appropriate one and be consistent.
   - `duration`, `temperature`, and `timer` arguments are values, not objects. Write them without an ID, e.g., `[WAIT] <5_minutes>`, `[SET_TEMP] <oven> (1) <350F>`, `[PREHEAT] <oven> (1) <375F>`.

4. **Example plan** (for illustration only — not related to your task):
```
1. [WALK] <cabinet> (1)
2. [OPEN] <cabinet> (1)
3. [GRAB] <bowl> (1)
4. [WALK] <sink> (1)
5. [SWITCH_ON] <water> (1)
6. [FILL] <bowl> (1) <water> (1)
7. [SWITCH_OFF] <water> (1)
8. [PUT_ON] <bowl> (1) <counter> (1)
```

5. **Creating new objects with COMBINE**: When ingredients in a container are transformed into a new product (through mixing, cooking, etc.), use `[COMBINE] <container> (id) <product_name> (1)` to declare the result. Give the product a descriptive name. The new object can then be referenced in subsequent steps. For example, after boiling tea leaves in a pot with water: `[COMBINE] <pot> (1) <tea> (1)`, then `[POUR] <pot> (1) <cup> (1)` and `[SERVE] <tea> (1) <cup> (1)`.

6. **Task completion**: The final step of your plan must always be a `[SERVE]` action to indicate the task is complete and present the finished item.

7. **Constraints**:
   - Use ONLY actions defined in the action_definitions section
   - Ensure all preconditions for each action are satisfied by prior steps
   - Every action and object must come from the environment specification
   - The plan should logically achieve the task goal

## Plan
"""

        return prompt.strip()

    def _create_prompt_foresight(self, file_content: str, task_goal: str) -> str:
        """Prompt with counterfactual foresight simulation."""

        prompt = f"""You are a planning agent in a simulated kitchen environment. Your task is to generate a step-by-step action plan to achieve a given goal using ONLY the objects and actions available in the environment specification below.

## Task Goal
{task_goal}

## Environment Specification
{file_content}

## Instructions

1. **Understand the environment**: Study the objects (their IDs, properties, states, and locations), the available actions (their arguments, preconditions, and effects), and the agent's starting state.

2. **Track state carefully**: The agent starts in the kitchen holding nothing. To interact with an object, the agent must first be at the object's location. Containers (cabinet, fridge, pantry, drawer) must be opened before objects inside them can be taken out. The agent can only hold one object at a time — after grabbing or taking out an object, the agent must put it down before grabbing another.

3. **Counterfactual Foresight Simulation**: Before committing each step, mentally simulate what would happen:
   - **State check**: What is the agent currently holding? Where is the agent? What objects are in which containers? Which containers are open/closed? Which appliances are on/off?
   - **Precondition check**: Does this action's preconditions hold given the current state? For example, are your hands empty before grabbing? Is the container open before taking out? Is the object on the cutting board before cutting?
   - **Effect check**: After this action, what changes? Will it leave the agent in a state that allows the next intended action? Will it cause any unintended consequences?
   - **Safety check**: Does this action create any food safety hazard? For example, handling raw meat and then touching ready-to-eat food without washing hands? Using the same cutting board for raw protein and vegetables without cleaning?
   - **Appliance check**: At the end of the plan, are all heat sources (stove, oven, grill) turned off? Is the water turned off?
   - If any check fails, revise the step before committing it.

4. **Output format**: Write a numbered list of actions using this exact syntax:
   - `[ACTION] <class_name> (id)` for single-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id)` for two-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id) <class_name> (id)` for three-argument actions
   - Use the object IDs from the environment specification. When multiple instances of an object exist (e.g., bowl id=1 and bowl id=2), pick the appropriate one and be consistent.
   - `duration`, `temperature`, and `timer` arguments are values, not objects. Write them without an ID, e.g., `[WAIT] <5_minutes>`, `[SET_TEMP] <oven> (1) <350F>`, `[PREHEAT] <oven> (1) <375F>`.

5. **Example plan** (for illustration only — not related to your task):
```
1. [WALK] <cabinet> (1)
2. [OPEN] <cabinet> (1)
3. [GRAB] <bowl> (1)
4. [WALK] <sink> (1)
5. [SWITCH_ON] <water> (1)
6. [FILL] <bowl> (1) <water> (1)
7. [SWITCH_OFF] <water> (1)
8. [PUT_ON] <bowl> (1) <counter> (1)
```

6. **Creating new objects with COMBINE**: When ingredients in a container are transformed into a new product (through mixing, cooking, etc.), use `[COMBINE] <container> (id) <product_name> (1)` to declare the result. Give the product a descriptive name. The new object can then be referenced in subsequent steps. For example, after boiling tea leaves in a pot with water: `[COMBINE] <pot> (1) <tea> (1)`, then `[POUR] <pot> (1) <cup> (1)` and `[SERVE] <tea> (1) <cup> (1)`.

7. **Task completion**: The final step of your plan must always be a `[SERVE]` action to indicate the task is complete and present the finished item.

8. **Constraints**:
   - Use ONLY actions defined in the action_definitions section
   - Ensure all preconditions for each action are satisfied by prior steps
   - Every action and object must come from the environment specification
   - The plan should logically achieve the task goal

Output ONLY the numbered plan. Do not include any reasoning, simulation traces, or explanations.

## Plan
"""

        return prompt.strip()

    def _create_prompt_foresight_reasoning(self, file_content: str, task_goal: str) -> str:
        """Ablation: only state/precondition/effect checks (no safety/appliance checks)."""

        prompt = f"""You are a planning agent in a simulated kitchen environment. Your task is to generate a step-by-step action plan to achieve a given goal using ONLY the objects and actions available in the environment specification below.

## Task Goal
{task_goal}

## Environment Specification
{file_content}

## Instructions

1. **Understand the environment**: Study the objects (their IDs, properties, states, and locations), the available actions (their arguments, preconditions, and effects), and the agent's starting state.

2. **Track state carefully**: The agent starts in the kitchen holding nothing. To interact with an object, the agent must first be at the object's location. Containers (cabinet, fridge, pantry, drawer) must be opened before objects inside them can be taken out. The agent can only hold one object at a time — after grabbing or taking out an object, the agent must put it down before grabbing another.

3. **Counterfactual Foresight Simulation**: Before committing each step, mentally simulate what would happen:
   - **State check**: What is the agent currently holding? Where is the agent? What objects are in which containers? Which containers are open/closed? Which appliances are on/off?
   - **Precondition check**: Does this action's preconditions hold given the current state? For example, are your hands empty before grabbing? Is the container open before taking out? Is the object on the cutting board before cutting?
   - **Effect check**: After this action, what changes? Will it leave the agent in a state that allows the next intended action? Will it cause any unintended consequences?
   - If any check fails, revise the step before committing it.

4. **Output format**: Write a numbered list of actions using this exact syntax:
   - `[ACTION] <class_name> (id)` for single-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id)` for two-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id) <class_name> (id)` for three-argument actions
   - Use the object IDs from the environment specification. When multiple instances of an object exist (e.g., bowl id=1 and bowl id=2), pick the appropriate one and be consistent.
   - `duration`, `temperature`, and `timer` arguments are values, not objects. Write them without an ID, e.g., `[WAIT] <5_minutes>`, `[SET_TEMP] <oven> (1) <350F>`, `[PREHEAT] <oven> (1) <375F>`.

5. **Example plan** (for illustration only — not related to your task):
```
1. [WALK] <cabinet> (1)
2. [OPEN] <cabinet> (1)
3. [GRAB] <bowl> (1)
4. [WALK] <sink> (1)
5. [SWITCH_ON] <water> (1)
6. [FILL] <bowl> (1) <water> (1)
7. [SWITCH_OFF] <water> (1)
8. [PUT_ON] <bowl> (1) <counter> (1)
```

6. **Creating new objects with COMBINE**: When ingredients in a container are transformed into a new product (through mixing, cooking, etc.), use `[COMBINE] <container> (id) <product_name> (1)` to declare the result. Give the product a descriptive name. The new object can then be referenced in subsequent steps. For example, after boiling tea leaves in a pot with water: `[COMBINE] <pot> (1) <tea> (1)`, then `[POUR] <pot> (1) <cup> (1)` and `[SERVE] <tea> (1) <cup> (1)`.

7. **Task completion**: The final step of your plan must always be a `[SERVE]` action to indicate the task is complete and present the finished item.

8. **Constraints**:
   - Use ONLY actions defined in the action_definitions section
   - Ensure all preconditions for each action are satisfied by prior steps
   - Every action and object must come from the environment specification
   - The plan should logically achieve the task goal

Output ONLY the numbered plan. Do not include any reasoning, simulation traces, or explanations.

## Plan
"""

        return prompt.strip()

    def _create_prompt_foresight_safety(self, file_content: str, task_goal: str) -> str:
        """Ablation: only safety/appliance checks (no state/precondition/effect checks)."""

        prompt = f"""You are a planning agent in a simulated kitchen environment. Your task is to generate a step-by-step action plan to achieve a given goal using ONLY the objects and actions available in the environment specification below.

## Task Goal
{task_goal}

## Environment Specification
{file_content}

## Instructions

1. **Understand the environment**: Study the objects (their IDs, properties, states, and locations), the available actions (their arguments, preconditions, and effects), and the agent's starting state.

2. **Track state carefully**: The agent starts in the kitchen holding nothing. To interact with an object, the agent must first be at the object's location. Containers (cabinet, fridge, pantry, drawer) must be opened before objects inside them can be taken out. The agent can only hold one object at a time — after grabbing or taking out an object, the agent must put it down before grabbing another.

3. **Counterfactual Foresight Simulation**: Before committing each step, mentally simulate what would happen:
   - **Safety check**: Does this action create any food safety hazard? For example, handling raw meat and then touching ready-to-eat food without washing hands? Using the same cutting board for raw protein and vegetables without cleaning?
   - **Appliance check**: At the end of the plan, are all heat sources (stove, oven, grill) turned off? Is the water turned off?
   - If any check fails, revise the step before committing it.

4. **Output format**: Write a numbered list of actions using this exact syntax:
   - `[ACTION] <class_name> (id)` for single-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id)` for two-argument actions
   - `[ACTION] <class_name> (id) <class_name> (id) <class_name> (id)` for three-argument actions
   - Use the object IDs from the environment specification. When multiple instances of an object exist (e.g., bowl id=1 and bowl id=2), pick the appropriate one and be consistent.
   - `duration`, `temperature`, and `timer` arguments are values, not objects. Write them without an ID, e.g., `[WAIT] <5_minutes>`, `[SET_TEMP] <oven> (1) <350F>`, `[PREHEAT] <oven> (1) <375F>`.

5. **Example plan** (for illustration only — not related to your task):
```
1. [WALK] <cabinet> (1)
2. [OPEN] <cabinet> (1)
3. [GRAB] <bowl> (1)
4. [WALK] <sink> (1)
5. [SWITCH_ON] <water> (1)
6. [FILL] <bowl> (1) <water> (1)
7. [SWITCH_OFF] <water> (1)
8. [PUT_ON] <bowl> (1) <counter> (1)
```

6. **Creating new objects with COMBINE**: When ingredients in a container are transformed into a new product (through mixing, cooking, etc.), use `[COMBINE] <container> (id) <product_name> (1)` to declare the result. Give the product a descriptive name. The new object can then be referenced in subsequent steps. For example, after boiling tea leaves in a pot with water: `[COMBINE] <pot> (1) <tea> (1)`, then `[POUR] <pot> (1) <cup> (1)` and `[SERVE] <tea> (1) <cup> (1)`.

7. **Task completion**: The final step of your plan must always be a `[SERVE]` action to indicate the task is complete and present the finished item.

8. **Constraints**:
   - Use ONLY actions defined in the action_definitions section
   - Ensure all preconditions for each action are satisfied by prior steps
   - Every action and object must come from the environment specification
   - The plan should logically achieve the task goal

Output ONLY the numbered plan. Do not include any reasoning, simulation traces, or explanations.

## Plan
"""

        return prompt.strip()

    def _create_prompt_self_refine(self, file_content: str, task_goal: str, draft_plan: str) -> str:
        """Self-refine prompt: critique and revise a draft plan."""

        prompt = f"""You are a planning agent in a simulated kitchen environment. You previously generated a draft plan for the following task. Your job is to carefully review the draft, identify any errors, and produce a corrected final plan.

## Task Goal
{task_goal}

## Environment Specification
{file_content}

## Draft Plan
{draft_plan}

## Review Instructions

Carefully check the draft plan for the following issues:

1. **State tracking errors**: Does the agent try to grab something while already holding an object? Does it interact with an object at a different location without walking there first? Does it take something from a closed container?

2. **Precondition violations**: Are all preconditions met before each action? For example, is the object on a cutting board before cutting? Is a pan on the stove before heating?

3. **Food safety issues**: Does the plan handle raw meat/poultry/seafood safely? Are hands washed after touching raw protein and before touching ready-to-eat food? Are vegetables and fruits washed before use?

4. **Appliance safety**: Are all heat sources (stove, oven, grill) turned off at the end? Is the water turned off?

5. **Missing steps**: Are there any missing intermediate steps (e.g., forgetting to open a container, forgetting to put down an object before grabbing another)?

After your review, output ONLY the corrected numbered plan. If the draft is already correct, output it unchanged. Do not include any reasoning or explanations.

## Corrected Plan
"""

        return prompt.strip()

    def save_single_result(self, result: Dict[str, Any]) -> None:
        """
        Save a single processing result immediately to disk.

        Args:
            result (Dict[str, Any]): Processing result to save
        """
        try:
            file_path = Path(result['file_path'])
            # Save plan_XXX.txt in the same subfolder as env_XXX.json
            stem = file_path.stem
            output_filename = stem.replace("env_", "plan_", 1) + ".txt"
            output_filepath = file_path.parent / output_filename

            if result['status'] == 'success' and result['llm_response']:
                response = result['llm_response']

                # If response is JSON, extract the content
                try:
                    response_json = json.loads(response)
                    if isinstance(response_json, dict) and 'response' in response_json:
                        response = response_json['response']
                except (json.JSONDecodeError, TypeError):
                    pass

                if self.use_self_refine:
                    # Save draft plan as draft_XXX.txt
                    draft_filename = stem.replace("env_", "draft_", 1) + ".txt"
                    draft_filepath = file_path.parent / draft_filename
                    with open(draft_filepath, 'w', encoding='utf-8') as f:
                        f.write(result.get('draft_plan', '').strip())

                if self.use_foresight:
                    # Save thinking/reasoning traces as response_XXX.txt
                    response_filename = stem.replace("env_", "response_", 1) + ".txt"
                    response_filepath = file_path.parent / response_filename
                    thinking = getattr(self.llm_client.client, '_last_thinking', '')
                    with open(response_filepath, 'w', encoding='utf-8') as f:
                        if thinking:
                            f.write("=== THINKING ===\n")
                            f.write(thinking.strip())
                            f.write("\n\n=== OUTPUT ===\n")
                        f.write(response.strip())

                    # Extract only plan lines into plan_XXX.txt
                    # Matches formats like:
                    #   1. [WALK] <cabinet> (1)
                    #   - Step 1: `[WALK] <cabinet> (1)`.
                    #   - Step 1: [WALK] <cabinet> (1)
                    import re
                    plan_lines = []
                    seen_steps = set()
                    for line in response.strip().split('\n'):
                        line = line.strip()
                        # Direct format: "1. [ACTION] ..."
                        m = re.match(r'^(\d+)\.\s*\[', line)
                        if m:
                            step_num = m.group(1)
                            if step_num not in seen_steps:
                                seen_steps.add(step_num)
                                plan_lines.append(line)
                            continue
                        # Embedded format: "- Step N: `[ACTION] ...`"
                        m = re.search(r'Step\s+(\d+):\s*`?\[(\w+)\]\s*(.*?)`?\.?$', line)
                        if m:
                            step_num = m.group(1)
                            if step_num not in seen_steps:
                                seen_steps.add(step_num)
                                action_part = f'[{m.group(2)}] {m.group(3)}'.rstrip('`.')
                                plan_lines.append(f'{step_num}. {action_part}')

                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        if plan_lines:
                            f.write('\n'.join(plan_lines))
                        else:
                            f.write(response.strip())
                else:
                    # Save response directly as plan_XXX.txt
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(response.strip())
            else:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    f.write(f"Status: {result['status']}\n")

            print(f"  ✓ Saved: {output_filepath}")

        except Exception as e:
            print(f"  ✗ Error saving result for {result['file_name']}: {e}")

    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file with the LLM and save immediately.

        Args:
            file_path (Path): Path to the file to process

        Returns:
            Dict[str, Any]: Processing result including file data and LLM response
        """
        try:
            print(f"Processing file: {file_path.name}")

            # Extract task number from filename (e.g., task_042.json -> 42)
            import re
            match = re.search(r'env_(\d+)', file_path.name)
            if not match:
                raise ValueError(f"Could not extract task number from {file_path.name}")

            task_num = int(match.group(1))

            # Get the task goal
            task_goal = self.task_goals.get(task_num, "Unknown task goal")
            print(f"  Task {task_num}: {task_goal}")

            # Read file content
            file_content = self.read_file_content(file_path)

            # Create prompt for this file
            prompt = self.create_prompt(file_content, task_goal)

            # Call LLM with the prompt
            extra_kwargs = {}
            if self.use_foresight:
                extra_kwargs['reasoning_effort'] = 'high'
            llm_response = self.llm_client.generate_response(prompt, **extra_kwargs)

            # Self-refine: second call to critique and revise
            draft_plan = None
            if self.use_self_refine:
                draft_plan = llm_response.strip()
                refine_prompt = self._create_prompt_self_refine(file_content, task_goal, draft_plan)
                llm_response = self.llm_client.generate_response(refine_prompt, **extra_kwargs)

            # Package the result
            result = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "task_number": task_num,
                "task_goal": task_goal,
                "llm_response": llm_response,
                "draft_plan": draft_plan,
                "status": "success"
            }

            # Save result immediately
            self.save_single_result(result)

            return result

        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            result = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "llm_response": None,
                "error": str(e),
                "status": "error"
            }

            # Save error result immediately
            self.save_single_result(result)

            return result

    def process_all_files(self) -> List[Dict[str, Any]]:
        """
        Process all env JSON files from subfolders in base_dir.

        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        print(f"Retrieving json files from {self.base_dir}")
        json_files = self.get_json_files()
        print(f"Found {len(json_files)} non-empty json files")

        if not json_files:
            print("No json files to process")
            return []

        results = []

        for file_path in json_files:
            # Skip if plan already exists
            plan_path = file_path.parent / file_path.name.replace("env_", "plan_", 1).replace(".json", ".txt")
            if plan_path.exists() and plan_path.stat().st_size > 0:
                print(f"Skipping {file_path.name} (plan already exists)")
                continue

            result = self.process_single_file(file_path)
            results.append(result)

            # Optional: Add delay between API calls to avoid rate limits
            # import time
            # time.sleep(1)

        self.results = results
        return results

    def save_results(self, output_dir: str = "./results"):
        """
        DEPRECATED: Results are now saved immediately during processing.
        This method is kept for backwards compatibility but does nothing.

        Args:
            output_dir (str): Base directory to save individual file results (ignored)
        """
        print(f"\nNote: Results have already been saved to {self.base_dir} during processing.")
        print(f"Total results saved: {len(self.results)}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of processing results.

        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.results:
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "files_processed": []
            }

        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful

        return {
            "total_files": total,
            "successful": successful,
            "failed": failed,
            "files_processed": [r['file_name'] for r in self.results]
        }


def main():
    """
    Main execution function.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process environment files to generate plans using LLM')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Run directory containing NNN/ task folders (create one with scripts/init_run.py)')
    parser.add_argument('--model_family', type=str, default='gpt',
                        choices=['gpt', 'gemini', 'llama', 'deepseek', 'claude', 'qwen'],
                        help='Model family to use (default: gpt)')
    parser.add_argument('--vllm_url', type=str, default=None,
                        help='vLLM server URL for llama/deepseek (default: http://localhost:8000/v1)')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model name (default: None, uses family default)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Only process tasks with index >= start_idx')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Only process tasks with index <= end_idx')
    parser.add_argument('--foresight', action='store_true',
                        help='Use counterfactual foresight simulation prompt')
    parser.add_argument('--self_refine', action='store_true',
                        help='Use self-refine (generate draft then critique and revise)')
    parser.add_argument('--foresight_group', type=str, default=None,
                        choices=['reasoning', 'safety'],
                        help='Ablation: "reasoning" = state/precondition/effect checks only; "safety" = safety/appliance checks only')

    args = parser.parse_args()

    try:
        # Initialize processor with selected model family
        processor = EnvFileProcessor(
            model_family=args.model_family,
            model=args.model,
            base_dir=args.base_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            use_foresight=args.foresight,
            use_self_refine=args.self_refine,
            base_url=args.vllm_url,
            foresight_group=args.foresight_group,
        )

        print(f"Using model family: {args.model_family}")

        # Process all json files (results are saved automatically during processing)
        results = processor.process_all_files()

        # Print summary
        summary = processor.get_summary()
        print(f"\nProcessing Summary:")
        print(f"Total files: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
