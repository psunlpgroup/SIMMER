# SIMMER

**Benchmarking Latent Failures in LLM Executable Planning with a World Model**

SIMMER evaluates LLM planning by *executing* generated plans against a symbolic world
model, rather than by checking surface-level plan similarity. It targets a class of error
that conventional benchmarks miss: **latent failures** — steps that violate no
precondition and produce no execution-time feedback, yet silently compromise the goal.
Many of them are **irreversible**: once triggered, no subsequent action can restore a
valid world state.

> A robot slices raw chicken on a cutting board, cooks it, then reuses the *unwashed*
> board to chop lettuce for a salad. Every action executes successfully. The plan is
> still unsafe, and by the time the salad is served the contamination cannot be undone.

The benchmark has three parts:

| Component | What it is |
|---|---|
| **Symbolic world model** | 77 actions and 262 objects in the kitchen domain, curated from real wikiHow and Instructables cooking scripts (~46,800 semantically realistic interactions) |
| **Failure taxonomy** | Immediate failures (block execution) vs. latent failures (propagate silently), with irreversible latent failures called out separately |
| **State machine executor** | Simulates a plan step by step, tracks fine-grained state, and emits a structured failure report |

## Contents

```
world_model/          77 action definitions + 262 object definitions (the world model)
benchmark/tasks/      100 cooking tasks; each NNN/ holds task_NNN.txt + env_NNN.json
simmer/               importable package: state machine executor and LLM clients
scripts/              plan generation, evaluation, statistics
  baselines/          ReAct, RAP, ProgPrompt, LLM+P, Inner Monologue, few-shot
examples/             a worked example of a latent, irreversible failure
```

## Install

```bash
git clone https://github.com/psunlpgroup/SIMMER.git
cd SIMMER
pip install -r requirements.txt
```

The executor itself has no third-party dependencies — `numpy` and `scipy` are only used
by the statistical analysis, and the LLM SDKs only by the planners.

## Quickstart: see a latent failure

`examples/chicken_salad/` contains the scenario above as a runnable plan.

```bash
python -m simmer.state_machine \
    examples/chicken_salad/env_900.json \
    examples/chicken_salad/plan_900.txt
```

```
Found 1 failure(s):

  [Step 0] (latent) Food safety: lettuce contaminated by raw chicken_breast  |  action: [FOOD_SAFETY_AUDIT]
```

Every step satisfied its preconditions, so nothing failed during execution. The
contamination was introduced when the chicken was sliced, spread silently through the
shared cutting board to the lettuce, and only surfaced in the post-execution audit — by
which point it could no longer be undone.

`plan_900_clean.txt` fixes it by reordering the workflow — chop the lettuce on the clean
board *before* the raw chicken ever touches it — and reports no failures:

```bash
python -m simmer.state_machine \
    examples/chicken_salad/env_900.json \
    examples/chicken_salad/plan_900_clean.txt
```

## Evaluating your own planner

### 1. Create a run directory

A run directory mirrors `benchmark/tasks/`: one `NNN/` folder per task, each holding the
task goal and the initial world state. Planners write `plan_NNN.txt` into these folders
and the evaluator writes `result_NNN.txt` alongside, so a run stays self-contained.

```bash
python scripts/init_run.py my_run              # all 100 tasks
python scripts/init_run.py my_run --end_idx 10 # a small subset, for smoke tests
```

### 2. Generate plans

Either drop your own `plan_NNN.txt` files into `runs/my_run/NNN/`, or use the bundled
planners. Plans use the VirtualHome-style format
`[ACTION] <object_class> (object_id)`, one step per line.

```bash
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY

python scripts/generate_plan.py --base_dir runs/my_run --model_family gpt
python scripts/generate_plan.py --base_dir runs/my_run --model_family gpt --self_refine
python scripts/generate_plan.py --base_dir runs/my_run --model_family gpt --foresight
```

`--model_family` accepts `gpt`, `claude`, `gemini`, `deepseek`, `llama`, `qwen`.
Open-weight models are served through an OpenAI-compatible endpoint — point
`--vllm_url` at your vLLM server (or set `VLLM_URL`).

Prior-work baselines live in `scripts/baselines/` and take the same arguments:

```bash
python scripts/baselines/react.py --base_dir runs/my_run --model_family gpt
```

### 3. Evaluate

```bash
python scripts/evaluate_plans.py --base_dir runs/my_run
```

This executes every plan against the world model and reports, per model and per task,
the counts of immediate, latent, and irreversible failures, writing an annotated
`result_NNN.txt` next to each plan.

For confidence intervals and significance tests between methods:

```bash
python scripts/bootstrap_analysis.py \
    --dirs runs/baseline:Vanilla runs/foresight:Foresight
```

### Using the executor directly

```python
from simmer import ACTION_DEFS, KitchenStateMachine

sm = KitchenStateMachine('benchmark/tasks/000/env_000.json', str(ACTION_DEFS))
for failure in sm.execute_plan_file('runs/my_run/000/plan_000.txt'):
    print(failure.failure_type, failure.step, failure.reason, failure.reversible)
```

## The world model

Definitions follow the PDDL paradigm. An action is
`⟨args, preconditions, effects⟩`; an object is `⟨properties, states, location⟩`, where
properties are immutable affordances and states are mutable attributes.

```jsonc
// world_model/action_def.json
"grab": { "definition": {
    "args": ["object"],
    "preconditions": ["agent_hands_empty", "object_grabbable"],
    "effects": ["agent_holding_object"] } }

// world_model/object_def.json
"knife": { "properties": ["grabbable", "sharp", "tool"],
           "states": ["clean"],
           "location": "drawer" }
```

Each task's `env_NNN.json` carries the initial object instances for that task plus the
subset of action definitions shown to the planner in its prompt. **Evaluation always
uses the canonical `world_model/action_def.json`**, not the copy inlined in the task
environment.

## Benchmark statistics

| | |
|---|---|
| Tasks | 100 cooking scripts, 12 techniques |
| Natural-language steps per task | 2–18 (mean 9.1) |
| Objects per task | 22–56 (mean 31.5) |
| Actions available per task | 21–34 (mean 26.6) |
| World model coverage | all 77 actions and all 262 objects appear across the task set |

## Failure detection

Detection runs in two phases:

- **Phase 1 — step-by-step execution** catches *immediate* failures. Each action is
  parsed, its arguments bound to roles, and its preconditions evaluated against the
  current state; violations are recorded. State transitions then update the world,
  including implicit propagation such as contamination spread.
- **Phase 2 — post-execution audit** catches *latent* failures. The final state is
  scanned for unsafe conditions that no individual step flagged: food carrying uncooked
  contamination, appliances left on, unwashed produce in the finished dish.

A failure is marked irreversible when no subsequent action could have restored a valid
state — cross-contamination of a raw item that is then served, for instance, as opposed
to a reversible omission like forgetting to add salt.

## Known limitations

- **Decontamination requires a `dirty` state.** Contact with raw protein sets an
  object's `contaminated_by` attribute but does not add the `dirty` state, while the
  `wash` action requires `dirty` as a precondition. A surface that starts `clean` and
  becomes contaminated therefore cannot be washed clean, so the modeled recovery path is
  unreachable for it — avoiding contamination requires reordering the plan rather than
  washing. In practice this is rarely load-bearing (across the 600 baseline plans behind
  the paper's main table, only 3 ever attempt to wash a non-`dirty` object), but it does
  constrain how a "correct" plan can be written.
- The world model is deliberately scoped to the kitchen domain; it is not intended as a
  general household simulator.

## Citation

```bibtex
@inproceedings{simmer2026,
  title     = {SIMMER: Benchmarking Latent Failures in LLM Executable Planning with a World Model},
  author    = {Xiaoxin Lu and Ranran Haoran Zhang and Rui Zhang},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

The cooking scripts in `benchmark/tasks/` are derived from community-contributed guides
on wikiHow and Instructables and are redistributed here in processed form for research
use; please respect the original sources' terms.
