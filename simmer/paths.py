"""Canonical filesystem locations, resolved from the installed package.

Every path here is absolute, so scripts behave the same regardless of the
working directory they are invoked from.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WORLD_MODEL_DIR = REPO_ROOT / 'world_model'
ACTION_DEFS = WORLD_MODEL_DIR / 'action_def.json'
OBJECT_DEFS = WORLD_MODEL_DIR / 'object_def.json'

TASKS_DIR = REPO_ROOT / 'benchmark' / 'tasks'
RUNS_DIR = REPO_ROOT / 'runs'
