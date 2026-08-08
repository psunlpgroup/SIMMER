"""SIMMER: Benchmarking Latent Failures in LLM Executable Planning with a World Model."""

from simmer.paths import (
    ACTION_DEFS,
    OBJECT_DEFS,
    REPO_ROOT,
    RUNS_DIR,
    TASKS_DIR,
    WORLD_MODEL_DIR,
)

__version__ = '1.0.0'

__all__ = [
    'KitchenStateMachine',
    'Failure',
    'FailureType',
    'ACTION_DEFS',
    'OBJECT_DEFS',
    'TASKS_DIR',
    'RUNS_DIR',
    'WORLD_MODEL_DIR',
    'REPO_ROOT',
]

# The executor is imported lazily so that `python -m simmer.state_machine`
# does not load the module twice (PEP 562).
_LAZY = {'KitchenStateMachine', 'Failure', 'FailureType'}


def __getattr__(name):
    if name in _LAZY:
        from simmer import state_machine
        return getattr(state_machine, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)
