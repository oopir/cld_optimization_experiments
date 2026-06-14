"""
Backward-compatible training imports.

Shared primitives now live in `src.base.training`; lazy-training-test-specific
training code lives in `src.lazy_training_test.training`.
"""

from .base import training as _base_training
from .lazy_training_test import training as _lazy_training
from .base.training import (
    BaseModelVars,
    _apply_training_step,
    _configure_deterministic_backend,
    _copy_state_dict_to_cpu,
    _forward_backward,
    _forward_backward_tensors,
    _init_base_model_vars,
    _load_rng_state,
    _save_rng_state,
    _zero_grads,
    run_full_batch_training_checkpoints,
    seed_training_run,
)
from .lazy_training_test.training import (
    LinearizationVars,
    MultiSeedWorkerArgs,
    ResumeState,
    TrainArgs,
    train,
    train_multiseed,
)

__all__ = [
    "BaseModelVars",
    "LinearizationVars",
    "MultiSeedWorkerArgs",
    "ResumeState",
    "TrainArgs",
    "_apply_training_step",
    "_configure_deterministic_backend",
    "_copy_state_dict_to_cpu",
    "_forward_backward",
    "_forward_backward_tensors",
    "_init_base_model_vars",
    "_load_rng_state",
    "_save_rng_state",
    "_zero_grads",
    "run_full_batch_training_checkpoints",
    "seed_training_run",
    "train",
    "train_multiseed",
]


def __getattr__(name):
    if hasattr(_lazy_training, name):
        return getattr(_lazy_training, name)
    if hasattr(_base_training, name):
        return getattr(_base_training, name)
    raise AttributeError(name)
