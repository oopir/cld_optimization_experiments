"""
Backward-compatible facade for the lazy_training_test experiment.

The original main experiment API remains importable from `src.exp`.
"""

from .lazy_training_test import core as _core
from .lazy_training_test import eta as _eta
from .lazy_training_test.core import (
    EXP1_CHECKPOINT_PREFIX,
    build_from_config_mapping,
    infer_effective_track_every,
    label_from_alpha_beta,
    resume_from_ckpt,
    run_exp,
)
from .lazy_training_test.eta import tune_eta_for_exp

__all__ = [
    "EXP1_CHECKPOINT_PREFIX",
    "build_from_config_mapping",
    "infer_effective_track_every",
    "label_from_alpha_beta",
    "resume_from_ckpt",
    "run_exp",
    "tune_eta_for_exp",
]


def __getattr__(name):
    if hasattr(_core, name):
        return getattr(_core, name)
    if hasattr(_eta, name):
        return getattr(_eta, name)
    raise AttributeError(name)
