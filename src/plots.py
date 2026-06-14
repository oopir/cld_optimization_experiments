"""Backward-compatible plotting facade."""

from .alpha_sweep import plotting as _alpha_plotting
from .alpha_sweep.plotting import plot_test_error_vs_alpha
from .lazy_training_test import plotting as _lazy_plotting
from .lazy_training_test.plotting import plot_ex1_multiseed

__all__ = ["plot_ex1_multiseed", "plot_test_error_vs_alpha"]


def __getattr__(name):
    if hasattr(_lazy_plotting, name):
        return getattr(_lazy_plotting, name)
    if hasattr(_alpha_plotting, name):
        return getattr(_alpha_plotting, name)
    raise AttributeError(name)
