"""Initialization-scale experiment package."""

__all__ = ["InitScaleConfig", "plot_from_rows", "run_experiment"]


def __getattr__(name):
    if name in __all__:
        from .core import InitScaleConfig, plot_from_rows, run_experiment

        values = {
            "InitScaleConfig": InitScaleConfig,
            "plot_from_rows": plot_from_rows,
            "run_experiment": run_experiment,
        }
        return values[name]
    raise AttributeError(name)
