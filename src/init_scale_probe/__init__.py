"""Binary initialization-scale probe package."""

__all__ = ["InitScaleProbeConfig", "plot_probe_from_rows", "run_probe"]


def __getattr__(name):
    if name in __all__:
        from .core import InitScaleProbeConfig, plot_probe_from_rows, run_probe

        values = {
            "InitScaleProbeConfig": InitScaleProbeConfig,
            "plot_probe_from_rows": plot_probe_from_rows,
            "run_probe": run_probe,
        }
        return values[name]
    raise AttributeError(name)
