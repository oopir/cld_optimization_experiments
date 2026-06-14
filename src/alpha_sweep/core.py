from __future__ import annotations

from typing import Any, Mapping

from ..config import ExpConfig, RunOpts
from ..lazy_training_test.core import build_from_config_mapping as _build_lazy_config
from ..lazy_training_test.core import run_exp
from ..lazy_training_test.eta import tune_eta_for_exp


def _validate_alpha_sweep_config(config: ExpConfig) -> None:
    if not getattr(config, "alphas", None):
        raise ValueError("alpha_sweep requires experiment.alphas to be non-empty.")


def build_from_config_mapping(cfg: Mapping[str, Any]) -> tuple[ExpConfig, RunOpts]:
    """Parse an alpha-sweep config using the backward-compatible main schema."""
    config, run_opts = _build_lazy_config(dict(cfg))
    _validate_alpha_sweep_config(config)
    return config, run_opts


def run_alpha_sweep(config: ExpConfig, run_opts: RunOpts, gpu_ids):
    """Run the alpha sweep as an independent experiment surface."""
    _validate_alpha_sweep_config(config)
    return run_exp(config=config, run_opts=run_opts, gpu_ids=gpu_ids)


def tune_eta_for_alpha_sweep(config: ExpConfig, tuning_cfg: Mapping[str, Any], gpu_ids) -> None:
    """Tune eta over alpha/beta pairs for the alpha-sweep experiment."""
    _validate_alpha_sweep_config(config)
    if "mode" not in tuning_cfg:
        tuning_cfg = dict(tuning_cfg)
        tuning_cfg["mode"] = "per_alpha_beta"
    tune_eta_for_exp(config, tuning_cfg, gpu_ids=gpu_ids)
