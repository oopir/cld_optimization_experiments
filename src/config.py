"""
Backward-compatible configuration and checkpoint API for the main experiment.

This module intentionally stays importable as `src.config` because existing
torch checkpoints can pickle config objects and helper references at this path.
New experiment code may wrap or import these dataclasses, but moving their
canonical definitions would make old checkpoint loading riskier.
"""

from pathlib import Path
from dataclasses import dataclass, field, fields, MISSING, is_dataclass
from typing import Optional, List
import torch

from . import metric_checkpoints  # noqa: F401; keep old checkpoint classes importable for torch.load
from .metric_config import METRIC_SCHEMA_VERSION, resolve_metric_plan

@dataclass
class ExpConfig:
    # parallelization
    seeds: list = field(default_factory=lambda: [0])
    device: str = "cpu"
    # data
    dataset: str = "digits"
    n: int = 10
    random_labels: bool = False
    reserve_last: int = 1000
    # model
    m: int = 1
    init_type: str = "standard"
    # training
    epochs: int = 1
    eta: float  = 1.0
    eta_mode: str = "scalar"               # "scalar", "per_beta", "per_alpha_beta"
    eta_table_path: Optional[str] = None   # path to YAML table
    eta_default: Optional[float] = None    # fallback if key missing (defaults to eta if None)
    betas: list = field(default_factory=lambda: [])
    alphas: list = field(default_factory=lambda: [])
    regularization_scale: float = 1.0
    lam_fc1: Optional[float] = None
    lam_fc2: Optional[float] = None
    # early stopping & turning off noise
    noise_free_after_epoch: Optional[int] = None
    early_stop_metric: Optional[str] = None
    early_stop_goal: str = "min"
    early_stop_value: Optional[float] = None
    # stats
    use_linearized: bool = True
    same_noise: bool = False
    tracked_metrics: Optional[List[str]] = None
    track_jacobian: bool = True # Legacy metric shim: only used to build default metrics when tracked_metrics is omitted.
    jac_probe_size: int = 10
    track_every: int = 10
    print_every: int = 100 # Legacy metric shim: only used to build default metrics when tracked_metrics is omitted.
    collect_feature_stats: bool = True


@dataclass
class RunOpts:
    ckpt_dir:         Path
    save_ckpt:        bool = False # for saving progress after training
    load_ckpt:        bool = False # for plotting/extending an existing ckpt
    load_ckpt_name:   Optional[Path] = None
    resume_from_ckpt: bool = False
    new_total_epochs: Optional[int] = None
    config_overrides: Optional[List[str]] = None
    plot_output_dir:  Path = Path("plots/lazy_training_test")


def save_checkpoint(path, results, config: ExpConfig):
    metric_plan = resolve_metric_plan(
        tracked_metrics=getattr(config, "tracked_metrics", None),
        use_linearized=getattr(config, "use_linearized", True),
        track_jacobian=getattr(config, "track_jacobian", True),
        collect_feature_stats=getattr(config, "collect_feature_stats", True),
        early_stop_metric=getattr(config, "early_stop_metric", None),
    )
    payload = {
        "type": "exp1",
        "config": config,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "tracked_metrics": list(metric_plan.tracked_metrics),
        "results": results,
    }
    torch.save(payload, path)

def patch_loaded_config(config):
    """
    Fill fields missing from dataclass checkpoint configs.

    This runs after torch.load succeeds. It does not replace metric_checkpoints.py,
    which is needed earlier if an old pickle refers to that module path.
    """
    if not is_dataclass(config):
        return config

    for f in fields(config):
        if hasattr(config, f.name):
            continue

        if f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())
        else:
            raise AttributeError(
                f"Loaded checkpoint config is missing required field {f.name!r} "
                f"and no default exists."
            )

    return config

def _load_checkpoint_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)

    payload_type = payload.get("type", "exp1") # 2nd argument "tolerates" old ckpts w/o "type" field
    if payload_type != "exp1":
        raise ValueError(f"Unexpected checkpoint type: {payload_type}")

    payload["config"] = patch_loaded_config(payload["config"])
    return payload

def load_checkpoint(path):
    payload = _load_checkpoint_payload(path)
    return payload["results"], payload["config"]

def load_checkpoint_with_metadata(path):
    payload = _load_checkpoint_payload(path)
    metadata = {
        "metric_schema_version": payload.get("metric_schema_version"),
        "tracked_metrics": payload.get("tracked_metrics"),
        "has_metric_metadata": "tracked_metrics" in payload,
    }
    return payload["results"], payload["config"], metadata
