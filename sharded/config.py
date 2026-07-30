from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Mapping, Optional
import math

import yaml


SHARDED_METRIC_SCHEMA_VERSION = 1

BASE_HISTORY_METRICS = (
    "train_loss",
    "train_acc",
    "test_acc",
    "param_dist",
    "feat_gram_lambda",
)
LIN_HISTORY_METRICS = (
    "lin_train_loss",
    "lin_train_acc",
    "lin_test_acc",
    "lin_param_dist",
)
PAIR_HISTORY_METRICS = (
    "nn_lin_param_dist",
    "jacobian_dist",
)
SUPPORTED_HISTORY_METRICS = BASE_HISTORY_METRICS + LIN_HISTORY_METRICS + PAIR_HISTORY_METRICS
PRINT_METRICS = ("train_loss", "train_acc", "test_acc")
LINEARIZED_METRICS = LIN_HISTORY_METRICS + ("nn_lin_param_dist",)
DEPRECATED_FEATURE_METRICS = ("feat_rel_dist", "feat_cos_dist")


@dataclass(frozen=True)
class ShardedMetricPlan:
    tracked_metrics: tuple[str, ...]
    history_metrics: frozenset[str]
    compute_metrics: frozenset[str]

    @property
    def needs_linearized_metrics(self) -> bool:
        return any(name in self.compute_metrics for name in LINEARIZED_METRICS)

    @property
    def needs_jacobian_reference(self) -> bool:
        return "jacobian_dist" in self.compute_metrics


@dataclass
class ShardedExpConfig:
    # parallelization
    seeds: list[int] = field(default_factory=lambda: [0])
    device: str = "cuda"
    gpu_indices: Optional[list[int]] = None
    # data
    dataset: str = "digits"
    n: int = 10
    random_labels: bool = False
    reserve_last: int = 1000
    # model
    m: int = 1
    L: int = 1
    activation: str = "tanh"
    init_type: str = "standard"
    # training
    epochs: int = 1
    eta: float = 1.0
    eta_mode: str = "scalar"
    eta_table_path: Optional[str] = None
    eta_default: Optional[float] = None
    betas: list[float] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    regularization_scale: float = 1.0
    lam_fc1: Optional[float] = None
    lam_hidden: Optional[float] = None
    lam_fc2: Optional[float] = None
    noise_free_after_epoch: Optional[int] = None
    early_stop_metric: Optional[str] = None
    early_stop_goal: str = "min"
    early_stop_value: Optional[float] = None
    # stats
    use_linearized: bool = True
    same_noise: bool = False
    tracked_metrics: Optional[list[str]] = None
    track_jacobian: bool = True
    jac_probe_size: int = 1
    track_every: int = 10
    print_every: int = 100
    collect_feature_stats: bool = True
    checkpoint_state: str = "metrics_only"


@dataclass
class ShardedRunOpts:
    ckpt_dir: Path
    save_ckpt: bool = False
    load_ckpt: bool = False
    load_ckpt_name: Optional[Path] = None
    resume_from_ckpt: bool = False
    new_total_epochs: Optional[int] = None
    config_overrides: Optional[list[str]] = None
    plot_output_dir: Path = Path("plots")


def _parse_beta(beta) -> float:
    if isinstance(beta, str) and beta in {".inf", "inf", "+inf"}:
        return math.inf
    return float(beta)


def _prepare_dataclass_kwargs(cls, kwargs: Mapping) -> dict:
    field_names = {f.name for f in fields(cls)}
    unknown = set(kwargs) - field_names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} config field(s): {', '.join(sorted(unknown))}")
    return {k: v for k, v in kwargs.items() if k in field_names}


def _expand_path_arg(kwargs: dict, key: str) -> None:
    if key in kwargs and kwargs[key] is not None:
        kwargs[key] = Path(kwargs[key]).expanduser()


def build_from_config_mapping(cfg: Mapping) -> tuple[ShardedExpConfig, ShardedRunOpts]:
    exp_kwargs = dict(cfg.get("experiment", cfg))
    run_kwargs = dict(cfg.get("run", {}))

    if "betas" in exp_kwargs:
        exp_kwargs["betas"] = [_parse_beta(b) for b in exp_kwargs["betas"]]

    exp_config = ShardedExpConfig(**_prepare_dataclass_kwargs(ShardedExpConfig, exp_kwargs))

    _expand_path_arg(run_kwargs, "ckpt_dir")
    _expand_path_arg(run_kwargs, "load_ckpt_name")
    _expand_path_arg(run_kwargs, "plot_output_dir")
    run_opts = ShardedRunOpts(**_prepare_dataclass_kwargs(ShardedRunOpts, run_kwargs))
    return exp_config, run_opts


def load_mapping(path: Path) -> dict:
    with path.expanduser().open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def default_tracked_metrics(use_linearized: bool, track_jacobian: bool) -> list[str]:
    metrics = ["train_loss", "feat_gram_lambda"]
    if use_linearized:
        metrics.extend(["lin_train_loss", "lin_param_dist", "nn_lin_param_dist"])
    if track_jacobian:
        metrics.append("jacobian_dist")
    return metrics


def resolve_metric_plan(config: ShardedExpConfig) -> ShardedMetricPlan:
    if config.tracked_metrics is None:
        requested = default_tracked_metrics(config.use_linearized, config.track_jacobian)
    else:
        requested = list(config.tracked_metrics)

    if config.early_stop_metric is not None and config.early_stop_metric not in requested:
        requested.append(config.early_stop_metric)

    deduped = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    deprecated = sorted(set(deduped) & set(DEPRECATED_FEATURE_METRICS))
    if deprecated:
        raise ValueError(
            "The sharded engine only supports feat_gram_lambda for feature metrics; "
            f"unsupported metric(s): {', '.join(deprecated)}"
        )

    unknown = sorted(set(deduped) - set(SUPPORTED_HISTORY_METRICS))
    if unknown:
        raise ValueError(f"Unsupported sharded metric(s): {', '.join(unknown)}")

    lin_requested = sorted(set(deduped) & set(LINEARIZED_METRICS))
    if lin_requested and not config.use_linearized:
        raise ValueError(
            "Linearized metric(s) requested while use_linearized=False: "
            + ", ".join(lin_requested)
        )

    if config.early_stop_metric in PAIR_HISTORY_METRICS:
        raise ValueError(
            f"early_stop_metric must be scalar; {config.early_stop_metric!r} stores an (L2, cosine) tuple."
        )

    compute_metrics = set(deduped)
    compute_metrics.update(PRINT_METRICS)
    return ShardedMetricPlan(
        tracked_metrics=tuple(deduped),
        history_metrics=frozenset(deduped),
        compute_metrics=frozenset(compute_metrics),
    )


def validate_config(config: ShardedExpConfig, world_size: int) -> None:
    if config.L < 1:
        raise ValueError(f"L must be >= 1, got {config.L}")
    if config.activation != "tanh":
        raise ValueError("The sharded v1 engine only supports activation='tanh'.")
    if config.jac_probe_size < 1:
        raise ValueError(f"jac_probe_size must be >= 1, got {config.jac_probe_size}.")
    if config.gpu_indices is not None:
        if not str(config.device).startswith("cuda"):
            raise ValueError("gpu_indices can only be set when device starts with 'cuda'.")
        if len(config.gpu_indices) != world_size:
            raise ValueError(
                f"gpu_indices length ({len(config.gpu_indices)}) must equal "
                f"torchrun world_size ({world_size})."
            )
        if len(set(config.gpu_indices)) != len(config.gpu_indices):
            raise ValueError(f"gpu_indices must be unique, got {config.gpu_indices}.")
        invalid = [
            idx for idx in config.gpu_indices
            if not isinstance(idx, int) or idx < 0 or idx > 7
        ]
        if invalid:
            raise ValueError(
                "gpu_indices must contain integer CUDA device indices in the range 0..7; "
                f"invalid value(s): {invalid}."
            )
    if config.m % world_size != 0:
        raise ValueError(
            f"m={config.m} must be divisible by world_size={world_size} for v1 row sharding."
        )
    if config.checkpoint_state not in {"metrics_only", "sharded_state"}:
        raise ValueError("checkpoint_state must be 'metrics_only' or 'sharded_state'.")
    if config.early_stop_goal not in {"min", "max"}:
        raise ValueError("early_stop_goal must be 'min' or 'max'.")
    resolve_metric_plan(config)


def _format_scalar_for_key(x: float) -> str:
    if math.isinf(float(x)):
        return "inf"
    x_float = float(x)
    if x_float.is_integer():
        return str(int(x_float))
    return str(x_float)


def load_eta_table(path: Optional[str]) -> dict:
    if path is None:
        return {}
    eta_path = Path(path).expanduser()
    if not eta_path.is_file():
        print(f"[eta] WARNING: eta_table_path '{eta_path}' not found; using defaults.")
        return {}
    with eta_path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_eta(config: ShardedExpConfig, alpha: float, beta: float) -> float:
    default_eta = config.eta if config.eta_default is None else config.eta_default
    if config.eta_mode == "scalar" or config.eta_table_path is None:
        return float(config.eta)

    table = load_eta_table(config.eta_table_path)
    if config.eta_mode == "per_beta":
        per_beta = table.get("per_beta", {})
        key = _format_scalar_for_key(beta)
        return float(per_beta.get(key, default_eta))

    if config.eta_mode == "per_alpha_beta":
        per_ab = table.get("per_alpha_beta", {})
        key = f"alpha={_format_scalar_for_key(alpha)},beta={_format_scalar_for_key(beta)}"
        return float(per_ab.get(key, default_eta))

    raise ValueError(f"Unknown eta_mode: {config.eta_mode}")
