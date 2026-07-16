from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import math

import numpy as np
import torch

from ..base.data import load_binary_classification_data
from ..base.training import run_full_batch_training_checkpoints, seed_training_run
from .metrics import (
    append_k_to_ntk_label_energy_metric,
    append_k_to_ntk_residual_energy_metric,
    ALL_METRICS as METRIC_ALL_METRICS,
    CORE_METRICS as METRIC_CORE_METRICS,
    DEFAULT_METRICS as METRIC_DEFAULT_METRICS,
    GRADIENT_METRICS as METRIC_GRADIENT_METRICS,
    LAYERWISE_METRICS as METRIC_LAYERWISE_METRICS,
    PARAMETER_METRICS as METRIC_PARAMETER_METRICS,
    is_ntk_metric,
    ntk_loss_weighted_average_base_metric,
    ntk_loss_weighted_average_dependencies,
    ntk_metric_needs_initial_matrix,
    NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES,
    parse_ntk_energy_metric,
    compute_ntk_matrix,
    get_metrics,
)

# -------------------------------------------------------------------------- #
# ------------------------------- constants -------------------------------- #
# -------------------------------------------------------------------------- #

CORE_METRICS = METRIC_CORE_METRICS
LAYERWISE_METRICS = METRIC_LAYERWISE_METRICS
PARAMETER_METRICS = METRIC_PARAMETER_METRICS
ALL_METRICS = METRIC_ALL_METRICS
DEFAULT_METRICS = METRIC_DEFAULT_METRICS
GRADIENT_METRICS = METRIC_GRADIENT_METRICS
SWEEP_AXES = ("n", "m", "alpha", "beta", "training_steps", "synthetic_anisotropy_power")
SUPPORTED_DATASETS = {"digits", "mnist", "synthetic_isotropic", "synthetic_anisotropic"}


# -------------------------------------------------------------------------- #
# --------------------- config normalization and validation ---------------- #
# -------------------------------------------------------------------------- #

def normalize_common_sweep_config(config: Any) -> None:
    """Normalize config fields shared by training-stats and at-init-stats experiments."""
    config.n_values = [int(x) for x in config.n_values]
    config.m_values = [int(x) for x in config.m_values]
    config.alpha_values = [float(x) for x in config.alpha_values]
    config.beta_values = [_parse_beta_value(x) for x in config.beta_values]
    config.eta = float(config.eta)
    config.negative_classes = [int(x) for x in config.negative_classes]
    config.positive_classes = [int(x) for x in config.positive_classes]
    config.reserve_last = int(config.reserve_last)
    config.synthetic_d_in = int(config.synthetic_d_in)
    config.synthetic_test_size = int(config.synthetic_test_size)
    config.synthetic_projection_fraction = float(config.synthetic_projection_fraction)
    config.synthetic_anisotropy_power = float(config.synthetic_anisotropy_power)
    if config.synthetic_anisotropy_powers is None:
        config.synthetic_anisotropy_powers = [config.synthetic_anisotropy_power]
    else:
        config.synthetic_anisotropy_powers = [float(x) for x in config.synthetic_anisotropy_powers]

    config.batch_size = int(config.batch_size)
    config.jacobian_batch_size = int(config.jacobian_batch_size)
    config.output_dir = Path(config.output_dir).expanduser()
    config.plot_metrics = list(config.plot_metrics)
    if config.tracked_metrics is not None:
        config.tracked_metrics = list(config.tracked_metrics)
    config.ntk_label_energy_k_values = sorted({int(k) for k in config.ntk_label_energy_k_values})
    config.plot_format = str(config.plot_format)
    config.plot_heatmaps = bool(config.plot_heatmaps)
    config.parallel = bool(config.parallel)
    if config.gpu_ids is not None:
        config.gpu_ids = [int(x) for x in config.gpu_ids]
    config.init_chunk_size = int(config.init_chunk_size)
    config.adaptive_gpu_packing = bool(config.adaptive_gpu_packing)
    config.gpu_memory_safety_fraction = float(config.gpu_memory_safety_fraction)
    config.gpu_reserved_memory_mb = int(config.gpu_reserved_memory_mb)
    config.max_workers_per_gpu = int(config.max_workers_per_gpu)
    config.retry_on_oom = bool(config.retry_on_oom)
    config.oom_shrink_factor = float(config.oom_shrink_factor)
    config.min_batch_size = int(config.min_batch_size)
    if config.progress_interval_seconds is not None:
        config.progress_interval_seconds = float(config.progress_interval_seconds)
    config.progress_detail = str(config.progress_detail)


def normalize_init_seeds(config: Any) -> None:
    """Normalize the shared initialization-seed surface."""
    config.init_seeds, config.num_inits, config.init_seed_start = normalize_seed_list(
        config.init_seeds,
        config.num_inits,
        config.init_seed_start,
        seeds_name="init_seeds",
        count_name="num_inits",
    )


def normalize_seed_list(
    seeds: Optional[Sequence[int]],
    count: int,
    start: int,
    *,
    seeds_name: str,
    count_name: str,
) -> Tuple[List[int], int, int]:
    """Normalize an explicit seed list or generate one from count/start fields."""
    count = int(count)
    start = int(start)
    if seeds is None:
        if count <= 0:
            raise ValueError(f"{count_name} must be positive when {seeds_name} is omitted.")
        return [start + k for k in range(count)], count, start

    normalized = [int(x) for x in seeds]
    if not normalized:
        raise ValueError(f"{seeds_name} must be non-empty.")
    return normalized, count, start


def validate_common_data_and_sweep_fields(config: Any) -> None:
    """Validate data, model, and sweep fields shared by both experiments."""
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.dataset!r}")
    if config.synthetic_d_in <= 0:
        raise ValueError("synthetic_d_in must be positive.")
    if config.synthetic_test_size < 0:
        raise ValueError("synthetic_test_size must be non-negative.")
    if not (0 < config.synthetic_projection_fraction <= 1):
        raise ValueError("synthetic_projection_fraction must be in (0, 1].")
    if config.synthetic_anisotropy_power < 0:
        raise ValueError("synthetic_anisotropy_power must be non-negative.")
    if not config.synthetic_anisotropy_powers:
        raise ValueError("synthetic_anisotropy_powers must be non-empty when provided.")
    if any(power < 0 for power in config.synthetic_anisotropy_powers):
        raise ValueError("synthetic_anisotropy_powers must be non-negative.")
    if not config.n_values:
        raise ValueError("n_values must be non-empty.")
    if not config.m_values:
        raise ValueError("m_values must be non-empty.")
    if not config.alpha_values:
        raise ValueError("alpha_values must be non-empty.")
    if not config.beta_values:
        raise ValueError("beta_values must be non-empty.")
    if any((not math.isinf(beta)) and beta <= 0 for beta in config.beta_values):
        raise ValueError("beta_values must be positive or infinity.")
    if config.eta < 0:
        raise ValueError("eta must be non-negative.")
    if not config.negative_classes:
        raise ValueError("negative_classes must be non-empty.")
    if not config.positive_classes:
        raise ValueError("positive_classes must be non-empty.")

    overlap = set(config.negative_classes) & set(config.positive_classes)
    if overlap:
        raise ValueError(f"positive_classes and negative_classes overlap: {sorted(overlap)}")


def validate_common_execution_fields(config: Any) -> None:
    """Validate execution and plotting controls shared by both experiments."""
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.jacobian_batch_size <= 0:
        raise ValueError("jacobian_batch_size must be positive.")
    if config.init_chunk_size <= 0:
        raise ValueError("init_chunk_size must be positive.")
    if not (0 < config.gpu_memory_safety_fraction <= 1):
        raise ValueError("gpu_memory_safety_fraction must be in (0, 1].")
    if config.gpu_reserved_memory_mb < 0:
        raise ValueError("gpu_reserved_memory_mb must be non-negative.")
    if config.max_workers_per_gpu <= 0:
        raise ValueError("max_workers_per_gpu must be positive.")
    if not (0 < config.oom_shrink_factor < 1):
        raise ValueError("oom_shrink_factor must be in (0, 1).")
    if config.min_batch_size <= 0:
        raise ValueError("min_batch_size must be positive.")
    if config.progress_interval_seconds is not None and config.progress_interval_seconds < 0:
        raise ValueError("progress_interval_seconds must be non-negative or null.")
    if config.progress_detail not in {"summary", "grid"}:
        raise ValueError("progress_detail must be one of: summary, grid.")
    if config.plot_format not in {"combined", "individual", "both"}:
        raise ValueError("plot_format must be one of: combined, individual, both.")


def resolve_common_metric_fields(config: Any) -> None:
    """Validate plot/tracked metric names and expand tracked metrics needed for plots."""
    if not config.ntk_label_energy_k_values:
        raise ValueError("ntk_label_energy_k_values must be non-empty.")
    if any(k <= 0 for k in config.ntk_label_energy_k_values):
        raise ValueError("ntk_label_energy_k_values must contain only positive integers.")

    valid_tracked_metrics = _valid_tracked_metric_names(config.ntk_label_energy_k_values)
    valid_plot_metrics = _valid_plot_metric_names(config.ntk_label_energy_k_values)

    unknown_plot_metrics = _unknown_metric_names(config.plot_metrics, valid_plot_metrics)
    if unknown_plot_metrics:
        hint = _configured_dynamic_metric_error_hint(unknown_plot_metrics) or ""
        raise ValueError(f"Unknown plot metric(s): {', '.join(unknown_plot_metrics)}{hint}")

    config.tracked_metrics = _resolve_tracked_metrics_for_plots(config.tracked_metrics, config.plot_metrics)

    unknown_tracked_metrics = _unknown_metric_names(config.tracked_metrics, valid_tracked_metrics)
    if unknown_tracked_metrics:
        hint = _configured_dynamic_metric_error_hint(unknown_tracked_metrics) or ""
        raise ValueError(f"Unknown tracked metric(s): {', '.join(unknown_tracked_metrics)}{hint}")

    untracked_plot_metrics = sorted(
        name for name in set(config.plot_metrics) - set(config.tracked_metrics)
        if ntk_loss_weighted_average_dependencies(name) is None
    )
    if untracked_plot_metrics:
        raise ValueError(
            "plot_metrics must be included in tracked_metrics; missing: "
            + ", ".join(untracked_plot_metrics)
        )


def _parse_beta_value(value: Any) -> float:
    """Parse YAML/CSV beta values, accepting string spellings of infinity."""
    if isinstance(value, str) and value.strip().lower() in {".inf", "inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(value)


def _valid_tracked_metric_names(ntk_label_energy_k_values: Sequence[int]) -> set:
    """Return per-run metrics plus configured dynamic NTK metric names."""
    return set(ALL_METRICS) | {
        metric_name
        for k in ntk_label_energy_k_values
        for metric_name in (
            append_k_to_ntk_label_energy_metric(k),
            append_k_to_ntk_residual_energy_metric(k),
        )
    }


def _valid_plot_metric_names(ntk_label_energy_k_values: Sequence[int]) -> set:
    """Return per-run plot metrics plus configured loss-weighted average names."""
    return _valid_tracked_metric_names(ntk_label_energy_k_values) | {
        *NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES,
    }


def _unknown_metric_names(metric_names: Sequence[str], valid_names: set) -> List[str]:
    """Return metrics that are not known for the configured dynamic metric set."""
    return sorted(set(metric_names) - valid_names)


def _configured_dynamic_metric_error_hint(metric_names: Sequence[str]) -> Optional[str]:
    """Return a helpful config hint for unknown dynamic metric names."""
    if any(parse_ntk_energy_metric(name) is not None for name in metric_names):
        return ". Add matching k values to ntk_label_energy_k_values."
    return None


def _expand_loss_weighted_average_metrics(metric_names: Sequence[str]) -> List[str]:
    """Replace loss-weighted average aliases with the raw metrics needed to compute them."""
    expanded: List[str] = []
    # Expand user-facing loss-weighted averages into the raw per-run metrics stored in rows.
    for metric_name in metric_names:
        dependencies = ntk_loss_weighted_average_dependencies(metric_name)
        for name in dependencies or (metric_name,):
            if name not in expanded:
                expanded.append(name)
    return expanded


def _resolve_tracked_metrics_for_plots(tracked_metrics: Optional[Sequence[str]], plot_metrics: Sequence[str]) -> List[str]:
    """Return the raw tracked metrics required by the requested plot metrics."""
    if tracked_metrics is None:
        resolved = list(DEFAULT_METRICS)
        add_plotted_ntk_metrics = True
    else:
        resolved = _expand_loss_weighted_average_metrics(tracked_metrics)
        add_plotted_ntk_metrics = False

    # Add comparison metrics and raw dependencies needed by requested plots.
    for metric_name in plot_metrics:
        dependencies = ntk_loss_weighted_average_dependencies(metric_name)
        base_metric = ntk_loss_weighted_average_base_metric(metric_name)
        implied_metrics = []
        if base_metric is not None:
            implied_metrics.append(base_metric)
        if metric_name == "residual_initial_ntk_alignment":
            implied_metrics.append("residual_ntk_alignment")
        if metric_name == "residual_ntk_alignment_over_initial":
            implied_metrics.extend(("residual_ntk_alignment", "residual_initial_ntk_alignment"))
        if metric_name == "residual_ntk_alignment_trace_normalized_over_initial":
            implied_metrics.extend(("residual_ntk_alignment", "residual_initial_ntk_alignment"))
        if metric_name == "task_initial_ntk_alignment":
            implied_metrics.append("task_ntk_alignment")
        if metric_name == "task_ntk_alignment_over_initial":
            implied_metrics.extend(("task_ntk_alignment", "task_initial_ntk_alignment"))
        if metric_name == "task_ntk_alignment_trace_normalized_over_initial":
            implied_metrics.extend(("task_ntk_alignment", "task_initial_ntk_alignment"))
        if dependencies is not None:
            implied_metrics.extend(dependencies)
        elif add_plotted_ntk_metrics and is_ntk_metric(metric_name):
            implied_metrics.append(metric_name)
        for implied_metric in implied_metrics:
            if implied_metric not in resolved:
                resolved.append(implied_metric)
    return resolved


# -------------------------------------------------------------------------- #
# -------------------------- runtime and row checks ------------------------ #
# -------------------------------------------------------------------------- #

def resolve_run_device(config: Any) -> str:
    """Resolve `auto` and fail early for unavailable CUDA devices."""
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable.")
    return device


def missing_plot_metrics_from_rows(config: Any, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return requested plot metrics that cannot be reconstructed from a raw rows CSV."""
    if not rows:
        return list(config.plot_metrics)
    first_row = rows[0]
    return [
        name for name in config.plot_metrics
        if name not in first_row
        and any(
            dependency not in first_row
            for dependency in (ntk_loss_weighted_average_dependencies(name) or (name,))
        )
    ]


def validate_plot_metrics_in_rows(config: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Raise when plot-only rows cannot support the requested plot metrics."""
    missing_plot_metrics = missing_plot_metrics_from_rows(config, rows)
    if missing_plot_metrics:
        raise ValueError(
            "Rows CSV does not contain requested plot metric(s): "
            + ", ".join(missing_plot_metrics)
        )


def metric_names_present_in_rows(config: Any, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return configured tracked metrics physically present in a rows CSV."""
    if not rows:
        return []
    return [name for name in (config.tracked_metrics or []) if name in rows[0]]


# -------------------------------------------------------------------------- #
# ------------------------------ row generation ---------------------------- #
# -------------------------------------------------------------------------- #

def run_sweep_rows(
    config: Any,
    *,
    device: str,
    data_seeds: Sequence[int],
    synthetic_anisotropy_powers: Sequence[float],
) -> List[Dict[str, Any]]:
    """Run the shared serial sweep grid and return deterministically sorted rows."""
    rows: List[Dict[str, Any]] = []
    for n in config.n_values:
        for data_seed in data_seeds:
            for anisotropy_power in synthetic_anisotropy_powers:
                run_config = replace(config, synthetic_anisotropy_power=float(anisotropy_power))
                data = load_sweep_data(run_config, n=n, data_seed=data_seed, device=device)
                for m in run_config.m_values:
                    for alpha in run_config.alpha_values:
                        for beta in run_config.beta_values:
                            for init_seed in run_config.init_seeds or []:
                                rows.extend(
                                    _rows_for_trained_initialization(
                                        run_config,
                                        data,
                                        n=n,
                                        m=m,
                                        alpha=alpha,
                                        beta=beta,
                                        data_seed=data_seed,
                                        init_seed=init_seed,
                                        device=device,
                                    )
                                )
    return sort_rows(rows)


def load_sweep_data(config: Any, n: int, data_seed: int, device: str) -> Mapping[str, Any]:
    """Load the binary-classification data for one sweep point."""
    return load_binary_classification_data(
        dataset=config.dataset,
        n=n,
        negative_classes=config.negative_classes,
        positive_classes=config.positive_classes,
        random_labels=config.random_labels,
        device=device,
        seed=data_seed,
        reserve_last=config.reserve_last,
        synthetic_d_in=config.synthetic_d_in,
        synthetic_test_size=config.synthetic_test_size,
        synthetic_projection_fraction=config.synthetic_projection_fraction,
        synthetic_anisotropy_power=config.synthetic_anisotropy_power,
    )


def _row_from_metrics(
    config: Any,
    data: Mapping[str, Any],
    n: int,
    m: int,
    alpha: float,
    beta: float,
    training_step: int,
    data_seed: int,
    init_seed: int,
    device: str,
    metrics: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Build one raw CSV row from scalar metrics.

    `metrics` is expected to map every configured tracked metric name to a
    scalar numeric value, e.g. float or int. Tuple/list-valued metrics are not
    supported by the CSV summary code.

    `training_step` is one scalar checkpoint value; the output column remains
    "training_steps" because that is the public sweep-axis name.
    """
    row = {
        "dataset": config.dataset,
        "init_type": config.init_type,
        "n": int(n),
        "n_effective": int(data["n_effective"]),
        "m": int(m),
        "alpha": float(alpha),
        "beta": float(beta),
        "training_steps": int(training_step),
        "synthetic_anisotropy_power": float(config.synthetic_anisotropy_power),
        "eta": float(config.eta),
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "device": device,
    }
    for metric_name in config.tracked_metrics or []:
        row[metric_name] = metrics[metric_name]
    return row


def _rows_for_trained_initialization(
    config: Any,
    data: Mapping[str, Any],
    n: int,
    m: int,
    alpha: float,
    beta: float,
    data_seed: int,
    init_seed: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Train one scalar-output trajectory and emit rows at requested steps."""
    seed_training_run(init_seed, device)

    X = data["X_train"]
    y = data["y_train_binary"]
    X_test = data.get("X_test")
    y_test = data.get("y_test_binary")
    tracked_metrics = config.tracked_metrics or []
    requested_steps = set(int(step) for step in config.training_step_values)
    needs_initial_ntk = any(ntk_metric_needs_initial_matrix(name) for name in tracked_metrics)
    measurement_steps = sorted(requested_steps | ({0} if needs_initial_ntk else set()))
    initial_ntk_matrix: Optional[torch.Tensor] = None
    rows: List[Dict[str, Any]] = []

    def measure(training_step, base):
        nonlocal initial_ntk_matrix
        if needs_initial_ntk and initial_ntk_matrix is None:
            initial_ntk_matrix = compute_ntk_matrix(
                base.model,
                X,
                batch_size=config.jacobian_batch_size,
            )
        if int(training_step) not in requested_steps:
            return
        metrics = get_metrics(
            base.model,
            X,
            y,
            tracked_metrics,
            batch_size=config.batch_size,
            jacobian_batch_size=config.jacobian_batch_size,
            X_test=X_test,
            y_test=y_test,
            initial_ntk_matrix=initial_ntk_matrix,
        )
        rows.append(
            _row_from_metrics(
                config,
                data,
                n=n,
                m=m,
                alpha=alpha,
                beta=beta,
                training_step=training_step,
                data_seed=data_seed,
                init_seed=init_seed,
                device=device,
                metrics=metrics,
            )
        )

    run_full_batch_training_checkpoints(
        d_in=int(data["d_in"]),
        d_out=1,
        m=m,
        init_type=config.init_type,
        alpha=alpha,
        device=device,
        X_train=X,
        targets=y,
        beta=beta,
        eta=config.eta,
        checkpoint_steps=measurement_steps,
        measure_fn=measure,
        batch_size=config.batch_size,
        lam_fc1=None,
        lam_fc2=None,
        regularization_scale=1.0,
    )
    return rows


# -------------------------------------------------------------------------- #
# ----------------------------- row summaries ------------------------------ #
# -------------------------------------------------------------------------- #

def sort_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return rows in a stable order across serial and parallel runs."""
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                int(row["n"]),
                int(row["data_seed"]),
                float(row["synthetic_anisotropy_power"]),
                int(row["m"]),
                float(row["alpha"]),
                float(row["beta"]),
                int(row["init_seed"]),
                int(row["training_steps"]),
            ),
        )
    ]


def summarize_all_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Summarize all raw seed rows at each fixed at-init-stats sweep point.

    Unlike the historical report-data-seed summary, this aggregates across both
    data and initialization seeds. The decomposed init/data variability summaries
    below remain responsible for separating those two variance sources in plots.
    """
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    group_keys = (
        "dataset",
        "init_type",
        "n",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
    )

    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {name: value for name, value in zip(group_keys, key)}
        summary["n_effective"] = float(np.mean([float(row["n_effective"]) for row in group_rows]))
        summary["num_data_seeds"] = len({int(row["data_seed"]) for row in group_rows})
        summary["num_inits"] = len({int(row["init_seed"]) for row in group_rows})
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())
        _add_loss_weighted_average_metrics(summary)
        summary_rows.append(summary)

    return summary_rows


def summarize_training_stats_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Summarize a training-stats run, refusing ambiguous multi-data-seed rows."""
    data_seeds = sorted({int(row["data_seed"]) for row in rows})
    if len(data_seeds) != 1:
        raise ValueError(
            "training_stats summaries require exactly one data_seed in the raw rows; "
            f"found {data_seeds}. Re-run or filter the rows before plotting."
        )
    return summarize_rows(rows, metric_names, data_seed=data_seeds[0])


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    data_seed: int,
) -> List[Dict[str, Any]]:
    """Summarize one selected data seed over initialization seeds."""
    selected = [row for row in rows if int(row["data_seed"]) == int(data_seed)]
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    group_keys = (
        "dataset",
        "init_type",
        "n",
        "n_effective",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
        "data_seed",
    )

    for row in selected:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {name: value for name, value in zip(group_keys, key)}
        summary["num_inits"] = len(group_rows)
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())
        _add_loss_weighted_average_metrics(summary)
        summary_rows.append(summary)

    return summary_rows


def _safe_average_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or NaN for zero denominators."""
    return float("nan") if denominator == 0.0 else float(numerator / denominator)


def _add_loss_weighted_average_metrics(summary: Dict[str, Any]) -> None:
    """
    Add loss-weighted NTK averages from raw means.

    Current empirical_loss is ||r||^2 / n instead of ||r||^2 / (2n), but the
    missing factor 1/2 cancels in E[loss * metric] / E[loss].
    """
    loss_mean = summary.get("empirical_loss_mean")
    if loss_mean is None:
        return
    loss_mean = float(loss_mean)

    for out_name, (_, product_name) in NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES.items():
        product_key = f"{product_name}_mean"
        if product_key in summary:
            summary[f"{out_name}_mean"] = _safe_average_ratio(float(summary[product_key]), loss_mean)


def summarize_data_averaged_init_variability_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Summarize init-seed variation after averaging each init seed over data seeds.

    For every `(n, m, alpha, beta, training_steps, init_seed)`, compute the
    data-seed mean first. Then aggregate those init-seed means at each sweep
    point. The resulting metric std is therefore the standard deviation across
    sampled initializations, not across individual data splits.
    """
    per_seed_groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    per_seed_keys = (
        "dataset",
        "init_type",
        "n",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
        "init_seed",
    )
    for row in rows:
        key = tuple(row[name] for name in per_seed_keys)
        per_seed_groups.setdefault(key, []).append(row)

    sweep_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    sweep_keys = (
        "dataset",
        "init_type",
        "n",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
    )
    for key, group_rows in per_seed_groups.items():
        per_seed = {name: value for name, value in zip(per_seed_keys, key)}
        per_seed["n_effective"] = float(np.mean([float(row["n_effective"]) for row in group_rows]))
        per_seed["num_data_seeds"] = len(group_rows)
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=float)
            per_seed[f"{metric_name}_mean"] = float(values.mean())
        sweep_key = tuple(per_seed[name] for name in sweep_keys)
        sweep_groups.setdefault(sweep_key, []).append(per_seed)

    summary_rows: List[Dict[str, Any]] = []
    for key, seed_rows in sorted(sweep_groups.items(), key=lambda item: item[0]):
        summary = {name: value for name, value in zip(sweep_keys, key)}
        summary["n_effective"] = float(np.mean([float(row["n_effective"]) for row in seed_rows]))
        summary["num_data_seeds"] = float(np.mean([float(row["num_data_seeds"]) for row in seed_rows]))
        summary["num_inits"] = len(seed_rows)
        for metric_name in metric_names:
            values = np.asarray([float(row[f"{metric_name}_mean"]) for row in seed_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())
        _add_loss_weighted_average_metrics(summary)
        summary_rows.append(summary)

    return summary_rows


def summarize_init_averaged_data_variability_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Summarize data-seed variation after averaging each data seed over init seeds.

    For every `(n, m, alpha, beta, training_steps, data_seed)`, compute the
    init-seed mean first. Then aggregate those data-seed means at each sweep
    point. The resulting metric std is therefore the standard deviation across
    sampled datasets, not across individual initializations.
    """
    per_seed_groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    per_seed_keys = (
        "dataset",
        "init_type",
        "n",
        "n_effective",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
        "data_seed",
    )
    for row in rows:
        key = tuple(row[name] for name in per_seed_keys)
        per_seed_groups.setdefault(key, []).append(row)

    sweep_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    sweep_keys = (
        "dataset",
        "init_type",
        "n",
        "m",
        "alpha",
        "beta",
        "training_steps",
        "synthetic_anisotropy_power",
        "eta",
    )
    for key, group_rows in per_seed_groups.items():
        per_seed = {name: value for name, value in zip(per_seed_keys, key)}
        per_seed["num_inits"] = len(group_rows)
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=float)
            per_seed[f"{metric_name}_mean"] = float(values.mean())
        sweep_key = tuple(per_seed[name] for name in sweep_keys)
        sweep_groups.setdefault(sweep_key, []).append(per_seed)

    summary_rows: List[Dict[str, Any]] = []
    for key, seed_rows in sorted(sweep_groups.items(), key=lambda item: item[0]):
        summary = {name: value for name, value in zip(sweep_keys, key)}
        summary["n_effective"] = float(np.mean([float(row["n_effective"]) for row in seed_rows]))
        summary["num_data_seeds"] = len(seed_rows)
        summary["num_inits"] = float(np.mean([float(row["num_inits"]) for row in seed_rows]))
        for metric_name in metric_names:
            values = np.asarray([float(row[f"{metric_name}_mean"]) for row in seed_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std())
        _add_loss_weighted_average_metrics(summary)
        summary_rows.append(summary)

    return summary_rows


# -------------------------------------------------------------------------- #
# --------------------------------- csv I/O -------------------------------- #
# -------------------------------------------------------------------------- #

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write row dictionaries to CSV using the first row's field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_output(output_dir: Path, rows_filename: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Path]:
    """Write raw rows and remove stale summary CSVs from older runs."""
    prepare_output_dir(output_dir)
    paths = {"rows": output_dir / rows_filename}
    write_csv(paths["rows"], rows)
    return paths


def prepare_output_dir(output_dir: Path) -> None:
    """Create an output directory and clear summary CSVs from older code paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_summary_csv(output_dir)


def _remove_stale_summary_csv(output_dir: Path) -> None:
    """Remove stale persisted summary CSVs from older experiment outputs."""
    for filename in (
        "_at_init_stats_summary.csv",
        "_training_stats_summary.csv",
        "_init_scale_summary.csv",
    ):
        summary_path = output_dir / filename
        if summary_path.exists():
            summary_path.unlink()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV and coerce obvious numeric scalar fields."""
    with Path(path).expanduser().open("r", newline="") as f:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def _coerce_csv_value(value: str) -> Any:
    """Coerce CSV strings back to simple int/float scalars when possible."""
    if value == "":
        return value
    try:
        as_int = int(value)
        if str(as_int) == value:
            return as_int
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
