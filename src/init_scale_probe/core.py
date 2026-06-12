from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import math

import numpy as np
import torch

from ..data import load_binary_classification_data
from ..stats import (
    BINARY_ALL_METRICS,
    BINARY_CORE_METRICS,
    BINARY_GRADIENT_METRICS,
    BINARY_LAYERWISE_METRICS,
    BINARY_PARAMETER_METRICS,
    get_binary_probe_stats,
)
from ..training import run_full_batch_training_checkpoints, seed_training_run

# -------------------------------------------------------------------------- #
# ------------------------------- constants -------------------------------- #
# -------------------------------------------------------------------------- #

CORE_METRICS = BINARY_CORE_METRICS
LAYERWISE_METRICS = BINARY_LAYERWISE_METRICS
PARAMETER_METRICS = BINARY_PARAMETER_METRICS
ALL_METRICS = BINARY_ALL_METRICS
GRADIENT_METRICS = BINARY_GRADIENT_METRICS
SWEEP_AXES = ("n", "m", "alpha", "beta", "training_steps")


def _parse_beta_value(value: Any) -> float:
    """Parse YAML/CSV beta values, accepting string spellings of infinity."""
    if isinstance(value, str) and value.strip().lower() in {".inf", "inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(value)

# -------------------------------------------------------------------------- #
# ------------------------------- config ----------------------------------- #
# -------------------------------------------------------------------------- #

@dataclass
class InitScaleProbeConfig:
    # data
    dataset: str = "digits"
    random_labels: bool = False
    reserve_last: int = 1000
    negative_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    positive_classes: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    # sweep
    n_values: List[int] = field(default_factory=lambda: [10])
    m_values: List[int] = field(default_factory=lambda: [10])
    alpha_values: List[float] = field(default_factory=lambda: [1.0])
    beta_values: List[float] = field(default_factory=lambda: [float("inf")])
    training_step_values: List[int] = field(default_factory=lambda: [0])
    eta: float = 1.0
    init_type: str = "alpha"
    # randomness
    data_seeds: List[int] = field(default_factory=lambda: [0])
    init_seeds: Optional[List[int]] = None # Effective __post_init__ default is init_seed_start + k.
    num_inits: int = 2
    init_seed_start: int = 10000
    report_data_seed: Optional[int] = None # Effective __post_init__ default is data_seeds[0].
    # execution
    device: str = "cpu"
    batch_size: int = 1024
    jacobian_batch_size: int = 256
    parallel: bool = False
    gpu_ids: Optional[List[int]] = None
    init_chunk_size: int = 4
    adaptive_gpu_packing: bool = True
    gpu_memory_safety_fraction: float = 0.75
    gpu_reserved_memory_mb: int = 1000
    max_workers_per_gpu: int = 8
    retry_on_oom: bool = True
    oom_shrink_factor: float = 0.5
    min_batch_size: int = 1
    # logging
    progress_interval_seconds: Optional[float] = None
    progress_detail: str = "summary"
    # plotting
    tracked_metrics: Optional[List[str]] = None # Effective __post_init__ default is all metrics.
    plot_metrics: List[str] = field(default_factory=lambda: list(CORE_METRICS))
    plot_format: str = "combined"
    plot_facets: str = "auto"
    plot_facet_threshold: int = 6
    plot_heatmaps: bool = True
    # output
    output_dir: Path = Path("plots/init_scale_probe_debug")

    def __post_init__(self):
        self.n_values = [int(x) for x in self.n_values]
        self.m_values = [int(x) for x in self.m_values]
        self.alpha_values = [float(x) for x in self.alpha_values]
        self.beta_values = [_parse_beta_value(x) for x in self.beta_values]
        self.training_step_values = sorted({int(x) for x in self.training_step_values})
        self.eta = float(self.eta)
        self.data_seeds = [int(x) for x in self.data_seeds]
        self.num_inits = int(self.num_inits)
        self.init_seed_start = int(self.init_seed_start)
        self.negative_classes = [int(x) for x in self.negative_classes]
        self.positive_classes = [int(x) for x in self.positive_classes]
        self.reserve_last = int(self.reserve_last)
        self.batch_size = int(self.batch_size)
        self.jacobian_batch_size = int(self.jacobian_batch_size)
        self.output_dir = Path(self.output_dir).expanduser()
        self.plot_metrics = list(self.plot_metrics)
        self.plot_format = str(self.plot_format)
        self.plot_facets = str(self.plot_facets)
        self.plot_facet_threshold = int(self.plot_facet_threshold)
        self.plot_heatmaps = bool(self.plot_heatmaps)
        self.parallel = bool(self.parallel)
        if self.gpu_ids is not None:
            self.gpu_ids = [int(x) for x in self.gpu_ids]
        self.init_chunk_size = int(self.init_chunk_size)
        self.adaptive_gpu_packing = bool(self.adaptive_gpu_packing)
        self.gpu_memory_safety_fraction = float(self.gpu_memory_safety_fraction)
        self.gpu_reserved_memory_mb = int(self.gpu_reserved_memory_mb)
        self.max_workers_per_gpu = int(self.max_workers_per_gpu)
        self.retry_on_oom = bool(self.retry_on_oom)
        self.oom_shrink_factor = float(self.oom_shrink_factor)
        self.min_batch_size = int(self.min_batch_size)
        if self.progress_interval_seconds is not None:
            self.progress_interval_seconds = float(self.progress_interval_seconds)
        self.progress_detail = str(self.progress_detail)

        if self.dataset not in {"digits", "mnist"}:
            raise ValueError(f"Unsupported dataset: {self.dataset!r}")
        if not self.n_values:
            raise ValueError("n_values must be non-empty.")
        if not self.m_values:
            raise ValueError("m_values must be non-empty.")
        if not self.alpha_values:
            raise ValueError("alpha_values must be non-empty.")
        if not self.beta_values:
            raise ValueError("beta_values must be non-empty.")
        if not self.training_step_values:
            raise ValueError("training_step_values must be non-empty.")
        if any(step < 0 for step in self.training_step_values):
            raise ValueError("training_step_values must be non-negative.")
        if any((not math.isinf(beta)) and beta <= 0 for beta in self.beta_values):
            raise ValueError("beta_values must be positive or infinity.")
        if self.eta < 0:
            raise ValueError("eta must be non-negative.")
        if not self.data_seeds:
            raise ValueError("data_seeds must be non-empty.")
        if self.init_seeds is None:
            if self.num_inits <= 0:
                raise ValueError("num_inits must be positive when init_seeds is omitted.")
            self.init_seeds = [self.init_seed_start + k for k in range(self.num_inits)]
        else:
            self.init_seeds = [int(x) for x in self.init_seeds]
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.jacobian_batch_size <= 0:
            raise ValueError("jacobian_batch_size must be positive.")
        if self.init_chunk_size <= 0:
            raise ValueError("init_chunk_size must be positive.")
        if not (0 < self.gpu_memory_safety_fraction <= 1):
            raise ValueError("gpu_memory_safety_fraction must be in (0, 1].")
        if self.gpu_reserved_memory_mb < 0:
            raise ValueError("gpu_reserved_memory_mb must be non-negative.")
        if self.max_workers_per_gpu <= 0:
            raise ValueError("max_workers_per_gpu must be positive.")
        if not (0 < self.oom_shrink_factor < 1):
            raise ValueError("oom_shrink_factor must be in (0, 1).")
        if self.min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive.")
        if self.progress_interval_seconds is not None and self.progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be non-negative or null.")
        if self.progress_detail not in {"summary", "grid"}:
            raise ValueError("progress_detail must be one of: summary, grid.")
        if self.plot_format not in {"combined", "individual", "both"}:
            raise ValueError("plot_format must be one of: combined, individual, both.")
        if self.plot_facets not in {"auto", "off"}:
            raise ValueError("plot_facets must be one of: auto, off.")
        if self.plot_facet_threshold <= 0:
            raise ValueError("plot_facet_threshold must be positive.")
        if not self.negative_classes:
            raise ValueError("negative_classes must be non-empty.")
        if not self.positive_classes:
            raise ValueError("positive_classes must be non-empty.")

        overlap = set(self.negative_classes) & set(self.positive_classes)
        if overlap:
            raise ValueError(f"positive_classes and negative_classes overlap: {sorted(overlap)}")

        if self.tracked_metrics is None:
            self.tracked_metrics = list(ALL_METRICS)
        else:
            self.tracked_metrics = list(self.tracked_metrics)

        unknown_tracked_metrics = sorted(set(self.tracked_metrics) - set(ALL_METRICS))
        if unknown_tracked_metrics:
            raise ValueError(f"Unknown tracked metric(s): {', '.join(unknown_tracked_metrics)}")

        unknown_plot_metrics = sorted(set(self.plot_metrics) - set(ALL_METRICS))
        if unknown_plot_metrics:
            raise ValueError(f"Unknown plot metric(s): {', '.join(unknown_plot_metrics)}")

        untracked_plot_metrics = sorted(set(self.plot_metrics) - set(self.tracked_metrics))
        if untracked_plot_metrics:
            raise ValueError(
                "plot_metrics must be included in tracked_metrics; missing: "
                + ", ".join(untracked_plot_metrics)
            )

        if self.report_data_seed is None:
            self.report_data_seed = int(self.data_seeds[0])
        else:
            self.report_data_seed = int(self.report_data_seed)
        if self.report_data_seed not in self.data_seeds:
            raise ValueError(
                f"report_data_seed={self.report_data_seed} is not in data_seeds={self.data_seeds}."
            )

# -------------------------------------------------------------------------- #
# ------------------- probe execution & output generation ------------------ #
# -------------------------------------------------------------------------- #

def _row_from_metrics(
    config: InitScaleProbeConfig,
    binary_data: Mapping[str, Any],
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
    Build one raw CSV row from scalar probe metrics.

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
        "n_effective": int(binary_data["n_effective"]),
        "m": int(m),
        "alpha": float(alpha),
        "beta": float(beta),
        "training_steps": int(training_step),
        "eta": float(config.eta),
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "device": device,
    }
    for metric_name in config.tracked_metrics or []:
        row[metric_name] = metrics[metric_name]
    return row

def _rows_for_trained_initialization(
    config: InitScaleProbeConfig,
    binary_data: Mapping[str, Any],
    n: int,
    m: int,
    alpha: float,
    beta: float,
    data_seed: int,
    init_seed: int,
    device: str,
) -> List[Dict[str, Any]]:
    """Train one scalar binary trajectory and emit rows at requested steps."""
    seed_training_run(init_seed, device)

    X = binary_data["X_train"]
    y = binary_data["y_train_binary"]
    tracked_metrics = config.tracked_metrics or []
    rows: List[Dict[str, Any]] = []

    def measure(training_step, base):
        metrics = get_binary_probe_stats(
            base.model,
            X,
            y,
            tracked_metrics,
            batch_size=config.batch_size,
            jacobian_batch_size=config.jacobian_batch_size,
        )
        rows.append(
            _row_from_metrics(
                config,
                binary_data,
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
        d_in=int(binary_data["d_in"]),
        d_out=1,
        m=m,
        init_type=config.init_type,
        alpha=alpha,
        device=device,
        X_train=X,
        targets=y,
        beta=beta,
        eta=config.eta,
        checkpoint_steps=config.training_step_values,
        measure_fn=measure,
        batch_size=config.batch_size,
        lam_fc1=None,
        lam_fc2=None,
        regularization_scale=1.0,
    )
    return rows

def sort_probe_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return rows in a stable order across serial and parallel runs."""
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                int(row["n"]),
                int(row["data_seed"]),
                int(row["m"]),
                float(row["alpha"]),
                float(row["beta"]),
                int(row["init_seed"]),
                int(row["training_steps"]),
            ),
        )
    ]

def run_probe(config: InitScaleProbeConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Path]]:
    if config.parallel:
        from .parallel import run_probe_parallel
        return run_probe_parallel(config)

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for n in config.n_values:
        for data_seed in config.data_seeds:
            binary_data = load_binary_classification_data(
                dataset=config.dataset,
                n=n,
                negative_classes=config.negative_classes,
                positive_classes=config.positive_classes,
                random_labels=config.random_labels,
                device=device,
                seed=data_seed,
                reserve_last=config.reserve_last,
            )
            for m in config.m_values:
                for alpha in config.alpha_values:
                    for beta in config.beta_values:
                        for init_seed in config.init_seeds or []:
                            rows.extend(
                                _rows_for_trained_initialization(
                                    config,
                                    binary_data,
                                    n=n,
                                    m=m,
                                    alpha=alpha,
                                    beta=beta,
                                    data_seed=data_seed,
                                    init_seed=init_seed,
                                    device=device,
                                )
                            )

    rows = sort_probe_rows(rows)

    summary_rows = summarize_rows(rows, config.tracked_metrics or [], report_data_seed=config.report_data_seed)
    data_seed_summary_rows = summarize_data_seed_rows(rows, config.tracked_metrics or [])

    paths = {
        "rows": output_dir / "init_scale_rows.csv",
        "summary": output_dir / "init_scale_summary.csv",
    }
    write_csv(paths["rows"], rows)
    write_csv(paths["summary"], summary_rows)
    from .plotting import plot_probe_summaries

    plot_paths = plot_probe_summaries(
        summary_rows,
        config,
        output_dir,
        data_seed_summary_rows=data_seed_summary_rows,
    )
    paths.update(plot_paths)
    return rows, summary_rows, paths

def plot_probe_from_rows(
    config: InitScaleProbeConfig,
    rows_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Path]]:
    """Regenerate summaries and plots from a saved raw rows CSV."""
    rows_path = Path(rows_path or config.output_dir / "init_scale_rows.csv").expanduser()
    rows = read_csv(rows_path)
    if not rows:
        raise ValueError(f"Rows CSV is empty: {rows_path}")
    rows = sort_probe_rows(rows)

    missing_plot_metrics = [name for name in config.plot_metrics if name not in rows[0]]
    if missing_plot_metrics:
        raise ValueError(
            "Rows CSV does not contain requested plot metric(s): "
            + ", ".join(missing_plot_metrics)
        )

    metric_names = [name for name in (config.tracked_metrics or []) if name in rows[0]]
    summary_rows = summarize_rows(rows, metric_names, report_data_seed=config.report_data_seed)
    data_seed_summary_rows = summarize_data_seed_rows(rows, metric_names)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": rows_path,
        "summary": config.output_dir / "init_scale_summary.csv",
    }
    write_csv(paths["summary"], summary_rows)
    from .plotting import plot_probe_summaries

    paths.update(
        plot_probe_summaries(
            summary_rows,
            config,
            config.output_dir,
            data_seed_summary_rows=data_seed_summary_rows,
        )
    )
    return rows, summary_rows, paths

def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    report_data_seed: int,
) -> List[Dict[str, Any]]:
    selected = [row for row in rows if int(row["data_seed"]) == int(report_data_seed)]
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
        summary_rows.append(summary)

    return summary_rows

def summarize_data_seed_rows(
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
        "eta",
        "data_seed",
    )
    for row in rows:
        key = tuple(row[name] for name in per_seed_keys)
        per_seed_groups.setdefault(key, []).append(row)

    sweep_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    sweep_keys = ("dataset", "init_type", "n", "m", "alpha", "beta", "training_steps", "eta")
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
        summary_rows.append(summary)

    return summary_rows

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
