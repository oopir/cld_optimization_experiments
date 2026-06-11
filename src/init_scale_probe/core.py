from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import math
import random

import numpy as np
import torch

from ..data import load_binary_classification_data
from ..model import TwoLayerNet

# -------------------------------------------------------------------------- #
# ------------------------------- constants -------------------------------- #
# -------------------------------------------------------------------------- #

CORE_METRICS = (
    "mean_abs_output",
    "mean_output_sq",
    "mean_preactivation_norm",
    "mean_abs_residual",
    "empirical_loss",
    "mean_output_grad_norm",
    "mean_output_grad_norm_sq",
    "empirical_loss_grad_norm",
)

FORWARD_METRICS = (
    "mean_abs_output",
    "mean_output_sq",
    "mean_preactivation_norm",
    "mean_abs_residual",
    "empirical_loss",
)

OUTPUT_GRAD_METRICS = (
    "mean_output_grad_norm",
    "mean_output_grad_norm_sq",
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_sq_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_sq_fc2",
)
EMPIRICAL_LOSS_GRAD_METRICS = (
    "empirical_loss_grad_norm",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
)

LAYERWISE_METRICS = (
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_sq_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_sq_fc2",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
)

PARAMETER_METRICS = (
    "fc1_weight_fro_norm",
    "fc1_weight_spectral_norm",
)

ALL_METRICS = CORE_METRICS + LAYERWISE_METRICS + PARAMETER_METRICS
GRADIENT_METRICS = OUTPUT_GRAD_METRICS + EMPIRICAL_LOSS_GRAD_METRICS
SWEEP_AXES = ("n", "m", "alpha")

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
# --------------------------- metric computation --------------------------- #
# -------------------------------------------------------------------------- #

@torch.no_grad()
def _compute_forward_metrics(
    model: TwoLayerNet,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    metric_names = set(metric_names)
    n = X.shape[0]
    totals = {name: 0.0 for name in metric_names}
    output_metric_names = {"mean_abs_output", "mean_output_sq", "mean_abs_residual", "empirical_loss"}
    needs_output = bool(metric_names & output_metric_names)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        yb = y[start:end]

        z = model.fc1(xb)
        if "mean_preactivation_norm" in metric_names:
            totals["mean_preactivation_norm"] += float(torch.linalg.vector_norm(z, dim=1).sum().item())
        if not needs_output:
            continue

        out = model.fc2(torch.tanh(z))
        if model.init_type == "alpha" and model.alpha != 0:
            out = out / model.alpha
        out = out.view(-1, 1)

        if "mean_abs_output" in metric_names:
            totals["mean_abs_output"] += float(out.abs().sum().item())
        if "mean_output_sq" in metric_names:
            totals["mean_output_sq"] += float(out.pow(2).sum().item())
        residual = out - yb
        if "mean_abs_residual" in metric_names:
            totals["mean_abs_residual"] += float(residual.abs().sum().item())
        if "empirical_loss" in metric_names:
            totals["empirical_loss"] += float(residual.pow(2).sum().item())

    return {name: value / n for name, value in totals.items()}

@torch.no_grad()
def _compute_parameter_metrics(
    model: TwoLayerNet,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """Compute initialization-only parameter norms that do not depend on data."""
    metric_names = set(metric_names)
    W1 = model.fc1.weight.detach()
    out: Dict[str, float] = {}
    if "fc1_weight_fro_norm" in metric_names:
        out["fc1_weight_fro_norm"] = float(torch.linalg.matrix_norm(W1, ord="fro").item())
    if "fc1_weight_spectral_norm" in metric_names:
        out["fc1_weight_spectral_norm"] = float(torch.linalg.matrix_norm(W1, ord=2).item())
    return out

@torch.no_grad()
def _compute_gradient_metrics(
    model: TwoLayerNet,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """
    Compute exact scalar-output parameter-gradient norms for the current model.

    For f(x) = scale * fc2(tanh(fc1(x))) with no biases:
      ||grad_fc2 f(x)||^2 = scale^2 * ||h||^2
      ||grad_fc1 f(x)||^2 = scale^2 * ||x||^2 *
          sum_j fc2_j^2 * (1 - h_j^2)^2
    """
    W1 = model.fc1.weight.detach()
    W2 = model.fc2.weight.detach().view(-1)
    scale = 1.0 / float(model.alpha) if model.init_type == "alpha" and model.alpha != 0 else 1.0
    scale_sq = scale * scale
    n = X.shape[0]
    metric_names = set(metric_names)
    needs_output_grad = bool(metric_names & set(OUTPUT_GRAD_METRICS))
    needs_empirical_loss_grad = bool(metric_names & set(EMPIRICAL_LOSS_GRAD_METRICS))

    emp_fc1_sum = torch.zeros_like(W1) if needs_empirical_loss_grad else None
    emp_fc2_sum = torch.zeros_like(W2) if needs_empirical_loss_grad else None

    mean_metrics = tuple(metric_names & set(OUTPUT_GRAD_METRICS))
    totals = {name: 0.0 for name in mean_metrics}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]

        z = xb @ W1.T
        h = torch.tanh(z)

        one_minus_h2 = 1.0 - h.pow(2)
        fc1_out_grad_sq = None
        fc2_out_grad_sq = None
        total_out_grad_sq = None

        if needs_output_grad:
            x_norm_sq = xb.pow(2).sum(dim=1)
            fc2_out_grad_sq = scale_sq * h.pow(2).sum(dim=1)
            fc1_weighted_sq = (W2.pow(2).view(1, -1) * one_minus_h2.pow(2)).sum(dim=1)
            fc1_out_grad_sq = scale_sq * x_norm_sq * fc1_weighted_sq
            total_out_grad_sq = fc1_out_grad_sq + fc2_out_grad_sq

        if needs_output_grad:
            if "mean_output_grad_norm" in metric_names:
                totals["mean_output_grad_norm"] += float(torch.sqrt(total_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq" in metric_names:
                totals["mean_output_grad_norm_sq"] += float(total_out_grad_sq.sum().item())
            if "mean_output_grad_norm_fc1" in metric_names:
                totals["mean_output_grad_norm_fc1"] += float(torch.sqrt(fc1_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq_fc1" in metric_names:
                totals["mean_output_grad_norm_sq_fc1"] += float(fc1_out_grad_sq.sum().item())
            if "mean_output_grad_norm_fc2" in metric_names:
                totals["mean_output_grad_norm_fc2"] += float(torch.sqrt(fc2_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq_fc2" in metric_names:
                totals["mean_output_grad_norm_sq_fc2"] += float(fc2_out_grad_sq.sum().item())

        if needs_empirical_loss_grad:
            yb = y[start:end].view(-1)
            residual = scale * (h @ W2).view(-1) - yb
            coeff = 2.0 * residual * scale
            emp_fc2_sum += (coeff.view(-1, 1) * h).sum(dim=0)
            emp_fc1_batch = coeff.view(-1, 1) * W2.view(1, -1) * one_minus_h2
            emp_fc1_sum += emp_fc1_batch.T @ xb

    out = {name: value / n for name, value in totals.items()}

    if needs_empirical_loss_grad:
        emp_fc1 = emp_fc1_sum / n
        emp_fc2 = emp_fc2_sum / n
        emp_fc1_norm = float(torch.linalg.vector_norm(emp_fc1).item())
        emp_fc2_norm = float(torch.linalg.vector_norm(emp_fc2).item())

        if "empirical_loss_grad_norm_fc1" in metric_names:
            out["empirical_loss_grad_norm_fc1"] = emp_fc1_norm
        if "empirical_loss_grad_norm_fc2" in metric_names:
            out["empirical_loss_grad_norm_fc2"] = emp_fc2_norm
        if "empirical_loss_grad_norm" in metric_names:
            out["empirical_loss_grad_norm"] = math.sqrt(emp_fc1_norm * emp_fc1_norm + emp_fc2_norm * emp_fc2_norm)

    return out


# -------------------------------------------------------------------------- #
# ------------------- probe execution & output generation ------------------ #
# -------------------------------------------------------------------------- #

def _row_for_initialization(
    config: InitScaleProbeConfig,
    binary_data: Mapping[str, Any],
    n: int,
    m: int,
    alpha: float,
    data_seed: int,
    init_seed: int,
    device: str,
) -> Dict[str, Any]:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    model = TwoLayerNet(binary_data["d_in"], m, d_out=1, init_type=config.init_type, alpha=alpha).to(device)
    model.eval()

    X = binary_data["X_train"]
    y = binary_data["y_train_binary"]
    tracked_metrics = config.tracked_metrics or []
    forward_metrics = [name for name in tracked_metrics if name in FORWARD_METRICS]
    gradient_metrics = [name for name in tracked_metrics if name in GRADIENT_METRICS]
    parameter_metrics = [name for name in tracked_metrics if name in PARAMETER_METRICS]

    metrics = {}
    if parameter_metrics:
        metrics.update(_compute_parameter_metrics(model, metric_names=parameter_metrics))
    if forward_metrics:
        metrics.update(_compute_forward_metrics(model, X, y, batch_size=config.batch_size, metric_names=forward_metrics))
    if gradient_metrics:
        metrics.update(
            _compute_gradient_metrics(
                model,
                X,
                y,
                batch_size=config.jacobian_batch_size,
                metric_names=gradient_metrics,
            )
        )

    row = {
        "dataset": config.dataset,
        "init_type": config.init_type,
        "n": int(n),
        "n_effective": int(binary_data["n_effective"]),
        "m": int(m),
        "alpha": float(alpha),
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "device": device,
    }
    for metric_name in tracked_metrics:
        row[metric_name] = metrics[metric_name]
    return row

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
                    for init_seed in config.init_seeds or []:
                        row = _row_for_initialization(
                            config,
                            binary_data,
                            n=n,
                            m=m,
                            alpha=alpha,
                            data_seed=data_seed,
                            init_seed=init_seed,
                            device=device,
                        )
                        rows.append(row)

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
    group_keys = ("dataset", "init_type", "n", "n_effective", "m", "alpha", "data_seed")

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

    For every `(n, m, alpha, data_seed)`, compute the init-seed mean first.
    Then aggregate those data-seed means at each `(n, m, alpha)`. The resulting
    metric std is therefore the standard deviation across sampled datasets, not
    across individual initializations.
    """
    per_seed_groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    per_seed_keys = ("dataset", "init_type", "n", "n_effective", "m", "alpha", "data_seed")
    for row in rows:
        key = tuple(row[name] for name in per_seed_keys)
        per_seed_groups.setdefault(key, []).append(row)

    sweep_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    sweep_keys = ("dataset", "init_type", "n", "m", "alpha")
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
