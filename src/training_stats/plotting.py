from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
import numpy as np

from .sweep import ALL_METRICS, SWEEP_AXES

# -------------------------------------------------------------------------- #
# ------------------------------- constants -------------------------------- #
# -------------------------------------------------------------------------- #

PDF_PAD_INCHES = 0.08
TRAINING_TITLE_WRAP_WIDTH = 52

LEGACY_METRICS = (
    "mean_loss",
    "mean_sample_loss_grad_norm",
    "mean_sample_loss_grad_norm_sq",
    "mean_sample_loss_grad_norm_fc1",
    "mean_sample_loss_grad_norm_sq_fc1",
    "mean_sample_loss_grad_norm_fc2",
    "mean_sample_loss_grad_norm_sq_fc2",
)

LOSS_GROUP_METRICS = (
    "empirical_loss",
    "train_error",
    "test_error",
)
PARAMETER_NORM_GROUP_METRICS = (
    "fc1_weight_fro_norm",
    "fc1_weight_fro_norm_normalized",
    "fc1_weight_spectral_norm",
    "fc1_weight_spectral_norm_normalized",
    "fc2_weight_euclidean_norm",
)
NTK_SPECTRUM_GROUP_METRICS = (
    "ntk_eig_min",
    "ntk_eig_mean",
    "ntk_eig_median",
    "ntk_eig_max",
)
RESIDUAL_NTK_ALIGNMENT_GROUP_METRICS = (
    "residual_ntk_alignment",
    "residual_ntk_alignment_over_ntk_eig_min",
    "residual_ntk_alignment_over_ntk_eig_mean",
    "residual_ntk_alignment_over_ntk_eig_max",
)
LOSS_WEIGHTED_NTK_GROUP_PAIRS = (
    ("residual_ntk_alignment", "loss_weighted_residual_ntk_alignment"),
    ("ntk_eig_min", "loss_weighted_ntk_eig_min"),
)
LOSS_WEIGHTED_NTK_GROUP_METRICS = tuple(name for pair in LOSS_WEIGHTED_NTK_GROUP_PAIRS for name in pair)
NTK_ALIGNMENT_DYNAMICS_GROUP_METRICS = (
    "residual_ntk_alignment_residual_dynamics_term",
    "residual_ntk_alignment_ntk_dynamics_term",
)
NTK_DRIFT_GROUP_METRICS = (
    "residual_initial_ntk_alignment",
    "residual_ntk_alignment_over_initial",
    "residual_ntk_alignment_trace_normalized_over_initial",
    "task_ntk_alignment",
    "task_initial_ntk_alignment",
    "task_ntk_alignment_over_initial",
    "task_ntk_alignment_trace_normalized_over_initial",
    "ntk_cos_dist",
    "ntk_rel_fro_dist",
)
NTK_DRIFT_RESIDUAL_COMPARISON_METRICS = (
    "residual_ntk_alignment",
    "residual_initial_ntk_alignment",
    "residual_ntk_alignment_over_initial",
    "residual_ntk_alignment_trace_normalized_over_initial",
)
NTK_DRIFT_TASK_COMPARISON_METRICS = (
    "task_ntk_alignment",
    "task_initial_ntk_alignment",
    "task_ntk_alignment_over_initial",
    "task_ntk_alignment_trace_normalized_over_initial",
)
TRAINING_LOG_Y_METRICS = (
    "empirical_loss",
    "residual_ntk_alignment",
    "residual_initial_ntk_alignment",
    "residual_ntk_alignment_over_ntk_eig_min",
    "residual_ntk_alignment_over_ntk_eig_mean",
    "residual_ntk_alignment_over_ntk_eig_max",
    "residual_ntk_alignment_over_initial",
    "task_ntk_alignment_over_initial",
    "loss_weighted_residual_ntk_alignment",
    "loss_weighted_ntk_eig_min",
)
GROUPED_METRIC_PDFS = (
    ("loss", LOSS_GROUP_METRICS),
    ("parameter_norms", PARAMETER_NORM_GROUP_METRICS),
    ("ntk_spectrum_metrics", NTK_SPECTRUM_GROUP_METRICS),
    ("ntk_energy_metrics", ("ntk_label_energy_top_", "ntk_residual_energy_top_")),
    ("residual_ntk_alignment_metrics", RESIDUAL_NTK_ALIGNMENT_GROUP_METRICS),
    ("loss_weighted_ntk_metrics", LOSS_WEIGHTED_NTK_GROUP_METRICS),
    ("ntk_alignment_dynamics_terms", NTK_ALIGNMENT_DYNAMICS_GROUP_METRICS),
    ("ntk_drift_metrics", NTK_DRIFT_GROUP_METRICS),
)


# -------------------------------------------------------------------------- #
# ----------------------------- public entrypoints ------------------------- #
# -------------------------------------------------------------------------- #


def plot_training_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create training-step curve and heatmap plot outputs only."""
    paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_plot_files(output_dir, metric_names=config.plot_metrics)
    if not summary_rows:
        return paths

    if _is_initialization_only(config):
        raise ValueError("plot_training_summaries requires multiple training steps.")

    grouped_metrics = _grouped_metric_names(config.plot_metrics)
    grouped_metric_names = {name for names in grouped_metrics.values() for name in names}

    for metric_name in config.plot_metrics:
        if f"{metric_name}_mean" not in summary_rows[0]:
            continue

        if config.plot_format in {"combined", "both"} and metric_name not in grouped_metric_names:
            path = output_dir / f"{metric_name}.pdf"
            figures: List[Optional[plt.Figure]] = [_make_training_curves_figure(summary_rows, metric_name)]
            if config.plot_heatmaps:
                figures.append(_make_nm_heatmaps_figure(summary_rows, metric_name))
            if _save_figures_pdf(figures, path):
                paths[f"plot_{metric_name}"] = path

        if config.plot_format in {"individual", "both"}:
            curve_path = output_dir / f"{metric_name}_training_curves.pdf"
            if _save_figure_pdf(_make_training_curves_figure(summary_rows, metric_name), curve_path):
                paths[f"plot_{metric_name}_training_curves"] = curve_path
            if config.plot_heatmaps:
                heatmap_path = output_dir / f"{metric_name}_nm_heatmaps.pdf"
                if _save_figure_pdf(_make_nm_heatmaps_figure(summary_rows, metric_name), heatmap_path):
                    paths[f"plot_{metric_name}_nm_heatmaps"] = heatmap_path

    if config.plot_format in {"combined", "both"}:
        paths.update(_plot_grouped_training_metric_pdfs(summary_rows, config, output_dir, grouped_metrics))

    return paths


def plot_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Compatibility wrapper for the training plotting entrypoint."""
    if _is_initialization_only(config):
        raise ValueError("Use at_init_stats.plotting for fixed-step initialization plots.")
    return plot_training_summaries(summary_rows, config, output_dir)


def _is_initialization_only(config: Any) -> bool:
    """Return True when the run has no training-step sweep."""
    return len(tuple(config.training_step_values or [])) == 1


# -------------------------------------------------------------------------- #
# ---------------------------- metric grouping ----------------------------- #
# -------------------------------------------------------------------------- #

def _grouped_metric_names(metric_names: Sequence[str]) -> Dict[str, List[str]]:
    """Return configured metric groups that should be written as grouped PDFs."""
    requested = set(metric_names)
    groups: Dict[str, List[str]] = {file_stem: [] for file_stem, _ in GROUPED_METRIC_PDFS}
    for metric_name in metric_names:
        for file_stem, matcher in GROUPED_METRIC_PDFS:
            if file_stem == "loss_weighted_ntk_metrics":
                continue
            if isinstance(matcher, str):
                matches = metric_name.startswith(matcher)
            elif matcher and isinstance(matcher[0], str) and matcher[0].endswith("_"):
                matches = any(metric_name.startswith(prefix) for prefix in matcher)
            else:
                matches = metric_name in matcher
            if matches:
                groups[file_stem].append(metric_name)
                break
    # Pair each loss-weighted average with its unweighted counterpart for direct comparison.
    for pair in LOSS_WEIGHTED_NTK_GROUP_PAIRS:
        if any(metric_name in requested for metric_name in pair):
            groups["loss_weighted_ntk_metrics"].extend(pair)
    drift_comparison_metrics = []
    if requested & set(NTK_DRIFT_RESIDUAL_COMPARISON_METRICS):
        drift_comparison_metrics.extend(NTK_DRIFT_RESIDUAL_COMPARISON_METRICS)
    if requested & set(NTK_DRIFT_TASK_COMPARISON_METRICS):
        drift_comparison_metrics.extend(NTK_DRIFT_TASK_COMPARISON_METRICS)
    if drift_comparison_metrics:
        groups["ntk_drift_metrics"] = _unique_metric_names(
            (*drift_comparison_metrics, *groups["ntk_drift_metrics"])
        )
    return {file_stem: names for file_stem, names in groups.items() if names}


def _unique_metric_names(metric_names: Sequence[str]) -> List[str]:
    """Return metric names in first-seen order without duplicates."""
    return list(dict.fromkeys(metric_names))


def _plot_grouped_training_metric_pdfs(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
    grouped_metrics: Mapping[str, Sequence[str]],
) -> Dict[str, Path]:
    """Create grouped training-sweep PDFs with one metric per page."""
    paths: Dict[str, Path] = {}
    for file_stem, metric_names in grouped_metrics.items():
        curve_figures: List[Optional[plt.Figure]] = []
        heatmap_figures: List[Optional[plt.Figure]] = []
        for metric_name in metric_names:
            if f"{metric_name}_mean" not in summary_rows[0]:
                continue
            curve_figures.append(_make_training_curves_figure(summary_rows, metric_name))
            if config.plot_heatmaps:
                heatmap_figures.append(_make_nm_heatmaps_figure(summary_rows, metric_name))
        if file_stem == "loss":
            curve_figures.append(_make_final_test_error_vs_m_figure(summary_rows))
        path = output_dir / f"{file_stem}.pdf"
        if _save_figures_pdf_equal_width(curve_figures, path):
            paths[f"plot_{file_stem}"] = path
        if config.plot_heatmaps:
            heatmap_path = output_dir / f"{file_stem}_nm_heatmaps.pdf"
            if _save_figures_pdf_equal_width(heatmap_figures, heatmap_path):
                paths[f"plot_{file_stem}_nm_heatmaps"] = heatmap_path
    return paths


# -------------------------------------------------------------------------- #
# ----------------------------- training plots ----------------------------- #
# -------------------------------------------------------------------------- #

def _make_training_curves_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> Optional[plt.Figure]:
    """
    Build metric-vs-training-step small multiples.

    Panels are fixed by `(n, beta)`: one row per sample size and one column per
    beta value. Within each panel, color encodes `m`.
    """
    mean_key = f"{metric_name}_mean"
    rows = [row for row in summary_rows if mean_key in row]
    if not rows:
        return None

    beta_values = _unique_values(rows, "beta")
    n_values = _unique_values(rows, "n")
    m_values = _unique_values(rows, "m")
    step_values = _unique_values(rows, "training_steps")
    if not step_values:
        return None
    x_uses_log_scale = _training_step_axis_uses_log_scale(step_values)
    step_tick_values = _training_step_tick_values(step_values, x_uses_log_scale)
    x_tick_values = [_training_step_plot_value(value, x_uses_log_scale) for value in step_tick_values]
    y_uses_log_scale = _training_curve_uses_log_y(metric_name, rows, mean_key)

    has_width_legend = len(m_values) > 1
    title = _wrap_figure_text(f"{_metric_label(metric_name)} vs training steps", width=TRAINING_TITLE_WRAP_WIDTH)
    title_line_count = title.count("\n") + 1
    fig_width = max(3.4 * len(beta_values) + (1.7 if has_width_legend else 0.9), 5.2)
    fig_height = max(2.35 * len(n_values) + 1.35 + 0.22 * (title_line_count - 1), 3.8)
    fig, axes = plt.subplots(
        len(n_values),
        len(beta_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(m_values), 1)))
    color_by_m = {m_value: colors[idx] for idx, m_value in enumerate(m_values)}
    seen_m_values = set()

    for row_idx, n in enumerate(n_values):
        for col_idx, beta in enumerate(beta_values):
            ax = axes[row_idx][col_idx]
            panel_rows = [row for row in rows if row["n"] == n and row["beta"] == beta]
            for m in m_values:
                line_rows = sorted(
                    [
                        row for row in panel_rows
                        if row["m"] == m and np.isfinite(float(row[mean_key]))
                    ],
                    key=lambda row: _sort_key(row["training_steps"]),
                )
                if not line_rows:
                    continue
                ax.plot(
                    np.asarray(
                        [_training_step_plot_value(row["training_steps"], x_uses_log_scale) for row in line_rows],
                        dtype=float,
                    ),
                    np.asarray([float(row[mean_key]) for row in line_rows], dtype=float),
                    marker="o",
                    linewidth=1.8,
                    markersize=4.0,
                    color=color_by_m[m],
                    linestyle="-",
                    label=f"m={_format_value(m)}",
                )
                seen_m_values.add(m)

            if row_idx == 0:
                ax.set_title(f"beta={_format_value(beta)}", fontsize=9)
            if x_uses_log_scale:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_minor_formatter(NullFormatter())
            if y_uses_log_scale:
                ax.set_yscale("log")
            ax.set_xticks(x_tick_values)
            ax.set_xticklabels([_format_value(value) for value in step_tick_values])
            ax.grid(False)
            if row_idx == len(n_values) - 1:
                ax.set_xlabel(_training_step_axis_label(x_uses_log_scale))
            if col_idx == 0:
                ax.set_ylabel("mean value")
            if col_idx == len(beta_values) - 1:
                ax.annotate(
                    f"n={_format_value(n)}",
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(12, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                    annotation_clip=False,
                )
            if _uses_zero_reference_line(metric_name):
                ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--", alpha=0.8)
            if _uses_one_reference_line(metric_name):
                ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--", alpha=0.8)

    top = 0.74 if title_line_count > 1 else 0.80
    right = 0.74 if seen_m_values else 0.90
    fig.subplots_adjust(
        left=0.20,
        right=right,
        bottom=0.18,
        top=top,
        wspace=0.30,
        hspace=0.42,
    )
    if seen_m_values:
        m_handles = [
            Line2D([0], [0], color=color_by_m[m], marker="o", linestyle="-", linewidth=2.0, markersize=5)
            for m in m_values
            if m in seen_m_values
        ]
        m_labels = [f"m={_format_value(m)}" for m in m_values if m in seen_m_values]
        fig.legend(
            m_handles,
            m_labels,
            loc="upper left",
            bbox_to_anchor=(right + 0.04, top),
            borderaxespad=0.0,
            frameon=False,
            title="width",
        )
    fig.suptitle(title, fontsize=12, y=0.965)
    return fig


def _training_step_axis_uses_log_scale(step_values: Sequence[Any]) -> bool:
    """Use log(step + 1) when every plotted checkpoint can be shifted positive."""
    shifted = np.asarray([float(value) + 1.0 for value in step_values], dtype=float)
    return bool(shifted.size > 0 and np.all(np.isfinite(shifted)) and np.all(shifted > 0.0))


def _training_step_plot_value(value: Any, use_log_scale: bool) -> float:
    """Return the x-coordinate for a training checkpoint without mutating row data."""
    value = float(value)
    return value + 1.0 if use_log_scale else value


def _training_step_tick_values(
    step_values: Sequence[Any],
    use_log_scale: bool,
    max_ticks: int = 5,
) -> List[Any]:
    """Return a sparse set of real checkpoint steps to label on the x-axis."""
    steps = list(sorted(step_values, key=_sort_key))
    if not use_log_scale or len(steps) <= max_ticks:
        return steps

    x = np.asarray([_training_step_plot_value(step, use_log_scale=True) for step in steps], dtype=float)
    targets = np.linspace(float(np.log(x[0])), float(np.log(x[-1])), max_ticks)

    tick_indices: List[int] = []
    log_x = np.log(x)
    for target in targets:
        idx = int(np.argmin(np.abs(log_x - target)))
        if idx not in tick_indices:
            tick_indices.append(idx)

    return [steps[idx] for idx in tick_indices]


def _training_step_axis_label(use_log_scale: bool) -> str:
    """Return the training-step x-axis label for linear or shifted-log plots."""
    label = _axis_label("training_steps")
    if use_log_scale:
        label += ", log(step + 1)"
    return label


def _training_curve_uses_log_y(
    metric_name: str,
    rows: Sequence[Mapping[str, Any]],
    mean_key: str,
) -> bool:
    """Return whether a training curve can safely use a log y-axis."""
    if not _prefers_training_log_y(metric_name):
        return False
    values = np.asarray([float(row[mean_key]) for row in rows if np.isfinite(float(row[mean_key]))], dtype=float)
    return bool(values.size > 0 and np.all(values > 0.0))


def _prefers_training_log_y(metric_name: str) -> bool:
    """Return whether a metric is visually clearer on a log training curve."""
    # NTK energy metrics are positive, but they are bounded fractions; keep their y-axis linear.
    return metric_name in TRAINING_LOG_Y_METRICS


def _make_nm_heatmaps_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> Optional[plt.Figure]:
    """Build beta-by-m heatmap panels for every `(n, training_steps)` pair."""
    mean_key = f"{metric_name}_mean"
    rows = [row for row in summary_rows if mean_key in row]
    if not rows:
        return None

    n_values = _unique_values(rows, "n")
    beta_values = _unique_values(rows, "beta")
    step_values = _unique_values(rows, "training_steps")
    if not n_values or not beta_values or not step_values:
        return None

    panels = []
    for n in n_values:
        for step in step_values:
            panel_rows = [
                row for row in rows
                if row["n"] == n and row["training_steps"] == step
            ]
            matrix, m_values, beta_panel_values = _heatmap_matrix(panel_rows, mean_key, "beta", "m")
            panels.append((n, step, matrix, m_values, beta_panel_values))

    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)].ravel()
        for _, _, matrix, _, _ in panels
        if np.isfinite(matrix).any()
    ]) if panels else np.asarray([])
    if finite_values.size == 0:
        return None

    use_log = bool(np.all(finite_values > 0)) and not _is_ntk_energy_metric(metric_name)
    transformed = [
        (n, step, np.log10(matrix) if use_log else matrix, m_values, beta_panel_values)
        for n, step, matrix, m_values, beta_panel_values in panels
    ]
    finite_transformed = np.concatenate([
        matrix[np.isfinite(matrix)].ravel()
        for _, _, matrix, _, _ in transformed
        if np.isfinite(matrix).any()
    ])
    vmin = float(finite_transformed.min())
    vmax = float(finite_transformed.max())

    title = _wrap_figure_text(f"{_metric_label(metric_name)} over beta and m", width=72)
    title_line_count = title.count("\n") + 1
    fig_width = max(3.3 * len(step_values) + 1.7, 5.4)
    fig_height = max(2.4 * len(n_values) + 1.4 + 0.18 * (title_line_count - 1), 4.2)
    fig, axes = plt.subplots(
        len(n_values),
        len(step_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )
    top = max(0.58, min(0.78, 0.95 - (1.0 + 0.18 * (title_line_count - 1)) / fig_height))
    fig.subplots_adjust(
        left=0.07,
        right=0.82,
        bottom=0.16,
        top=top,
        wspace=0.38,
        hspace=0.48,
    )
    n_value_offset_points = 34
    image = None
    axes_used = []
    for idx, (n, step, matrix, m_values, beta_panel_values) in enumerate(transformed):
        row_idx = idx // len(step_values)
        col_idx = idx % len(step_values)
        ax = axes[row_idx][col_idx]
        axes_used.append(ax)
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(m_values)))
        ax.set_xticklabels([_format_value(value) for value in m_values], rotation=45 if len(m_values) > 4 else 0)
        ax.set_yticks(range(len(beta_panel_values)))
        ax.set_yticklabels([_format_value(value) for value in beta_panel_values])
        if row_idx == 0:
            ax.set_title(_format_value(step), fontsize=9, pad=12)
        if row_idx == len(n_values) - 1:
            ax.set_xlabel(_axis_label("m"))
        if col_idx == 0:
            ax.set_ylabel(_axis_label("beta"), labelpad=8)
        if col_idx == len(step_values) - 1:
            ax.annotate(
                f"{_format_value(n)}",
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(n_value_offset_points, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
                annotation_clip=False,
            )
        ax.grid(False)

    if image is not None:
        label = f"log10({_metric_label(metric_name)} mean)" if use_log else f"{_metric_label(metric_name)} mean"
        colorbar = fig.colorbar(image, ax=axes_used, shrink=0.85, pad=0.04)
        _draw_metric_colorbar_label(colorbar.ax, label, metric_name)
    fig.canvas.draw()
    grid_left = axes[0][0].get_position().x0
    grid_right = axes[0][-1].get_position().x1
    grid_top = axes[0][0].get_position().y1
    grid_bottom = axes[-1][0].get_position().y0
    points_to_fig_x = 1.0 / (72.0 * fig.get_figwidth())
    points_to_fig_y = 1.0 / (72.0 * fig.get_figheight())
    n_value_x = grid_right + n_value_offset_points * points_to_fig_x
    n_title_x = n_value_x + 24 * points_to_fig_x
    n_beta_divider_x = grid_right + 26 * points_to_fig_x
    title_y = grid_top + 72 * points_to_fig_y
    step_label_y = grid_top + 48 * points_to_fig_y
    step_divider_y = grid_top + 26 * points_to_fig_y

    fig.text(0.5 * (grid_left + grid_right), title_y, title, ha="center", va="bottom", fontsize=13)
    fig.text(n_title_x, 0.5 * (grid_bottom + grid_top), _axis_label("n"), ha="center", va="center", rotation=90, fontsize=10)
    fig.add_artist(Line2D(
        [n_beta_divider_x, n_beta_divider_x],
        [grid_bottom, grid_top],
        transform=fig.transFigure,
        color="0.45",
        linewidth=0.8,
        linestyle=(0, (3, 3)),
    ))
    fig.text(0.5 * (grid_left + grid_right), step_label_y, _axis_label("training_steps"), ha="center", va="bottom", fontsize=10)
    fig.add_artist(Line2D(
        [grid_left, grid_right],
        [step_divider_y, step_divider_y],
        transform=fig.transFigure,
        color="0.45",
        linewidth=0.8,
        linestyle=(0, (3, 3)),
    ))
    return fig


def _make_final_test_error_vs_m_figure(
    summary_rows: Sequence[Mapping[str, Any]],
) -> Optional[plt.Figure]:
    """Build a final-step test-error summary plot appended to the loss PDF."""
    mean_key = "test_error_mean"
    std_key = "test_error_std"
    rows = [row for row in summary_rows if mean_key in row and np.isfinite(float(row[mean_key]))]
    if not rows:
        return None

    final_step = max(float(row["training_steps"]) for row in rows)
    rows = [row for row in rows if float(row["training_steps"]) == final_step]
    if not rows:
        return None

    beta_values = _unique_values(rows, "beta")
    n_values = _unique_values(rows, "n")
    m_values = _unique_values(rows, "m")
    if not beta_values or not n_values or not m_values:
        return None

    has_sample_size_legend = bool(n_values)
    fig_width = max(3.4 * len(beta_values) + (1.7 if has_sample_size_legend else 0.9), 5.2)
    fig_height = 3.8
    fig, axes = plt.subplots(
        1,
        len(beta_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
        constrained_layout=False,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(n_values), 1)))
    color_by_n = {n_value: colors[idx] for idx, n_value in enumerate(n_values)}
    seen_n_values = set()
    use_log_m = all(float(value) > 0.0 for value in m_values)

    for col_idx, beta in enumerate(beta_values):
        ax = axes[0][col_idx]
        panel_rows = [row for row in rows if row["beta"] == beta]
        for n in n_values:
            line_rows = sorted(
                [
                    row for row in panel_rows
                    if row["n"] == n and np.isfinite(float(row[mean_key]))
                ],
                key=lambda row: _sort_key(row["m"]),
            )
            if not line_rows:
                continue
            x = np.asarray([float(row["m"]) for row in line_rows], dtype=float)
            y = np.asarray([float(row[mean_key]) for row in line_rows], dtype=float)
            yerr = np.asarray([float(row.get(std_key, 0.0)) for row in line_rows], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=1.8,
                markersize=4.5,
                color=color_by_n[n],
                linestyle="-",
                label=f"n={_format_value(n)}",
            )
            seen_n_values.add(n)

        if use_log_m:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([float(value) for value in m_values])
        ax.set_xticklabels([_format_value(value) for value in m_values], rotation=30 if len(m_values) > 6 else 0)
        ax.set_title(f"beta={_format_value(beta)}", fontsize=9)
        ax.set_xlabel(_axis_label("m", log_scale=use_log_m))
        if col_idx == 0:
            ax.set_ylabel(_metric_label("test_error"))
        ax.grid(False)

    right = 0.74 if seen_n_values else 0.90
    fig.subplots_adjust(
        left=0.15,
        right=right,
        bottom=0.18,
        top=0.78,
        wspace=0.30,
    )

    if seen_n_values:
        n_handles = [
            Line2D([0], [0], color=color_by_n[n], marker="o", linestyle="-", linewidth=2.0, markersize=5)
            for n in n_values
            if n in seen_n_values
        ]
        n_labels = [f"n={_format_value(n)}" for n in n_values if n in seen_n_values]
        fig.legend(
            n_handles,
            n_labels,
            loc="upper left",
            bbox_to_anchor=(right + 0.04, 0.78),
            borderaxespad=0.0,
            frameon=False,
            title="sample size",
        )
    fig.suptitle(f"Final test error vs width (final step={_format_value(final_step)})", fontsize=13, y=0.965)
    return fig


# -------------------------------------------------------------------------- #
# ----------------------------- plot data helpers -------------------------- #
# -------------------------------------------------------------------------- #

def _heatmap_matrix(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    y_axis: str,
    x_axis: str,
) -> Tuple[np.ndarray, Tuple[Any, ...], Tuple[Any, ...]]:
    """Return a y-by-x matrix for one summary value column."""
    x_values = _unique_values(rows, x_axis)
    y_values = _unique_values(rows, y_axis)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for row in rows:
        if value_key in row:
            matrix[y_index[row[y_axis]], x_index[row[x_axis]]] = float(row[value_key])
    return matrix, x_values, y_values


# -------------------------------------------------------------------------- #
# ------------------------------- file utils ------------------------------- #
# -------------------------------------------------------------------------- #

def _save_figure_pdf(fig: Optional[plt.Figure], path: Path) -> bool:
    """Save one figure as a PDF and close it."""
    return _save_figures_pdf([fig], path)


def _save_figures_pdf(figures: Sequence[Optional[plt.Figure]], path: Path) -> bool:
    """Save one or more figures as a PDF and close every saved figure."""
    valid_figures = [fig for fig in figures if fig is not None]
    if not valid_figures:
        return False
    if len(valid_figures) == 1:
        valid_figures[0].savefig(path, bbox_inches="tight", pad_inches=PDF_PAD_INCHES)
        plt.close(valid_figures[0])
        return True
    _normalize_pdf_page_size(valid_figures)
    with PdfPages(path) as pdf:
        for fig in valid_figures:
            pdf.savefig(fig)
            plt.close(fig)
    return True


def _save_figures_pdf_equal_width(figures: Sequence[Optional[plt.Figure]], path: Path) -> bool:
    """Save figures as a multipage PDF after normalizing page dimensions."""
    valid_figures = [fig for fig in figures if fig is not None]
    if not valid_figures:
        return False
    _normalize_pdf_page_size(valid_figures)
    with PdfPages(path) as pdf:
        for fig in valid_figures:
            pdf.savefig(fig)
            plt.close(fig)
    return True


def _normalize_pdf_page_size(figures: Sequence[plt.Figure]) -> None:
    """Give every page in a multipage PDF the same explicit canvas size."""
    max_width = max(float(fig.get_figwidth()) for fig in figures)
    max_height = max(float(fig.get_figheight()) for fig in figures)
    for fig in figures:
        fig.set_size_inches(max_width, max_height, forward=True)


def _clear_plot_files(output_dir: Path, metric_names: Sequence[str] = ()) -> None:
    """Remove stale PDF plots from all generations of this experiment."""
    for path in (
        output_dir / "anisotropy_metrics.pdf",
        output_dir / "initialization_metrics.pdf",
        output_dir / "nm_metrics.pdf",
        output_dir / "ntk_spectrum_metrics.pdf",
        output_dir / "ntk_spectrum_metrics_nm_heatmaps.pdf",
        output_dir / "parameter_norms.pdf",
        output_dir / "parameter_norms_nm_heatmaps.pdf",
        output_dir / "ntk_eig_metrics.pdf",
        output_dir / "ntk_eig_metrics_nm_heatmaps.pdf",
        output_dir / "ntk_energy_metrics.pdf",
        output_dir / "ntk_energy_metrics_nm_heatmaps.pdf",
        output_dir / "ntk_label_energy_metrics.pdf",
        output_dir / "ntk_label_energy_metrics_nm_heatmaps.pdf",
        output_dir / "residual_ntk_alignment_metrics.pdf",
        output_dir / "residual_ntk_alignment_metrics_nm_heatmaps.pdf",
        output_dir / "loss_weighted_ntk_metrics.pdf",
        output_dir / "loss_weighted_ntk_metrics_nm_heatmaps.pdf",
        output_dir / "ntk_alignment_dynamics_terms.pdf",
        output_dir / "ntk_alignment_dynamics_terms_nm_heatmaps.pdf",
        output_dir / "ntk_drift_metrics.pdf",
        output_dir / "ntk_drift_metrics_nm_heatmaps.pdf",
        output_dir / "ntk_loss_product_metrics.pdf",
        output_dir / "ntk_loss_product_metrics_nm_heatmaps.pdf",
        output_dir / "ntk_loss_weighted_metrics.pdf",
        output_dir / "ntk_loss_weighted_metrics_nm_heatmaps.pdf",
    ):
        if path.exists():
            path.unlink()
    for metric_name in tuple(dict.fromkeys((*ALL_METRICS, *LEGACY_METRICS, *metric_names))):
        for axis in SWEEP_AXES:
            for path in (
                output_dir / f"{metric_name}_vs_{axis}.pdf",
                output_dir / f"at_init_stats_{metric_name}_vs_{axis}.pdf",
            ):
                if path.exists():
                    path.unlink()
        combined_path = output_dir / f"{metric_name}.pdf"
        if combined_path.exists():
            combined_path.unlink()
        for path in (
            output_dir / f"{metric_name}_training_curves.pdf",
            output_dir / f"{metric_name}_nm_heatmaps.pdf",
            output_dir / f"{metric_name}_initialization.pdf",
        ):
            if path.exists():
                path.unlink()
        for path in output_dir.glob(f"{metric_name}_heatmap_*.pdf"):
            path.unlink()
    for path in output_dir.glob("ntk_label_energy_top_*.pdf"):
        path.unlink()
    for path in output_dir.glob("ntk_residual_energy_top_*.pdf"):
        path.unlink()


# -------------------------------------------------------------------------- #
# -------------------------- labels and formatting ------------------------- #
# -------------------------------------------------------------------------- #

def _unique_values(rows: Sequence[Mapping[str, Any]], axis: Optional[str]) -> Tuple[Any, ...]:
    """Return sorted unique values for a sweep axis."""
    if axis is None:
        return (None,)
    return tuple(sorted({row[axis] for row in rows}, key=_sort_key))


def _wrap_figure_text(text: str, width: int = TRAINING_TITLE_WRAP_WIDTH) -> str:
    """Wrap figure-level text to at most two balanced lines."""
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= 2:
        return "\n".join(lines)

    words = text.split()
    if len(words) <= 1:
        return text
    total_chars = sum(len(word) for word in words) + len(words) - 1
    target = total_chars / 2.0
    best_idx = 1
    best_delta = float("inf")
    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        delta = abs(len(left) - target)
        if delta < best_delta:
            best_idx = idx
            best_delta = delta
    return " ".join(words[:best_idx]) + "\n" + " ".join(words[best_idx:])


def _metric_label(name: str) -> str:
    """Human-readable metric label for titles and axes."""
    normalized_labels = {
        "train_error": "train error",
        "test_error": "test error",
        "mean_preactivation_norm_normalized": "mean preactivation norm / sqrt(m)",
        "mean_output_grad_norm_fc2_normalized": "mean output grad norm fc2 / sqrt(m)",
        "empirical_loss_grad_norm_fc2_normalized": "empirical loss grad norm fc2 / sqrt(m)",
        "fc1_weight_fro_norm": "hidden layer Frobenius norm",
        "fc1_weight_fro_norm_normalized": "hidden layer Frobenius norm / sqrt(m)",
        "fc1_weight_spectral_norm": "hidden layer spectral norm",
        "fc1_weight_spectral_norm_normalized": "hidden layer spectral norm / (1 + sqrt(m/d_in))",
        "fc2_weight_euclidean_norm": "output layer Euclidean norm",
        "ntk_eig_min": "ntk eig min",
        "ntk_eig_max": "ntk eig max",
        "ntk_eig_mean": "ntk eig mean",
        "ntk_eig_median": "ntk eig median",
        "residual_ntk_alignment": "residual NTK alignment",
        "residual_ntk_alignment_over_ntk_eig_min": "residual NTK alignment / ntk eig min",
        "residual_ntk_alignment_over_ntk_eig_mean": "residual NTK alignment / ntk eig mean",
        "residual_ntk_alignment_over_ntk_eig_max": "residual NTK alignment / ntk eig max",
        "empirical_loss_times_residual_ntk_alignment": "empirical loss * residual NTK alignment",
        "empirical_loss_times_ntk_eig_min": "empirical loss * ntk eig min",
        "loss_weighted_residual_ntk_alignment": "loss-weighted residual NTK alignment",
        "loss_weighted_ntk_eig_min": "loss-weighted ntk eig min",
        "residual_ntk_alignment_residual_dynamics_term": "residual dynamics term",
        "residual_ntk_alignment_ntk_dynamics_term": "NTK dynamics term",
        "residual_initial_ntk_alignment": "residual initial NTK alignment",
        "residual_ntk_alignment_over_initial": "residual NTK alignment / initial",
        "residual_ntk_alignment_trace_normalized_over_initial": "trace-normalized residual NTK alignment / initial",
        "task_ntk_alignment": "task NTK alignment",
        "task_initial_ntk_alignment": "task initial NTK alignment",
        "task_ntk_alignment_over_initial": "task NTK alignment / initial",
        "task_ntk_alignment_trace_normalized_over_initial": "trace-normalized task NTK alignment / initial",
        "ntk_cos_dist": "NTK cosine distance from initialization",
        "ntk_rel_fro_dist": "NTK relative Frobenius distance from initialization",
    }
    if name in normalized_labels:
        return normalized_labels[name]
    if _is_ntk_label_energy_metric(name):
        return f"label energy top {name.rsplit('_', 1)[-1]}"
    if _is_ntk_residual_energy_metric(name):
        return f"residual energy top {name.rsplit('_', 1)[-1]}"
    return name.replace("_", " ")


def _is_ntk_label_energy_metric(name: str) -> bool:
    """Return whether a metric is a top-k NTK label-energy metric."""
    prefix = "ntk_label_energy_top_"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _is_ntk_residual_energy_metric(name: str) -> bool:
    """Return whether a metric is a top-k NTK residual-energy metric."""
    prefix = "ntk_residual_energy_top_"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _is_ntk_energy_metric(name: str) -> bool:
    """Return whether a metric is any top-k NTK energy metric."""
    return _is_ntk_label_energy_metric(name) or _is_ntk_residual_energy_metric(name)


def _uses_zero_reference_line(metric_name: str) -> bool:
    """Return whether plots for this metric should show y=0."""
    return metric_name in NTK_ALIGNMENT_DYNAMICS_GROUP_METRICS


def _uses_one_reference_line(metric_name: str) -> bool:
    """Return whether plots for this metric should show y=1."""
    return metric_name.endswith("_over_initial")


def _metric_label_parts(name: str) -> Tuple[str, str]:
    """Return the base metric label and red m-scaling suffix, if any."""
    m_scaled_suffixes = {
        "mean_preactivation_norm_normalized": " / sqrt(m)",
        "mean_output_grad_norm_fc2_normalized": " / sqrt(m)",
        "empirical_loss_grad_norm_fc2_normalized": " / sqrt(m)",
        "fc1_weight_fro_norm_normalized": " / sqrt(m)",
        "fc1_weight_spectral_norm_normalized": " / (1 + sqrt(m/d_in))",
    }
    suffix = m_scaled_suffixes.get(name, "")
    label = _metric_label(name)
    if suffix and label.endswith(suffix):
        return label[: -len(suffix)], suffix
    return label, ""


def _draw_metric_colorbar_label(colorbar_ax: plt.Axes, label: str, metric_name: str) -> None:
    """Draw a colorbar label, coloring m-related normalization red."""
    _, scaling_suffix = _metric_label_parts(metric_name)
    if not scaling_suffix or scaling_suffix not in label:
        colorbar_ax.set_ylabel(label)
        return

    colorbar_ax.set_ylabel(label.replace(scaling_suffix, ""), labelpad=24)
    colorbar_ax.text(
        3.2,
        0.5,
        scaling_suffix,
        transform=colorbar_ax.transAxes,
        color="red",
        rotation=90,
        ha="center",
        va="center",
        fontsize=plt.rcParams.get("axes.labelsize", "medium"),
        clip_on=False,
    )


def _axis_label(name: str, log_scale: bool = False) -> str:
    """Human-readable sweep-axis label."""
    if name == "synthetic_anisotropy_power":
        label = "anisotropy power"
    else:
        label = name.replace("_", " ")
    if log_scale:
        label += " (log scale)"
    return label


def _format_value(value: Any) -> str:
    """Format sweep values without unnecessary decimal noise."""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Sort numeric sweep values numerically and fall back to strings."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))
