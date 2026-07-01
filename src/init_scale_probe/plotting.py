from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
import numpy as np

from .core import ALL_METRICS, SWEEP_AXES, InitScaleProbeConfig

LEGACY_METRICS = (
    "mean_loss",
    "mean_sample_loss_grad_norm",
    "mean_sample_loss_grad_norm_sq",
    "mean_sample_loss_grad_norm_fc1",
    "mean_sample_loss_grad_norm_sq_fc1",
    "mean_sample_loss_grad_norm_fc2",
    "mean_sample_loss_grad_norm_sq_fc2",
)

INITIALIZATION_FIXED_AXES = ("alpha", "beta", "training_steps", "synthetic_anisotropy_power")
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
GROUPED_METRIC_PDFS = (
    ("ntk_spectrum_metrics", NTK_SPECTRUM_GROUP_METRICS),
    ("ntk_energy_metrics", ("ntk_label_energy_top_", "ntk_residual_energy_top_")),
    ("residual_ntk_alignment_metrics", RESIDUAL_NTK_ALIGNMENT_GROUP_METRICS),
    ("loss_weighted_ntk_metrics", LOSS_WEIGHTED_NTK_GROUP_METRICS),
    ("ntk_alignment_dynamics_terms", NTK_ALIGNMENT_DYNAMICS_GROUP_METRICS),
)


# -------------------------------------------------------------------------- #
# ------------------------------- line plots ------------------------------- #
# -------------------------------------------------------------------------- #

def plot_probe_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Create all configured plot outputs from the already-aggregated summary rows."""
    paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_probe_plot_files(output_dir, metric_names=config.plot_metrics)
    if not summary_rows:
        return paths

    if _is_initialization_only(config):
        if not _has_anisotropy_sweep(config):
            paths.update(
                _plot_initialization_only_summaries(
                    summary_rows,
                    config,
                    output_dir,
                    data_averaged_init_variability_rows=data_averaged_init_variability_rows,
                    init_averaged_data_variability_rows=init_averaged_data_variability_rows,
                )
            )
        paths.update(
            _plot_anisotropy_summaries(
                config,
                output_dir,
                data_averaged_init_variability_rows=data_averaged_init_variability_rows,
                init_averaged_data_variability_rows=init_averaged_data_variability_rows,
            )
        )
        return paths

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

    paths.update(
        _plot_anisotropy_summaries(
            config,
            output_dir,
            data_averaged_init_variability_rows=data_averaged_init_variability_rows,
            init_averaged_data_variability_rows=init_averaged_data_variability_rows,
        )
    )
    return paths


def _is_initialization_only(config: InitScaleProbeConfig) -> bool:
    """Return True when the run has no training-step sweep."""
    return len(tuple(config.training_step_values or [])) == 1


def _has_anisotropy_sweep(config: InitScaleProbeConfig) -> bool:
    """Return True when synthetic anisotropy power is an active sweep axis."""
    return len(tuple(config.synthetic_anisotropy_powers or [])) > 1


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
    return {file_stem: names for file_stem, names in groups.items() if names}


def _plot_grouped_training_metric_pdfs(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
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
        path = output_dir / f"{file_stem}.pdf"
        if _save_figures_pdf_equal_width(curve_figures, path):
            paths[f"plot_{file_stem}"] = path
        if config.plot_heatmaps:
            heatmap_path = output_dir / f"{file_stem}_nm_heatmaps.pdf"
            if _save_figures_pdf_equal_width(heatmap_figures, heatmap_path):
                paths[f"plot_{file_stem}_nm_heatmaps"] = heatmap_path
    return paths


def _save_grouped_metric_pdfs(
    grouped_metrics: Mapping[str, Sequence[str]],
    output_dir: Path,
    figure_factory: Callable[[Sequence[str]], Sequence[Optional[plt.Figure]]],
) -> Dict[str, Path]:
    """Save grouped metric-family PDFs using one shared figure factory."""
    paths: Dict[str, Path] = {}
    for file_stem, metric_names in grouped_metrics.items():
        path = output_dir / f"{file_stem}.pdf"
        if _save_figures_pdf_equal_width(figure_factory(metric_names), path):
            paths[f"plot_{file_stem}"] = path
    return paths


def _plot_initialization_only_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]],
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Create the n/m initialization plots, preserving trajectory plots elsewhere."""
    paths: Dict[str, Path] = {}
    data_avg_init_var_rows = list(data_averaged_init_variability_rows or summary_rows)
    init_avg_data_var_rows = list(init_averaged_data_variability_rows or summary_rows)
    if not data_avg_init_var_rows or not init_avg_data_var_rows:
        return paths

    if config.plot_format in {"combined", "both"}:
        path = output_dir / "nm_metrics.pdf"
        figures = _make_initialization_metrics_figures(
            data_avg_init_var_rows,
            init_avg_data_var_rows,
            config.plot_metrics,
            plot_heatmaps=config.plot_heatmaps,
            title_suffix=_label_state_from_bool(config.random_labels),
        )
        if _save_figures_pdf(figures, path):
            paths["plot_nm_metrics"] = path
        paths.update(
            _save_grouped_metric_pdfs(
                _grouped_metric_names(config.plot_metrics),
                output_dir,
                lambda metric_names: _make_initialization_metrics_figures(
                    data_avg_init_var_rows,
                    init_avg_data_var_rows,
                    metric_names,
                    plot_heatmaps=config.plot_heatmaps,
                    title_suffix=_label_state_from_bool(config.random_labels),
                ),
            )
        )

    for metric_name in config.plot_metrics:
        if f"{metric_name}_mean" not in data_avg_init_var_rows[0] or f"{metric_name}_mean" not in init_avg_data_var_rows[0]:
            continue
        if config.plot_format in {"individual", "both"}:
            path = output_dir / f"{metric_name}.pdf"
            figures = _make_initialization_metrics_figures(
                data_avg_init_var_rows,
                init_avg_data_var_rows,
                [metric_name],
                plot_heatmaps=config.plot_heatmaps,
                title_suffix=_label_state_from_bool(config.random_labels),
            )
            if _save_figures_pdf(figures, path):
                paths[f"plot_{metric_name}"] = path

    return paths


def _make_initialization_metrics_figures(
    data_avg_init_var_rows: Sequence[Mapping[str, Any]],
    init_avg_data_var_rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    plot_heatmaps: bool,
    title_suffix: str = "",
) -> List[plt.Figure]:
    """
    Build initialization-only pages with one metric per row.

    With heatmaps enabled, each metric row contains:
    init mean line, init std heatmap, data mean line, data std heatmap.
    Without heatmaps, each row contains only the two line panels.
    """
    available_metric_names = _available_metric_names(data_avg_init_var_rows, init_avg_data_var_rows, metric_names)
    if not available_metric_names:
        return []

    shared_ylims = {
        metric_name: _metric_ylim([data_avg_init_var_rows, init_avg_data_var_rows], metric_name)
        for metric_name in available_metric_names
    }
    metric_pages = _single_metric_pages(available_metric_names)
    fixed_axes, fixed_keys = _varying_fixed_axes_and_keys(data_avg_init_var_rows, init_avg_data_var_rows, INITIALIZATION_FIXED_AXES)

    figures: List[plt.Figure] = []
    for fixed_key in fixed_keys:
        init_panel_rows = _rows_matching_fixed_axes(data_avg_init_var_rows, fixed_axes, fixed_key)
        data_panel_rows = _rows_matching_fixed_axes(init_avg_data_var_rows, fixed_axes, fixed_key)
        if not init_panel_rows or not data_panel_rows:
            continue

        for page_idx, page_metric_names in enumerate(metric_pages):
            fig, axes = _make_metric_page(len(page_metric_names), plot_heatmaps)
            for row_idx, metric_name in enumerate(page_metric_names):
                _draw_metric_row(
                    fig,
                    axes[row_idx],
                    init_panel_rows,
                    data_panel_rows,
                    metric_name,
                    ylim=shared_ylims[metric_name],
                    x_axis="m",
                    plot_heatmaps=plot_heatmaps,
                )
            _finish_metric_page(
                fig,
                axes,
                init_panel_rows,
                page_metric_names,
                title_prefix="Initialization metrics over n and m",
                page_idx=page_idx,
                page_count=len(metric_pages),
                fixed_axes=fixed_axes,
                fixed_key=fixed_key,
                title_suffix=title_suffix,
            )
            figures.append(fig)

    return figures


def _available_metric_names(
    data_avg_init_var_rows: Sequence[Mapping[str, Any]],
    init_avg_data_var_rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> List[str]:
    """Return requested metrics available in both init- and data-seed summaries."""
    return [
        metric_name for metric_name in metric_names
        if f"{metric_name}_mean" in data_avg_init_var_rows[0] and f"{metric_name}_mean" in init_avg_data_var_rows[0]
    ]


def _make_metric_page(
    row_count: int,
    plot_heatmaps: bool,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create one metric page with stable spacing."""
    n_cols = 4 if plot_heatmaps else 2
    fig_width = 17.2 if plot_heatmaps else 9
    fig_height = max(3.35 * row_count + 1.35, 4.4)
    fig, axes = plt.subplots(
        row_count,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.94,
        bottom=0.25 if row_count == 1 else 0.14,
        top=0.72 if row_count == 1 else 0.81,
        wspace=0.42 if plot_heatmaps else 0.3,
        hspace=1.05,
    )
    return fig, axes


def _draw_metric_row(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    data_avg_init_var_rows: Sequence[Mapping[str, Any]],
    init_avg_data_var_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    ylim: Optional[Tuple[float, float]],
    x_axis: str,
    plot_heatmaps: bool,
) -> None:
    """Draw one init/data metric row, optionally with std heatmaps."""
    _draw_metric_line_panel(
        axes[0],
        data_avg_init_var_rows,
        metric_name,
        x_axis=x_axis,
        title="mean over all seeds; \nSD of data-averaged values across init seeds",
        # title="Avg over data seeds; SD across init seeds",
        ylim=ylim,
        ylabel="mean value",
        show_legend=False,
    )
    if plot_heatmaps:
        _draw_variability_heatmap_panel(
            fig,
            axes[1],
            data_avg_init_var_rows,
            metric_name,
            value_kind="std",
            x_axis=x_axis,
            title="Std over init",
            colorbar_label=None,
        )
        _draw_metric_line_panel(
            axes[2],
            init_avg_data_var_rows,
            metric_name,
            x_axis=x_axis,
            title="mean over all seeds; \nSD of init-averaged values across data seeds",
            # title="Avg over init seeds; SD across data seeds",
            ylim=ylim,
            ylabel="mean value",
            show_legend=False,
        )
        _draw_variability_heatmap_panel(
            fig,
            axes[3],
            init_avg_data_var_rows,
            metric_name,
            value_kind="std",
            x_axis=x_axis,
            title="Std over data",
            colorbar_label=None,
        )
    else:
        _draw_metric_line_panel(
            axes[1],
            init_avg_data_var_rows,
            metric_name,
            x_axis=x_axis,
            title="mean over all seeds; \nSD of init-averaged values across data seeds",
            # title="Avg over init seeds; SD across data seeds",
            ylim=ylim,
            ylabel="mean value",
            show_legend=False,
        )


def _finish_metric_page(
    fig: plt.Figure,
    axes: np.ndarray,
    data_avg_init_var_rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    title_prefix: str,
    page_idx: int,
    page_count: int,
    fixed_axes: Sequence[str],
    fixed_key: Sequence[Any],
    title_suffix: str = "",
) -> None:
    """Add page-level labels, row titles, and row separators."""
    title = title_prefix
    if page_count > 1:
        title += f" ({page_idx + 1}/{page_count})"
    subtitle = _fixed_axes_subtitle(fixed_axes, fixed_key)
    label_state = title_suffix or _label_state_subtitle(data_avg_init_var_rows)
    if label_state:
        subtitle = ", ".join(part for part in (subtitle, label_state) if part)
    if subtitle:
        title += f" ({subtitle})"
    fig.suptitle(title, fontsize=13)
    _draw_sample_size_legend(fig, data_avg_init_var_rows)
    fig.canvas.draw()
    for row_idx, metric_name in enumerate(metric_names):
        _draw_metric_row_title(fig, axes[row_idx], metric_name)
    _draw_metric_row_separators(fig, axes)


def _draw_metric_line_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    x_axis: str,
    title: str,
    ylim: Optional[Tuple[float, float]],
    ylabel: Optional[str] = None,
    show_legend: bool = True,
) -> None:
    """Draw metric vs one sweep axis with n as the visible series."""
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    n_values = _unique_values(rows, "n")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(n_values), 1)))
    color_by_n = {n_value: colors[idx] for idx, n_value in enumerate(n_values)}

    for n in n_values:
        line_rows = sorted(
            [
                row for row in rows
                if row["n"] == n and np.isfinite(float(row[mean_key]))
            ],
            key=lambda row: _sort_key(row[x_axis]),
        )
        if not line_rows:
            continue
        x = np.asarray([float(row[x_axis]) for row in line_rows], dtype=float)
        y = np.asarray([float(row[mean_key]) for row in line_rows], dtype=float)
        yerr = np.asarray([float(row.get(std_key, 0.0)) for row in line_rows], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=color_by_n[n],
            label=f"n={_format_value(n)}",
        )

    x_values = _unique_values(rows, x_axis)
    x_is_log_scale = bool(x_axis == "m" and x_values and all(float(value) > 0 for value in x_values))
    if x_is_log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks([float(value) for value in x_values])
    ax.set_xticklabels([_format_value(value) for value in x_values])
    ax.set_xlabel(_axis_label(x_axis, log_scale=x_is_log_scale))
    ax.set_ylabel(ylabel or _metric_label(metric_name))
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    if show_legend and len(n_values) > 1:
        ax.legend(frameon=False, fontsize=8, title="sample size n")


def _plot_anisotropy_summaries(
    config: InitScaleProbeConfig,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]],
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Create a single multipage PDF of metrics over synthetic anisotropy power."""
    data_avg_init_var_rows = list(data_averaged_init_variability_rows or [])
    init_avg_data_var_rows = list(init_averaged_data_variability_rows or [])
    if not data_avg_init_var_rows or not init_avg_data_var_rows:
        return {}
    powers = _unique_values([*data_avg_init_var_rows, *init_avg_data_var_rows], "synthetic_anisotropy_power")
    if len(powers) <= 1:
        return {}

    figures = _make_anisotropy_metrics_figures(
        data_avg_init_var_rows,
        init_avg_data_var_rows,
        config.plot_metrics,
        config.plot_heatmaps,
        title_suffix=_label_state_from_bool(config.random_labels),
    )
    path = output_dir / "anisotropy_metrics.pdf"
    paths: Dict[str, Path] = {}
    if _save_figures_pdf(figures, path):
        paths["plot_anisotropy_metrics"] = path
    paths.update(
        _save_grouped_metric_pdfs(
            _grouped_metric_names(config.plot_metrics),
            output_dir,
            lambda metric_names: _make_anisotropy_metrics_figures(
                data_avg_init_var_rows,
                init_avg_data_var_rows,
                metric_names,
                config.plot_heatmaps,
                title_suffix=_label_state_from_bool(config.random_labels),
            ),
        )
    )
    return paths


def _make_anisotropy_metrics_figures(
    data_avg_init_var_rows: Sequence[Mapping[str, Any]],
    init_avg_data_var_rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    plot_heatmaps: bool,
    title_suffix: str = "",
) -> List[plt.Figure]:
    """Build anisotropy pages with one metric per row."""
    available_metric_names = _available_metric_names(data_avg_init_var_rows, init_avg_data_var_rows, metric_names)
    if not available_metric_names:
        return []

    fixed_axes = ("m",)
    fixed_keys = sorted(
        {
            tuple(row[axis] for axis in fixed_axes)
            for row in [*data_avg_init_var_rows, *init_avg_data_var_rows]
        },
        key=lambda values: tuple(_sort_key(value) for value in values),
    )
    metric_pages = _single_metric_pages(available_metric_names)
    shared_ylims = {
        metric_name: _metric_ylim([data_avg_init_var_rows, init_avg_data_var_rows], metric_name)
        for metric_name in available_metric_names
    }

    figures: List[plt.Figure] = []
    for fixed_key in fixed_keys:
        init_panel_rows = _rows_matching_fixed_axes(data_avg_init_var_rows, fixed_axes, fixed_key)
        data_panel_rows = _rows_matching_fixed_axes(init_avg_data_var_rows, fixed_axes, fixed_key)
        if not init_panel_rows or not data_panel_rows:
            continue

        for page_idx, page_metric_names in enumerate(metric_pages):
            fig, axes = _make_metric_page(len(page_metric_names), plot_heatmaps)
            for row_idx, metric_name in enumerate(page_metric_names):
                _draw_metric_row(
                    fig,
                    axes[row_idx],
                    init_panel_rows,
                    data_panel_rows,
                    metric_name,
                    ylim=shared_ylims[metric_name],
                    x_axis="synthetic_anisotropy_power",
                    plot_heatmaps=plot_heatmaps,
                )
            _finish_metric_page(
                fig,
                axes,
                init_panel_rows,
                page_metric_names,
                title_prefix="Initialization metrics over n and anisotropy power",
                page_idx=page_idx,
                page_count=len(metric_pages),
                fixed_axes=fixed_axes,
                fixed_key=fixed_key,
                title_suffix=title_suffix,
            )
            figures.append(fig)

    return figures


def _draw_variability_heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    value_kind: str,
    x_axis: str,
    title: str,
    colorbar_label: Optional[str],
) -> None:
    """Draw an n-by-x heatmap for mean or std summary values."""
    value_key = f"{metric_name}_{value_kind}"
    matrix, x_values, n_values = _heatmap_matrix(rows, value_key, "n", x_axis)
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        ax.set_axis_off()
        ax.set_title(f"{title} (no finite values)", fontsize=10)
        return

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0 if value_kind == "std" and np.all(finite_values >= 0) else float(finite_values.min()),
        vmax=float(finite_values.max()),
    )
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([_format_value(value) for value in x_values], rotation=45 if len(x_values) > 4 else 0)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([_format_value(value) for value in n_values])
    ax.set_xlabel(_axis_label(x_axis))
    ax.set_ylabel(_axis_label("n"))
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    if colorbar_label:
        _draw_metric_colorbar_label(colorbar.ax, colorbar_label, metric_name)


def _axis_varies(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
    axis: str,
) -> bool:
    """Return whether an axis varies across either of two row sets."""
    values = {row[axis] for row in first_rows if axis in row}
    values.update(row[axis] for row in second_rows if axis in row)
    return len(values) > 1


def _varying_fixed_axes_and_keys(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
    candidate_axes: Sequence[str],
) -> Tuple[Tuple[str, ...], List[Tuple[Any, ...]]]:
    """Return varying fixed axes and their sorted fixed-value page keys."""
    fixed_axes = tuple(
        axis for axis in candidate_axes
        if _axis_varies(first_rows, second_rows, axis)
    )
    fixed_keys = sorted(
        {
            tuple(row[axis] for axis in fixed_axes)
            for row in [*first_rows, *second_rows]
        },
        key=lambda values: tuple(_sort_key(value) for value in values),
    ) or [()]
    return fixed_axes, fixed_keys


def _rows_matching_fixed_axes(
    rows: Sequence[Mapping[str, Any]],
    fixed_axes: Sequence[str],
    fixed_key: Sequence[Any],
) -> List[Mapping[str, Any]]:
    """Filter rows to one fixed-axis page of the initialization plot."""
    if not fixed_axes:
        return list(rows)
    return [
        row for row in rows
        if tuple(row[axis] for axis in fixed_axes) == tuple(fixed_key)
    ]


def _fixed_axes_subtitle(fixed_axes: Sequence[str], fixed_key: Sequence[Any]) -> str:
    """Format compact fixed-axis values for multipage titles."""
    return ", ".join(
        f"{_axis_label(axis)}={_format_value(value)}"
        for axis, value in zip(fixed_axes, fixed_key)
    )


def _label_state_subtitle(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact title note for random-label state when present."""
    if not rows or "random_labels" not in rows[0]:
        return ""
    values = {bool(row["random_labels"]) for row in rows if "random_labels" in row}
    if values == {True}:
        return "random labels"
    if values == {False}:
        return "true labels"
    return "mixed label states"


def _label_state_from_bool(random_labels: bool) -> str:
    """Return a compact title note for the configured label state."""
    return "random labels" if random_labels else "true labels"


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

    fig_width = max(3.4 * len(beta_values) + 1.0, 4.6)
    fig_height = max(2.25 * len(n_values) + 1.0, 3.4)
    fig, axes = plt.subplots(
        len(n_values),
        len(beta_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
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
                    np.asarray([float(row["training_steps"]) for row in line_rows], dtype=float),
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
            ax.set_xticks([float(value) for value in step_values])
            ax.set_xticklabels([_format_value(value) for value in step_values])
            ax.grid(False)
            if row_idx == len(n_values) - 1:
                ax.set_xlabel(_axis_label("training_steps"))
            if col_idx == 0:
                ax.set_ylabel(f"n={_format_value(n)}", rotation=0, labelpad=28, va="center")
            if _uses_zero_reference_line(metric_name):
                ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--", alpha=0.8)

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
            loc="outside right upper",
            frameon=False,
            title="width",
        )
    fig.suptitle(f"{_metric_label(metric_name)} vs training steps", fontsize=13)
    fig.text(0.01, 0.5, _metric_label(metric_name), va="center", rotation="vertical", fontsize=10)
    return fig


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

    fig_width = max(3.3 * len(step_values) + 1.0, 4.6)
    fig_height = max(2.4 * len(n_values) + 0.9, 3.8)
    fig, axes = plt.subplots(
        len(n_values),
        len(step_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    n_value_offset_points = -70
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
            ax.annotate(
                f"{_format_value(n)}",
                xy=(0.0, 0.5),
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
        colorbar = fig.colorbar(image, ax=axes_used, shrink=0.85)
        _draw_metric_colorbar_label(colorbar.ax, label, metric_name)
    fig.canvas.draw()
    grid_left = axes[0][0].get_position().x0
    grid_right = axes[0][-1].get_position().x1
    grid_top = axes[0][0].get_position().y1
    grid_bottom = axes[-1][0].get_position().y0
    points_to_fig_x = 1.0 / (72.0 * fig.get_figwidth())
    points_to_fig_y = 1.0 / (72.0 * fig.get_figheight())
    n_value_x = grid_left + n_value_offset_points * points_to_fig_x
    n_title_x = n_value_x - 24 * points_to_fig_x
    n_beta_divider_x = grid_left - 52 * points_to_fig_x
    title_y = grid_top + 72 * points_to_fig_y
    step_label_y = grid_top + 48 * points_to_fig_y
    step_divider_y = grid_top + 26 * points_to_fig_y

    fig.text(0.5 * (grid_left + grid_right), title_y, f"{_metric_label(metric_name)} over beta and m", ha="center", va="bottom", fontsize=13)
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


def _metric_ylim(
    row_sets: Sequence[Sequence[Mapping[str, Any]]],
    metric_name: str,
) -> Optional[Tuple[float, float]]:
    """Return shared linear y-limits for all line pages of a metric."""
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    lows = []
    highs = []
    for rows in row_sets:
        for row in rows:
            if mean_key not in row:
                continue
            mean = float(row[mean_key])
            std = float(row.get(std_key, 0.0))
            if math.isfinite(mean) and math.isfinite(std):
                lows.append(mean - std)
                highs.append(mean + std)
    if not lows:
        return None
    low = min(lows)
    high = max(highs)
    if low == high:
        pad = 0.05 * abs(low) if low != 0 else 1.0
        return low - pad, high + pad
    pad = 0.05 * (high - low)
    return low - pad, high + pad


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
# ------------------------------- utilities -------------------------------- #
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
        valid_figures[0].savefig(path, bbox_inches="tight")
        plt.close(valid_figures[0])
        return True
    with PdfPages(path) as pdf:
        for fig in valid_figures:
            pdf.savefig(fig)
            plt.close(fig)
    return True


def _save_figures_pdf_equal_width(figures: Sequence[Optional[plt.Figure]], path: Path) -> bool:
    """Save figures as a multipage PDF after normalizing page widths."""
    valid_figures = [fig for fig in figures if fig is not None]
    if not valid_figures:
        return False
    max_width = max(float(fig.get_figwidth()) for fig in valid_figures)
    for fig in valid_figures:
        fig.set_size_inches(max_width, fig.get_figheight(), forward=True)
    with PdfPages(path) as pdf:
        for fig in valid_figures:
            pdf.savefig(fig)
            plt.close(fig)
    return True


def _clear_probe_plot_files(output_dir: Path, metric_names: Sequence[str] = ()) -> None:
    """Remove stale PDF plots from all generations of this probe."""
    for path in (
        output_dir / "anisotropy_metrics.pdf",
        output_dir / "initialization_metrics.pdf",
        output_dir / "nm_metrics.pdf",
        output_dir / "ntk_spectrum_metrics.pdf",
        output_dir / "ntk_spectrum_metrics_nm_heatmaps.pdf",
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
                output_dir / f"init_scale_{metric_name}_vs_{axis}.pdf",
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


def _unique_values(rows: Sequence[Mapping[str, Any]], axis: Optional[str]) -> Tuple[Any, ...]:
    """Return sorted unique values for a sweep axis."""
    if axis is None:
        return (None,)
    return tuple(sorted({row[axis] for row in rows}, key=_sort_key))


def _single_metric_pages(values: Sequence[str]) -> List[Sequence[str]]:
    """Split values into one metric per page."""
    return [[value] for value in values]


def _metric_label(name: str) -> str:
    """Human-readable metric label for titles and axes."""
    normalized_labels = {
        "mean_preactivation_norm_normalized": "mean preactivation norm / sqrt(m)",
        "mean_output_grad_norm_fc2_normalized": "mean output grad norm fc2 / sqrt(m)",
        "empirical_loss_grad_norm_fc2_normalized": "empirical loss grad norm fc2 / sqrt(m)",
        "fc1_weight_fro_norm_normalized": "fc1 weight fro norm / sqrt(m)",
        "fc1_weight_spectral_norm_normalized": "fc1 weight spectral norm / (1 + sqrt(m/d_in))",
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
    }
    if name in normalized_labels:
        return normalized_labels[name]
    if _is_ntk_label_energy_metric(name):
        return f"label energy top {name.rsplit('_', 1)[-1]}"
    if _is_ntk_residual_energy_metric(name):
        return f"residual energy top {name.rsplit('_', 1)[-1]}"
    return name.replace("_", " ")


def _is_ntk_label_energy_metric(name: str) -> bool:
    prefix = "ntk_label_energy_top_"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _is_ntk_residual_energy_metric(name: str) -> bool:
    prefix = "ntk_residual_energy_top_"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _is_ntk_energy_metric(name: str) -> bool:
    return _is_ntk_label_energy_metric(name) or _is_ntk_residual_energy_metric(name)


def _uses_zero_reference_line(metric_name: str) -> bool:
    return metric_name in NTK_ALIGNMENT_DYNAMICS_GROUP_METRICS


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


def _draw_metric_row_title(
    fig: plt.Figure,
    row_axes: Sequence[plt.Axes],
    metric_name: str,
) -> None:
    """Draw one metric title above a row of subplot axes."""
    visible_axes = [ax for ax in row_axes if ax.get_visible()]
    if not visible_axes:
        return

    left = min(ax.get_position().x0 for ax in visible_axes)
    right = max(ax.get_position().x1 for ax in visible_axes)
    top = max(ax.get_position().y1 for ax in visible_axes)
    x = 0.5 * (left + right)
    y = min(top + 0.115, 0.925)
    _draw_metric_text(fig, x, y, metric_name, ha="center", va="bottom", fontsize=13)


def _draw_metric_row_separators(fig: plt.Figure, axes: np.ndarray) -> None:
    """Draw dashed horizontal separators between metric rows."""
    if axes.shape[0] <= 1:
        return

    for row_idx in range(axes.shape[0] - 1):
        upper_axes = [ax for ax in axes[row_idx] if ax.get_visible()]
        lower_axes = [ax for ax in axes[row_idx + 1] if ax.get_visible()]
        if not upper_axes or not lower_axes:
            continue

        left = min(
            min(ax.get_position().x0 for ax in upper_axes),
            min(ax.get_position().x0 for ax in lower_axes),
        )
        right = max(
            max(ax.get_position().x1 for ax in upper_axes),
            max(ax.get_position().x1 for ax in lower_axes),
        )
        upper_bottom = min(ax.get_position().y0 for ax in upper_axes)
        lower_top = max(ax.get_position().y1 for ax in lower_axes)
        y = 0.5 * (upper_bottom + lower_top)
        fig.add_artist(Line2D(
            [left, right],
            [y, y],
            transform=fig.transFigure,
            color="0.55",
            linewidth=0.8,
            linestyle=(0, (4, 4)),
        ))


def _draw_sample_size_legend(fig: plt.Figure, rows: Sequence[Mapping[str, Any]]) -> None:
    """Draw one prominent sample-size legend for an initialization page."""
    n_values = _unique_values(rows, "n")
    if len(n_values) <= 1:
        return

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(n_values), 1)))
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[idx],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=7.0,
        )
        for idx, _ in enumerate(n_values)
    ]
    labels = [f"n={_format_value(n)}" for n in n_values]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(labels),
        frameon=False,
        title="sample size",
        fontsize=9,
        title_fontsize=9,
        handlelength=2.0,
        columnspacing=1.8,
    )


def _draw_metric_text(
    fig: plt.Figure,
    x: float,
    y: float,
    metric_name: str,
    ha: str,
    va: str,
    fontsize: float,
) -> None:
    """Draw a metric label with the m-scaling suffix colored red."""
    base_label, scaling_suffix = _metric_label_parts(metric_name)
    if not scaling_suffix:
        fig.text(x, y, base_label, ha=ha, va=va, fontsize=fontsize)
        return

    if ha == "center":
        base_probe = fig.text(0.0, 0.0, base_label, ha="left", va=va, fontsize=fontsize, alpha=0.0)
        suffix_probe = fig.text(0.0, 0.0, scaling_suffix, ha="left", va=va, fontsize=fontsize, alpha=0.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        base_width = base_probe.get_window_extent(renderer=renderer).width / fig.bbox.width
        suffix_width = suffix_probe.get_window_extent(renderer=renderer).width / fig.bbox.width
        base_probe.remove()
        suffix_probe.remove()
        start_x = x - 0.5 * (base_width + suffix_width)
        fig.text(start_x, y, base_label, ha="left", va=va, fontsize=fontsize)
        fig.text(start_x + base_width, y, scaling_suffix, ha="left", va=va, fontsize=fontsize, color="red")
    else:
        text = fig.text(x, y, base_label, ha=ha, va=va, fontsize=fontsize)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width = text.get_window_extent(renderer=renderer).width / fig.bbox.width
        suffix_x = x + width if ha == "left" else x
        fig.text(suffix_x, y, scaling_suffix, ha="left", va=va, fontsize=fontsize, color="red")


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
