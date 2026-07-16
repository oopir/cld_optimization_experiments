from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
import numpy as np

from ..training_stats.plotting import (
    _axis_label,
    _clear_plot_files,
    _draw_metric_colorbar_label,
    _format_value,
    _grouped_metric_names,
    _heatmap_matrix,
    _metric_label,
    _metric_label_parts,
    _save_figures_pdf,
    _save_figures_pdf_equal_width,
    _sort_key,
    _unique_values,
)

INITIALIZATION_FIXED_AXES = ("alpha", "beta", "training_steps", "synthetic_anisotropy_power")

__all__ = ["plot_initialization_summaries", "plot_summaries"]


# -------------------------------------------------------------------------- #
# ----------------------------- public entrypoints ------------------------- #
# -------------------------------------------------------------------------- #

def plot_initialization_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Create initialization/n-m/anisotropy plot outputs only."""
    paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_plot_files(output_dir, metric_names=config.plot_metrics)
    if not summary_rows:
        return paths

    if not _is_initialization_only(config):
        raise ValueError("plot_initialization_summaries requires one fixed training step.")

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


def plot_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Compatibility wrapper for the at-init plotting entrypoint."""
    return plot_initialization_summaries(
        summary_rows,
        config,
        output_dir,
        data_averaged_init_variability_rows=data_averaged_init_variability_rows,
        init_averaged_data_variability_rows=init_averaged_data_variability_rows,
    )


def _is_initialization_only(config: Any) -> bool:
    """Return True when the run has no training-step sweep."""
    return len(tuple(config.training_step_values or [])) == 1


def _has_anisotropy_sweep(config: Any) -> bool:
    """Return True when synthetic anisotropy power is an active sweep axis."""
    return len(tuple(config.synthetic_anisotropy_powers or [])) > 1


# -------------------------------------------------------------------------- #
# -------------------------- initialization plots -------------------------- #
# -------------------------------------------------------------------------- #

def _plot_initialization_only_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: Any,
    output_dir: Path,
    data_averaged_init_variability_rows: Optional[Sequence[Mapping[str, Any]]],
    init_averaged_data_variability_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Create the n/m initialization plots."""
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


# -------------------------------------------------------------------------- #
# ---------------------------- anisotropy plots ---------------------------- #
# -------------------------------------------------------------------------- #

def _plot_anisotropy_summaries(
    config: Any,
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


# -------------------------------------------------------------------------- #
# ----------------------------- page helpers ------------------------------- #
# -------------------------------------------------------------------------- #

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


def _single_metric_pages(values: Sequence[str]) -> List[Sequence[str]]:
    """Split values into one metric per page."""
    return [[value] for value in values]


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
        base_text = fig.text(0.0, 0.0, base_label, ha="left", va=va, fontsize=fontsize, alpha=0.0)
        suffix_text = fig.text(0.0, 0.0, scaling_suffix, ha="left", va=va, fontsize=fontsize, alpha=0.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        base_width = base_text.get_window_extent(renderer=renderer).width / fig.bbox.width
        suffix_width = suffix_text.get_window_extent(renderer=renderer).width / fig.bbox.width
        base_text.remove()
        suffix_text.remove()
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


# -------------------------------------------------------------------------- #
# ------------------------------- utilities -------------------------------- #
# -------------------------------------------------------------------------- #

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
