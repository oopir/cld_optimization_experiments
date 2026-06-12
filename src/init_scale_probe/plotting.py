from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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


# -------------------------------------------------------------------------- #
# ------------------------------- plot plans ------------------------------- #
# -------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LinePlotPlan:
    """Layout decision for one metric-vs-axis plot.

    `line_axes` are encoded by line color/marker and appear in the legend.
    `facet_axis` is encoded by separate panels when there would otherwise be
    too many line groups. Sweep axes that are constant in the reported summary
    rows are omitted from both places, keeping labels focused on what varies.
    """

    axis: str
    line_axes: Tuple[str, ...]
    facet_axis: Optional[str]
    facet_values: Tuple[Any, ...]
    rows: int
    cols: int


def _line_plot_plan(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    axis: str,
) -> LinePlotPlan:
    """Choose line-group and facet variables for a single sweep-axis plot."""
    varying_other_axes = tuple(
        name for name in SWEEP_AXES
        if name != axis and len(_unique_values(summary_rows, name)) > 1
    )
    line_group_count = math.prod(
        len(_unique_values(summary_rows, name)) for name in varying_other_axes
    ) if varying_other_axes else 1

    facet_axis = None
    if (
        config.plot_facets == "auto"
        and line_group_count > config.plot_facet_threshold
        and varying_other_axes
    ):
        facet_axis = max(
            varying_other_axes,
            key=lambda name: (len(_unique_values(summary_rows, name)), name),
        )

    line_axes = tuple(name for name in varying_other_axes if name != facet_axis)
    facet_values = _unique_values(summary_rows, facet_axis) if facet_axis else (None,)
    rows, cols = _grid_shape(len(facet_values))
    return LinePlotPlan(
        axis=axis,
        line_axes=line_axes,
        facet_axis=facet_axis,
        facet_values=facet_values,
        rows=rows,
        cols=cols,
    )


def _grid_shape(count: int) -> Tuple[int, int]:
    """Return a compact panel grid with no hard-coded sweep-axis assumptions."""
    if count <= 1:
        return 1, 1
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    return rows, cols


def _varying_axes(summary_rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return sweep axes that vary in the reported summary rows."""
    return tuple(name for name in SWEEP_AXES if len(_unique_values(summary_rows, name)) > 1)


def _line_plot_axes(summary_rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return varying line-plot x-axes, temporarily excluding n and non-finite axes."""
    return tuple(
        axis
        for axis in _varying_axes(summary_rows)
        if axis != "n" and _axis_has_finite_values(summary_rows, axis)
    )


# -------------------------------------------------------------------------- #
# ------------------------------- line plots ------------------------------- #
# -------------------------------------------------------------------------- #

def plot_probe_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    output_dir: Path,
    data_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Create all configured plot outputs from the already-aggregated summary rows."""
    paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_probe_plot_files(output_dir)
    if not summary_rows:
        return paths

    varying_axes = _varying_axes(summary_rows)
    heatmap_axes = varying_axes if len(varying_axes) <= 3 else ()
    for metric_name in config.plot_metrics:
        if f"{metric_name}_mean" not in summary_rows[0]:
            continue

        line_axes = _line_plot_axes(summary_rows)
        data_seed_rows = _metric_data_seed_rows(data_seed_summary_rows, metric_name)
        data_seed_axes = _line_plot_axes(data_seed_rows) if data_seed_rows else ()

        if (line_axes or data_seed_axes) and config.plot_format in {"combined", "both"}:
            path = output_dir / f"{metric_name}.pdf"
            _plot_combined_metric(
                summary_rows,
                config,
                metric_name,
                line_axes,
                path,
                heatmap_axes=heatmap_axes,
                data_seed_summary_rows=data_seed_rows,
                data_seed_axes=data_seed_axes,
            )
            paths[f"plot_{metric_name}"] = path

        if line_axes and config.plot_format in {"individual", "both"}:
            for axis in line_axes:
                path = output_dir / f"{metric_name}_vs_{axis}.pdf"
                _plot_metric_axis(summary_rows, config, metric_name, axis, path)
                paths[f"plot_{metric_name}_vs_{axis}"] = path

        if config.plot_heatmaps and len(heatmap_axes) >= 2 and config.plot_format in {"individual", "both"}:
            paths.update(_plot_metric_heatmaps(summary_rows, metric_name, heatmap_axes, output_dir))

    return paths


def _metric_data_seed_rows(
    data_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]],
    metric_name: str,
) -> List[Mapping[str, Any]]:
    """Return data-seed aggregate rows only when they add real data-seed variation."""
    if not data_seed_summary_rows or f"{metric_name}_mean" not in data_seed_summary_rows[0]:
        return []
    if max(int(row.get("num_data_seeds", 1)) for row in data_seed_summary_rows) <= 1:
        return []
    return list(data_seed_summary_rows)


def _plot_combined_metric(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    metric_name: str,
    axes: Sequence[str],
    path: Path,
    heatmap_axes: Sequence[str],
    data_seed_summary_rows: Sequence[Mapping[str, Any]],
    data_seed_axes: Sequence[str],
) -> None:
    """Save one metric PDF with line plots first and optional heatmap pages after."""
    use_pdf = bool(data_seed_axes) or (config.plot_heatmaps and len(heatmap_axes) >= 2)
    shared_ylim = _metric_ylim([summary_rows, data_seed_summary_rows], metric_name)
    if use_pdf:
        with PdfPages(path) as pdf:
            if axes:
                fig = _make_combined_line_figure(summary_rows, config, metric_name, axes, ylim=shared_ylim)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            if data_seed_axes:
                fig = _make_combined_line_figure(
                    data_seed_summary_rows,
                    config,
                    metric_name,
                    data_seed_axes,
                    title=f"{_metric_label(metric_name)}: data-seed means",
                    panel_suffix="data-seed means",
                    ylim=shared_ylim,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            if config.plot_heatmaps and len(heatmap_axes) >= 2:
                for y_axis, x_axis in combinations(heatmap_axes, 2):
                    heatmap_fig = _make_heatmap_pair_figure(summary_rows, metric_name, y_axis, x_axis)
                    if heatmap_fig is not None:
                        pdf.savefig(heatmap_fig, bbox_inches="tight")
                        plt.close(heatmap_fig)
        return

    fig = _make_combined_line_figure(summary_rows, config, metric_name, axes, ylim=shared_ylim)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _make_combined_line_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    metric_name: str,
    axes: Sequence[str],
    title: Optional[str] = None,
    panel_suffix: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Build the line-plot dashboard page for one metric."""
    plans = [_line_plot_plan(summary_rows, config, axis) for axis in axes]
    max_cols = max((plan.cols if plan.facet_axis else 1) for plan in plans)
    total_rows = sum(plan.rows for plan in plans)
    fig_width = 5.8 * max_cols + 1.0
    fig_height = 3.4 * total_rows + 0.8
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(total_rows, max_cols)

    row_offset = 0
    for plan in plans:
        _draw_plan(
            fig,
            grid,
            row_offset,
            max_cols,
            summary_rows,
            metric_name,
            plan,
            panel_suffix=panel_suffix,
            ylim=ylim,
        )
        row_offset += plan.rows

    fig.suptitle(title or _metric_label(metric_name), fontsize=13)
    return fig


def _plot_metric_axis(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    metric_name: str,
    axis: str,
    path: Path,
) -> None:
    """Save the optional individual PDF for one metric-vs-axis view."""
    plan = _line_plot_plan(summary_rows, config, axis)
    fig_width = 5.8 * plan.cols + 1.0
    fig_height = 3.4 * plan.rows + 0.8
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(plan.rows, plan.cols)
    _draw_plan(fig, grid, 0, plan.cols, summary_rows, metric_name, plan)
    if plan.facet_axis:
        fig.suptitle(f"{_metric_label(metric_name)} vs {_axis_label(axis)}", fontsize=13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_plan(
    fig: plt.Figure,
    grid: Any,
    row_offset: int,
    max_cols: int,
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    plan: LinePlotPlan,
    panel_suffix: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Draw either a single axis-spanning panel or a grid of facet panels."""
    if plan.facet_axis is None:
        ax = fig.add_subplot(grid[row_offset:row_offset + 1, :max_cols])
        title = f"{_metric_label(metric_name)} vs {_axis_label(plan.axis)}"
        if panel_suffix:
            title += f" ({panel_suffix})"
        _draw_axis_plot(
            ax,
            summary_rows,
            metric_name,
            plan,
            facet_value=None,
            title=title,
            ylim=ylim,
        )
        return

    for idx, facet_value in enumerate(plan.facet_values):
        row = row_offset + idx // plan.cols
        col = idx % plan.cols
        ax = fig.add_subplot(grid[row, col])
        title = f"vs {_axis_label(plan.axis)}, {plan.facet_axis}={_format_value(facet_value)}"
        if panel_suffix:
            title += f" ({panel_suffix})"
        _draw_axis_plot(
            ax,
            summary_rows,
            metric_name,
            plan,
            facet_value=facet_value,
            title=title,
            ylim=ylim,
        )


def _draw_axis_plot(
    ax: plt.Axes,
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    plan: LinePlotPlan,
    facet_value: Optional[Any],
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Draw means with standard-deviation error bars for one plan panel."""
    rows = _facet_filtered_rows(summary_rows, plan.facet_axis, facet_value)
    grouped = _group_rows(rows, plan.line_axes)
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    y_values: List[float] = []
    legend_labels: List[str] = []

    for key, group_rows in sorted(grouped.items(), key=lambda item: _sort_key_tuple(item[0])):
        group_rows = sorted(group_rows, key=lambda row: _sort_key(row[plan.axis]))
        x = np.asarray([float(row[plan.axis]) for row in group_rows], dtype=float)
        y = np.asarray([float(row[mean_key]) for row in group_rows], dtype=float)
        yerr = np.asarray([float(row[std_key]) for row in group_rows], dtype=float)
        y_values.extend(y[np.isfinite(y)].tolist())

        label = _group_label(plan.line_axes, key)
        if label:
            legend_labels.append(label)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label or None)

    x_values = [float(row[plan.axis]) for row in rows]
    if x_values and all(value > 0 for value in x_values):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)

    ticks = _unique_values(rows, plan.axis)
    ax.set_xticks([float(value) for value in ticks])
    ax.set_xticklabels(
        [_format_value(value) for value in ticks],
        rotation=45 if len(ticks) > 6 else 0,
        ha="right" if len(ticks) > 6 else "center",
    )
    ax.grid(False)
    ax.set_xlabel(_axis_label(plan.axis))
    ax.set_ylabel(_metric_label(metric_name))
    ax.set_title(title)
    if len(set(legend_labels)) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))


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


# -------------------------------------------------------------------------- #
# -------------------------------- heatmaps -------------------------------- #
# -------------------------------------------------------------------------- #

def _plot_metric_heatmaps(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    varying_axes: Sequence[str],
    output_dir: Path,
) -> Dict[str, Path]:
    """Save heatmap overview PDFs for every pair of varying sweep axes."""
    paths: Dict[str, Path] = {}
    for y_axis, x_axis in combinations(varying_axes, 2):
        path = output_dir / f"{metric_name}_heatmap_{y_axis}_by_{x_axis}.pdf"
        if _plot_heatmap_pair(summary_rows, metric_name, y_axis, x_axis, path):
            paths[f"plot_{metric_name}_heatmap_{y_axis}_by_{x_axis}"] = path
    return paths


def _plot_heatmap_pair(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    y_axis: str,
    x_axis: str,
    path: Path,
) -> bool:
    """Save one standalone heatmap PDF for a pair of sweep axes."""
    fig = _make_heatmap_pair_figure(summary_rows, metric_name, y_axis, x_axis)
    if fig is None:
        return False
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _make_heatmap_pair_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    y_axis: str,
    x_axis: str,
) -> Optional[plt.Figure]:
    """Build a heatmap figure over two axes, faceting by the third axis if needed."""
    fixed_axes = [
        name for name in SWEEP_AXES
        if name not in {x_axis, y_axis} and len(_unique_values(summary_rows, name)) > 1
    ]
    fixed_axis = fixed_axes[0] if fixed_axes else None
    fixed_values = _unique_values(summary_rows, fixed_axis) if fixed_axis else (None,)
    panels = []

    for fixed_value in fixed_values:
        rows = _facet_filtered_rows(summary_rows, fixed_axis, fixed_value)
        matrix, x_values, y_values = _heatmap_matrix(rows, metric_name, y_axis, x_axis)
        panels.append((fixed_value, matrix, x_values, y_values))

    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)].ravel()
        for _, matrix, _, _ in panels
        if np.isfinite(matrix).any()
    ]) if panels else np.asarray([])
    if finite_values.size == 0:
        return None

    use_log = bool(np.all(finite_values > 0))
    transformed = [
        (fixed_value, np.log10(matrix) if use_log else matrix, x_values, y_values)
        for fixed_value, matrix, x_values, y_values in panels
    ]
    finite_transformed = np.concatenate([
        matrix[np.isfinite(matrix)].ravel()
        for _, matrix, _, _ in transformed
        if np.isfinite(matrix).any()
    ])
    vmin = float(finite_transformed.min())
    vmax = float(finite_transformed.max())

    rows_count, cols_count = _grid_shape(len(transformed))
    fig = plt.figure(
        figsize=(4.8 * cols_count + 1.0, 3.9 * rows_count + 0.9),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(rows_count, cols_count)
    axes_used = []
    image = None

    for idx, (fixed_value, matrix, x_values, y_values) in enumerate(transformed):
        row = idx // cols_count
        col = idx % cols_count
        ax = fig.add_subplot(grid[row, col])
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
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([_format_value(value) for value in x_values])
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([_format_value(value) for value in y_values])
        if len(x_values) > 5:
            ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel(_axis_label(x_axis))
        ax.set_ylabel(_axis_label(y_axis))
        title = f"{_metric_label(metric_name)}"
        if fixed_axis:
            title += f", {fixed_axis}={_format_value(fixed_value)}"
        ax.set_title(title)
        ax.grid(False)

    if image is not None:
        label = f"log10({_metric_label(metric_name)} mean)" if use_log else f"{_metric_label(metric_name)} mean"
        fig.colorbar(image, ax=axes_used, shrink=0.85, label=label)
    return fig


def _heatmap_matrix(
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    y_axis: str,
    x_axis: str,
) -> Tuple[np.ndarray, Tuple[Any, ...], Tuple[Any, ...]]:
    """Return a y-by-x matrix of summary means for one heatmap panel."""
    mean_key = f"{metric_name}_mean"
    x_values = _unique_values(rows, x_axis)
    y_values = _unique_values(rows, y_axis)
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for row in rows:
        matrix[y_index[row[y_axis]], x_index[row[x_axis]]] = float(row[mean_key])
    return matrix, x_values, y_values


# -------------------------------------------------------------------------- #
# ------------------------------- utilities -------------------------------- #
# -------------------------------------------------------------------------- #

def _clear_probe_plot_files(output_dir: Path) -> None:
    """Remove stale PDF plots from all generations of this probe."""
    for metric_name in ALL_METRICS + LEGACY_METRICS:
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
        for path in output_dir.glob(f"{metric_name}_heatmap_*.pdf"):
            path.unlink()


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    axes: Sequence[str],
) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    """Group rows by the values that should become separate line series."""
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[axis] for axis in axes)
        grouped.setdefault(key, []).append(row)
    return grouped


def _facet_filtered_rows(
    rows: Sequence[Mapping[str, Any]],
    facet_axis: Optional[str],
    facet_value: Optional[Any],
) -> List[Mapping[str, Any]]:
    """Filter rows to one facet value, or return all rows when not faceting."""
    if facet_axis is None:
        return list(rows)
    return [row for row in rows if row[facet_axis] == facet_value]


def _unique_values(rows: Sequence[Mapping[str, Any]], axis: Optional[str]) -> Tuple[Any, ...]:
    """Return sorted unique values for a sweep axis."""
    if axis is None:
        return (None,)
    return tuple(sorted({row[axis] for row in rows}, key=_sort_key))


def _axis_has_finite_values(rows: Sequence[Mapping[str, Any]], axis: str) -> bool:
    """Return False for axes like beta when an infinity value would break line plotting."""
    values = []
    for row in rows:
        try:
            values.append(float(row[axis]))
        except (TypeError, ValueError):
            return False
    return bool(values) and all(math.isfinite(value) for value in values)


def _group_label(axes: Sequence[str], key: Sequence[Any]) -> str:
    """Format a compact legend label from non-constant sweep axes."""
    return ", ".join(
        f"{axis}={_format_value(value)}"
        for axis, value in zip(axes, key)
    )


def _metric_label(name: str) -> str:
    """Human-readable metric label for titles and axes."""
    return name.replace("_", " ")


def _axis_label(name: str) -> str:
    """Human-readable sweep-axis label."""
    return name.replace("_", " ")


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


def _sort_key_tuple(values: Sequence[Any]) -> Tuple[Tuple[int, Any], ...]:
    """Sort tuple-valued group keys using the same rules as axis values."""
    return tuple(_sort_key(value) for value in values)
