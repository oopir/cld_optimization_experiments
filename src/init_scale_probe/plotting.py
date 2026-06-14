from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
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
# ------------------------------- line plots ------------------------------- #
# -------------------------------------------------------------------------- #

def plot_probe_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    output_dir: Path,
    init_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    data_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Create all configured plot outputs from the already-aggregated summary rows."""
    paths: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_probe_plot_files(output_dir)
    if not summary_rows:
        return paths

    if _is_initialization_only(config):
        return _plot_initialization_only_summaries(
            summary_rows,
            config,
            output_dir,
            init_seed_summary_rows=init_seed_summary_rows,
            data_seed_summary_rows=data_seed_summary_rows,
        )

    for metric_name in config.plot_metrics:
        if f"{metric_name}_mean" not in summary_rows[0]:
            continue

        if config.plot_format in {"combined", "both"}:
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

    return paths


def _is_initialization_only(config: InitScaleProbeConfig) -> bool:
    """Return True when the run only measures metrics at initialization."""
    return tuple(config.training_step_values or []) == (0,)


def _plot_initialization_only_summaries(
    summary_rows: Sequence[Mapping[str, Any]],
    config: InitScaleProbeConfig,
    output_dir: Path,
    init_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]],
    data_seed_summary_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Create the n/m initialization atlas plots, preserving trajectory plots elsewhere."""
    paths: Dict[str, Path] = {}
    init_rows = list(init_seed_summary_rows or summary_rows)
    data_rows = list(data_seed_summary_rows or summary_rows)
    if not init_rows or not data_rows:
        return paths

    for metric_name in config.plot_metrics:
        if f"{metric_name}_mean" not in init_rows[0] or f"{metric_name}_mean" not in data_rows[0]:
            continue

        if config.plot_format in {"combined", "both"}:
            path = output_dir / f"{metric_name}.pdf"
            if _save_figures_pdf(_make_initialization_atlas_figures(init_rows, data_rows, metric_name), path):
                paths[f"plot_{metric_name}"] = path

        if config.plot_format in {"individual", "both"}:
            path = output_dir / f"{metric_name}_initialization_atlas.pdf"
            if _save_figures_pdf(_make_initialization_atlas_figures(init_rows, data_rows, metric_name), path):
                paths[f"plot_{metric_name}_initialization_atlas"] = path

    return paths


def _make_initialization_atlas_figures(
    init_rows: Sequence[Mapping[str, Any]],
    data_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> List[plt.Figure]:
    """
    Build 2x2 n/m atlas pages for initialization-only experiments.

    Each page fixes non-n/m sweep axes. The left column shows mean metric vs m
    with one line per n; the right column shows the corresponding std heatmap
    over the n-by-m grid.
    """
    mean_key = f"{metric_name}_mean"
    init_rows = [row for row in init_rows if mean_key in row]
    data_rows = [row for row in data_rows if mean_key in row]
    if not init_rows or not data_rows:
        return []

    fixed_axes = tuple(axis for axis in ("alpha", "beta", "training_steps") if _axis_varies(init_rows, data_rows, axis))
    fixed_keys = sorted(
        {
            tuple(row[axis] for axis in fixed_axes)
            for row in [*init_rows, *data_rows]
        },
        key=lambda values: tuple(_sort_key(value) for value in values),
    ) or [()]

    figures: List[plt.Figure] = []
    shared_ylim = _metric_ylim([init_rows, data_rows], metric_name)
    for fixed_key in fixed_keys:
        init_panel_rows = _rows_matching_fixed_axes(init_rows, fixed_axes, fixed_key)
        data_panel_rows = _rows_matching_fixed_axes(data_rows, fixed_axes, fixed_key)
        if not init_panel_rows or not data_panel_rows:
            continue

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.6, 8.0),
            constrained_layout=True,
        )
        _draw_initialization_line_panel(
            axes[0][0],
            init_panel_rows,
            metric_name,
            title="Init variability: mean over data, +/-1 std init",
            ylim=shared_ylim,
        )
        _draw_variability_heatmap_panel(
            fig,
            axes[0][1],
            init_panel_rows,
            metric_name,
            value_kind="std",
            title="Init std over n and m",
            colorbar_label=f"init std of {_metric_label(metric_name)}",
        )
        _draw_initialization_line_panel(
            axes[1][0],
            data_panel_rows,
            metric_name,
            title="Data variability: mean over init, +/-1 std data",
            ylim=shared_ylim,
        )
        _draw_variability_heatmap_panel(
            fig,
            axes[1][1],
            data_panel_rows,
            metric_name,
            value_kind="std",
            title="Data std over n and m",
            colorbar_label=f"data std of {_metric_label(metric_name)}",
        )

        subtitle = _fixed_axes_subtitle(fixed_axes, fixed_key)
        title = f"{_metric_label(metric_name)}: initialization atlas"
        if subtitle:
            title += f" ({subtitle})"
        fig.suptitle(title, fontsize=13)
        figures.append(fig)

    return figures


def _draw_initialization_line_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    title: str,
    ylim: Optional[Tuple[float, float]],
) -> None:
    """Draw metric vs m with n as the visible non-aggregated series."""
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
            linewidth=1.5,
            color=color_by_n[n],
            label=f"n={_format_value(n)}",
        )

    m_values = _unique_values(rows, "m")
    if m_values and all(float(value) > 0 for value in m_values):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks([float(value) for value in m_values])
    ax.set_xticklabels([_format_value(value) for value in m_values])
    ax.set_xlabel(_axis_label("m"))
    ax.set_ylabel(_metric_label(metric_name))
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    if len(n_values) > 1:
        ax.legend(frameon=False, fontsize=8, title="sample size n")


def _draw_variability_heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    value_kind: str,
    title: str,
    colorbar_label: str,
) -> None:
    """Draw a n-by-m heatmap for mean or std summary values."""
    value_key = f"{metric_name}_{value_kind}"
    matrix, m_values, n_values = _heatmap_matrix(rows, value_key, "n", "m")
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
    ax.set_xticks(range(len(m_values)))
    ax.set_xticklabels([_format_value(value) for value in m_values], rotation=45 if len(m_values) > 4 else 0)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([_format_value(value) for value in n_values])
    ax.set_xlabel(_axis_label("m"))
    ax.set_ylabel(_axis_label("n"))
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label=colorbar_label)


def _axis_varies(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
    axis: str,
) -> bool:
    """Return whether an axis varies across either of two row sets."""
    values = {row[axis] for row in first_rows if axis in row}
    values.update(row[axis] for row in second_rows if axis in row)
    return len(values) > 1


def _rows_matching_fixed_axes(
    rows: Sequence[Mapping[str, Any]],
    fixed_axes: Sequence[str],
    fixed_key: Sequence[Any],
) -> List[Mapping[str, Any]]:
    """Filter rows to one fixed-axis page of the initialization atlas."""
    if not fixed_axes:
        return list(rows)
    return [
        row for row in rows
        if tuple(row[axis] for axis in fixed_axes) == tuple(fixed_key)
    ]


def _fixed_axes_subtitle(fixed_axes: Sequence[str], fixed_key: Sequence[Any]) -> str:
    """Format compact fixed-axis values for multipage atlas titles."""
    return ", ".join(
        f"{_axis_label(axis)}={_format_value(value)}"
        for axis, value in zip(fixed_axes, fixed_key)
    )


def _make_training_curves_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> Optional[plt.Figure]:
    """
    Build metric-vs-training-step small multiples.

    Panels are fixed by `beta`. Within each panel, every `(n, m)` pair gets a
    line: color encodes `m`, and marker shape encodes `n`.
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

    fig_width = max(4.0 * len(beta_values) + 1.2, 5.0)
    fig_height = 3.8
    fig, axes = plt.subplots(
        1,
        len(beta_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(m_values), 1)))
    color_by_m = {m_value: colors[idx] for idx, m_value in enumerate(m_values)}
    marker_shapes = ("o", "D", "^", "s", "v", "P", "X", "*", "<", ">")
    marker_by_n = {n_value: marker_shapes[idx % len(marker_shapes)] for idx, n_value in enumerate(n_values)}
    seen_n_values = set()
    seen_m_values = set()

    for col_idx, beta in enumerate(beta_values):
        ax = axes[0][col_idx]
        panel_rows = [row for row in rows if row["beta"] == beta]
        for n in n_values:
            n_rows = [row for row in panel_rows if row["n"] == n]
            for m in m_values:
                line_rows = sorted(
                    [
                        row for row in n_rows
                        if row["m"] == m and np.isfinite(float(row[mean_key]))
                    ],
                    key=lambda row: _sort_key(row["training_steps"]),
                )
                if not line_rows:
                    continue
                ax.plot(
                    np.asarray([float(row["training_steps"]) for row in line_rows], dtype=float),
                    np.asarray([float(row[mean_key]) for row in line_rows], dtype=float),
                    marker=marker_by_n[n],
                    linewidth=1.3,
                    markersize=4.0,
                    color=color_by_m[m],
                    linestyle="-",
                    label=f"n={_format_value(n)}, m={_format_value(m)}",
                )
                seen_n_values.add(n)
                seen_m_values.add(m)

        ax.set_title(f"beta={_format_value(beta)}", fontsize=9)
        ax.set_xticks([float(value) for value in step_values])
        ax.set_xticklabels([_format_value(value) for value in step_values])
        ax.grid(False)
        ax.set_xlabel(_axis_label("training_steps"))
        if col_idx == 0:
            ax.set_ylabel(_metric_label(metric_name))

    if seen_n_values:
        n_handles = [
            Line2D(
                [0],
                [0],
                color="0.25",
                marker=marker_by_n[n],
                linestyle="None",
                markersize=5,
            )
            for n in n_values
            if n in seen_n_values
        ]
        n_labels = [f"n={_format_value(n)}" for n in n_values if n in seen_n_values]
        n_legend = fig.legend(
            n_handles,
            n_labels,
            loc="outside right upper",
            frameon=False,
            title="sample size",
        )
        fig.add_artist(n_legend)
    if seen_m_values:
        m_handles = [
            Line2D([0], [0], color=color_by_m[m], linestyle="-", linewidth=1.6)
            for m in m_values
            if m in seen_m_values
        ]
        m_labels = [f"m={_format_value(m)}" for m in m_values if m in seen_m_values]
        fig.legend(
            m_handles,
            m_labels,
            loc="outside right lower",
            frameon=False,
            title="width",
        )
    fig.suptitle(f"{_metric_label(metric_name)} vs training steps", fontsize=13)
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

    use_log = bool(np.all(finite_values > 0))
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
        fig.colorbar(image, ax=axes_used, shrink=0.85, label=label)
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
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return True


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
        for path in (
            output_dir / f"{metric_name}_training_curves.pdf",
            output_dir / f"{metric_name}_nm_heatmaps.pdf",
            output_dir / f"{metric_name}_initialization_atlas.pdf",
        ):
            if path.exists():
                path.unlink()
        for path in output_dir.glob(f"{metric_name}_heatmap_*.pdf"):
            path.unlink()


def _unique_values(rows: Sequence[Mapping[str, Any]], axis: Optional[str]) -> Tuple[Any, ...]:
    """Return sorted unique values for a sweep axis."""
    if axis is None:
        return (None,)
    return tuple(sorted({row[axis] for row in rows}, key=_sort_key))


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
