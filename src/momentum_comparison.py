from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import ExpConfig, RunOpts
from .exp import (
    ResultsByLabel,
    _infer_last_epoch_from_results,
    _print_exp_config,
    _train_over_range,
)
from .langevin import MOMENTUM_DISCRETIZATIONS
from .plots import (
    _base_epoch_axis,
    _has_history,
    _mean_std_across_seeds,
    _mean_std_tuple_component,
)

ResultsByWidth = Dict[int, ResultsByLabel]


# -------------------------------------------------------------------------- #
# ------------------------------- dataclasses ------------------------------ #
# -------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MomentumComparisonVariant:
    label: str
    h: float
    gamma: float
    discretization: str


@dataclass(frozen=True)
class MomentumComparisonConfig:
    enabled: bool
    include_overdamped: bool
    widths: Tuple[int, ...]
    variants: Tuple[MomentumComparisonVariant, ...]


# -------------------------------------------------------------------------- #
# ----------------------------- config parsing ----------------------------- #
# -------------------------------------------------------------------------- #

def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return value as a mapping or raise a config error with context."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _validate_positive_finite(value: Any, name: str, allow_zero: bool = False) -> float:
    """Parse one numeric config value and enforce finite sign constraints."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric.") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if allow_zero:
        if out < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    elif out <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    """Parse one integer config value and require it to be positive."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer.") from None
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def _format_float_for_label(value: float) -> str:
    """Format floats compactly for generated legend labels."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _default_variant_label(h: float, gamma: float, discretization: str) -> str:
    """Build a readable label when the config omits one."""
    return f"{discretization} h={_format_float_for_label(h)} gamma={_format_float_for_label(gamma)}"


def parse_momentum_comparison_config(raw_cfg: Optional[Mapping[str, Any]]) -> MomentumComparisonConfig:
    """Parse and validate the optional top-level momentum_comparison block."""
    if raw_cfg is None:
        return MomentumComparisonConfig(enabled=False, include_overdamped=False, widths=(), variants=())

    cfg = _require_mapping(raw_cfg, "momentum_comparison")
    unknown = set(cfg) - {"enabled", "include_overdamped", "widths", "variants"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown momentum_comparison config field(s): {names}")
    enabled = bool(cfg.get("enabled", False))
    include_overdamped = bool(cfg.get("include_overdamped", False))
    raw_widths = cfg.get("widths", None)
    raw_variants = cfg.get("variants", [])

    if not enabled:
        return MomentumComparisonConfig(
            enabled=False,
            include_overdamped=include_overdamped,
            widths=(),
            variants=(),
        )
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("momentum_comparison.variants must be a non-empty list when enabled=True.")

    widths: Tuple[int, ...] = ()
    if raw_widths is not None:
        if not isinstance(raw_widths, list) or not raw_widths:
            raise ValueError("momentum_comparison.widths must be a non-empty list when provided.")
        parsed_widths = [
            _validate_positive_int(width, f"momentum_comparison.widths[{i}]")
            for i, width in enumerate(raw_widths)
        ]
        duplicates = {width for width in parsed_widths if parsed_widths.count(width) > 1}
        if duplicates:
            raise ValueError(f"Duplicate momentum comparison width(s): {sorted(duplicates)}")
        widths = tuple(parsed_widths)

    variants: List[MomentumComparisonVariant] = []
    seen_labels = set()
    for i, raw_variant in enumerate(raw_variants):
        variant = _require_mapping(raw_variant, f"momentum_comparison.variants[{i}]")
        unknown = set(variant) - {"label", "h", "gamma", "discretization"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown momentum_comparison.variants[{i}] field(s): {names}")
        h = _validate_positive_finite(variant.get("h"), f"momentum_comparison.variants[{i}].h")
        gamma = _validate_positive_finite(
            variant.get("gamma"), f"momentum_comparison.variants[{i}].gamma", allow_zero=True
        )
        discretization = variant.get("discretization", "baoab")
        if discretization not in MOMENTUM_DISCRETIZATIONS:
            choices = ", ".join(sorted(MOMENTUM_DISCRETIZATIONS))
            raise ValueError(
                f"momentum_comparison.variants[{i}].discretization must be one of {choices}; "
                f"got {discretization!r}"
            )

        label = variant.get("label")
        if label is None:
            label = _default_variant_label(h, gamma, discretization)
        label = str(label)
        if label in seen_labels:
            raise ValueError(f"Duplicate momentum comparison variant label: {label!r}")
        seen_labels.add(label)
        variants.append(MomentumComparisonVariant(label=label, h=h, gamma=gamma, discretization=discretization))

    return MomentumComparisonConfig(
        enabled=True,
        include_overdamped=include_overdamped,
        widths=widths,
        variants=tuple(variants),
    )


# -------------------------------------------------------------------------- #
# -------------------------- plot-result reshaping ------------------------- #
# -------------------------------------------------------------------------- #

def _strip_momentum_prefix(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one seed's momentum_* metrics into normal plot metric names."""
    stripped: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("momentum_"):
            stripped[key[len("momentum_"):]] = value
        elif not key.endswith("_hist") and key not in {
            "model_state_dict",
            "lin_params_state",
            "momentum_model_state_dict",
            "momentum_lin_params_state",
            "momentum_buffers_state",
            "momentum_lin_buffers_state",
        }:
            stripped[key] = value
    return stripped


def _base_plot_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only the overdamped metrics from a run that also tracked momentum."""
    base: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("momentum_"):
            continue
        base[key] = value
    return base


def _append_result_group(
    combined: ResultsByLabel,
    source: ResultsByLabel,
    suffix: str,
    transform,
) -> None:
    """Append transformed alpha/beta results under labels suffixed for plotting."""
    for run_label, by_seed in source.items():
        combined_label = f"{run_label} | {suffix}"
        combined[combined_label] = {
            seed: transform(metrics)
            for seed, metrics in by_seed.items()
        }


# -------------------------------------------------------------------------- #
# ----------------------------- plotting helpers --------------------------- #
# -------------------------------------------------------------------------- #

def _dedup_legend_entries(axes):
    """Collect unique legend entries from all visible axes."""
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        if not ax.get_visible():
            continue
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _split_comparison_label(label: str) -> Tuple[str, str]:
    """Split a plot label into its beta/condition part and variant part."""
    if " | " not in label:
        return label, label
    condition, variant = label.split(" | ", 1)
    return condition, variant


def _conditions_for_results(results_by_width: ResultsByWidth) -> Tuple[str, ...]:
    """Return beta/condition labels in first-seen order across all widths."""
    conditions = []
    seen = set()
    for results in results_by_width.values():
        for label in results.keys():
            condition, _ = _split_comparison_label(label)
            if condition in seen:
                continue
            seen.add(condition)
            conditions.append(condition)
    return tuple(conditions)


def _metric_available_for_cell(results: ResultsByLabel, condition: str, metric_key: str) -> bool:
    """Return whether one width/beta cell has the requested metric history."""
    return any(
        label_condition == condition and _has_history(by_seed, metric_key)
        for label, by_seed in results.items()
        for label_condition, _ in [_split_comparison_label(label)]
    )


def _plot_comparison_band(ax, x, mean, std, label, style):
    """Plot a comparison curve with lighter bands and consistent line style."""
    ax.plot(
        x,
        mean,
        label=label,
        color=style["color"],
        linestyle="-",
        linewidth=style["linewidth"],
        zorder=style["zorder"],
    )
    if len(x) == 1 and len(mean) == 1:
        ax.plot(
            x,
            mean,
            marker="o",
            color=style["color"],
            linestyle="None",
            markersize=4.0,
            zorder=style["zorder"] + 1,
        )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.10,
        color=style["color"],
        linewidth=0.0,
        zorder=1,
    )


def _ordered_result_items(results: ResultsByLabel):
    """Yield overdamped last so its baseline curve remains visible."""
    regular = []
    overdamped = []
    for item in results.items():
        _, variant = _split_comparison_label(item[0])
        if variant == "overdamped":
            overdamped.append(item)
        else:
            regular.append(item)
    return regular + overdamped


def _plot_scalar_metric_on_axis(
    ax,
    results: ResultsByLabel,
    condition: str,
    metric_key: str,
    track_every: int,
    styles,
) -> bool:
    """Plot one scalar metric page panel for one width/beta cell."""
    plotted = False
    for run_name, by_seed in _ordered_result_items(results):
        label_condition, variant = _split_comparison_label(run_name)
        if label_condition != condition:
            continue
        if not _has_history(by_seed, metric_key):
            continue
        mean, std, length = _mean_std_across_seeds(by_seed, metric_key)
        x = _base_epoch_axis(by_seed, track_every)[:length]
        style = styles[variant]
        _plot_comparison_band(ax, x, mean, std, variant, style)
        plotted = True
    return plotted


def _plot_pair_metric_on_axis(
    ax,
    results: ResultsByLabel,
    condition: str,
    metric_key: str,
    component: int,
    track_every: int,
    styles,
) -> bool:
    """Plot one tuple-metric component page panel for one width/beta cell."""
    plotted = False
    for run_name, by_seed in _ordered_result_items(results):
        label_condition, variant = _split_comparison_label(run_name)
        if label_condition != condition:
            continue
        if not _has_history(by_seed, metric_key):
            continue
        mean, std, length = _mean_std_tuple_component(by_seed, metric_key, component=component)
        if component == 0 and len(mean) > 0:
            mean[0] = max(mean[0], 1e-12)
        x = _base_epoch_axis(by_seed, track_every)[:length]
        style = styles[variant]
        _plot_comparison_band(ax, x, mean, std, variant, style)
        plotted = True
    return plotted


def _comparison_metric_specs(use_linearized: bool):
    """Describe metric pages for the multi-page width comparison PDF."""
    specs = [
        {
            "name": "train_loss",
            "title": "Training Loss",
            "ylabel": "Training loss",
            "key": "train_loss_hist",
            "kind": "scalar",
        },
        {
            "name": "test_loss",
            "title": "Test Loss",
            "ylabel": "Test loss",
            "key": "test_loss_hist",
            "kind": "scalar",
        },
        {
            "name": "jacobian_dist_l2",
            "title": "Jacobian Drift L2",
            "ylabel": "Distance (L2, normalized)",
            "key": "jacobian_dist_hist",
            "kind": "pair",
            "component": 0,
        },
        {
            "name": "jacobian_dist_co",
            "title": "Jacobian Drift Cosine",
            "ylabel": "Distance (cosine)",
            "key": "jacobian_dist_hist",
            "kind": "pair",
            "component": 1,
        },
        {
            "name": "feat_gram_lambda",
            "title": "Feature Gram Lambda",
            "ylabel": r"$\lambda_{\min}$",
            "key": "feat_gram_lambda_hist",
            "kind": "scalar",
            "yscale": "log",
        },
    ]
    if use_linearized:
        specs[1:1] = [
            {
                "name": "lin_train_loss",
                "title": "Linearized Training Loss",
                "ylabel": "Training loss",
                "key": "lin_train_loss_hist",
                "kind": "scalar",
            },
            {
                "name": "nn_to_lin_dist_l2",
                "title": "NN to Linearized Distance L2",
                "ylabel": "Distance (L2, normalized)",
                "key": "nn_lin_param_dist_hist",
                "kind": "pair",
                "component": 0,
            },
            {
                "name": "nn_to_lin_dist_co",
                "title": "NN to Linearized Distance Cosine",
                "ylabel": "Distance (cosine)",
                "key": "nn_lin_param_dist_hist",
                "kind": "pair",
                "component": 1,
            },
        ]
    return specs


def _styles_for_results(results_by_width: ResultsByWidth):
    """Assign stable colors to variant labels across all metric pages."""
    labels = []
    seen = set()
    for results in results_by_width.values():
        for label in results.keys():
            _, variant = _split_comparison_label(label)
            if variant in seen:
                continue
            seen.add(variant)
            labels.append(variant)
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        label: {
            "color": prop_colors[i % len(prop_colors)],
            "linewidth": 2.0 if label == "overdamped" else 1.6,
            "zorder": 4 if label == "overdamped" else 3,
        }
        for i, label in enumerate(labels)
    }


def _share_page_y_limits(axes):
    """Set one y-axis range across all visible subplots on a metric page."""
    visible_axes = [ax for ax in axes if ax.get_visible()]
    if not visible_axes:
        return

    ymin = min(ax.get_ylim()[0] for ax in visible_axes)
    ymax = max(ax.get_ylim()[1] for ax in visible_axes)
    if ymin == ymax:
        pad = abs(ymin) * 0.05 if ymin != 0 else 1.0
        ymin -= pad
        ymax += pad

    for ax in visible_axes:
        ax.set_ylim(ymin, ymax)


def _write_metric_page(pdf, results_by_width: ResultsByWidth, spec, track_every: int, styles) -> bool:
    """Write one metric page with rows by width and columns by beta."""
    widths = list(results_by_width.keys())
    conditions = _conditions_for_results(results_by_width)
    available_cells = {
        (width, condition): _metric_available_for_cell(results_by_width[width], condition, spec["key"])
        for width in widths
        for condition in conditions
    }
    if not any(available_cells.values()):
        warnings.warn(
            f"[momentum_comparison] skipping {spec['title']}; missing required metric {spec['key']!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    nrows = len(widths)
    ncols = max(1, len(conditions))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols + 2.0, 2.5 * nrows),
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for row, width in enumerate(widths):
        for col, condition in enumerate(conditions):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(condition)
            if col == 0:
                ax.set_ylabel(f"m={width}\n{spec['ylabel']}")
            else:
                ax.set_ylabel(spec["ylabel"])
            if row == len(widths) - 1:
                ax.set_xlabel("Epoch")
            else:
                ax.set_xlabel("")
            if spec.get("yscale") == "log":
                ax.set_yscale("log")

            available = available_cells[(width, condition)]
            if not available:
                ax.set_visible(False)
                continue

            results = results_by_width[width]
            if spec["kind"] == "pair":
                plotted = _plot_pair_metric_on_axis(
                    ax,
                    results,
                    condition,
                    spec["key"],
                    spec["component"],
                    track_every,
                    styles,
                )
            else:
                plotted = _plot_scalar_metric_on_axis(ax, results, condition, spec["key"], track_every, styles)
            if not plotted:
                ax.set_visible(False)

    visible_axes = [ax for ax in flat_axes if ax.get_visible()]
    _share_page_y_limits(visible_axes)
    handles, labels = _dedup_legend_entries(visible_axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.83, 0.5),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(spec["title"])
    fig.tight_layout(rect=[0, 0, 0.82, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return True


# -------------------------------------------------------------------------- #
# ------------------------------ "public API" ------------------------------ #
# -------------------------------------------------------------------------- #

def plot_momentum_width_comparison(
    results_by_width: ResultsByWidth,
    track_every: int,
    use_linearized: bool,
    plot_output_dir="plots",
) -> Path:
    """Write the multi-page comparison PDF with one metric per page."""
    plot_output_dir = Path(plot_output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_output_dir / "expr1_full.pdf"

    styles = _styles_for_results(results_by_width)
    wrote_any = False
    with PdfPages(output_path) as pdf:
        for spec in _comparison_metric_specs(use_linearized):
            wrote_any = _write_metric_page(pdf, results_by_width, spec, track_every, styles) or wrote_any

    if not wrote_any:
        raise RuntimeError("No tracked metrics were available for momentum comparison plotting.")
    return output_path


def run_momentum_comparison(
    config: ExpConfig,
    run_opts: RunOpts,
    comparison: MomentumComparisonConfig,
    gpu_ids: Optional[List[int]],
) -> Tuple[ResultsByWidth, ExpConfig]:
    """Run all configured widths and momentum variants, returning plot-ready groups."""
    if run_opts.load_ckpt or run_opts.resume_from_ckpt:
        raise ValueError("momentum_comparison only supports fresh runs; load/resume from checkpoint is disabled.")
    if run_opts.save_ckpt:
        raise ValueError("momentum_comparison does not support save_ckpt; comparison results are plot-only.")

    results_by_width: ResultsByWidth = {}
    final_epochs = config.epochs
    widths = comparison.widths or (config.m,)

    for width in widths:
        combined: ResultsByLabel = {}
        for i, variant in enumerate(comparison.variants):
            variant_config = replace(
                config,
                m=width,
                compare_momentum=True,
                momentum_h=variant.h,
                momentum_gamma=variant.gamma,
                momentum_discretization=variant.discretization,
            )
            print(
                f"[momentum_comparison] running width m={width}, "
                f"variant {i + 1}/{len(comparison.variants)}: {variant.label}"
            )
            _print_exp_config(variant_config)
            results = _train_over_range(variant_config, variant_config.alphas, variant_config.betas, gpu_ids)
            final_epochs = max(final_epochs, _infer_last_epoch_from_results(results, variant_config.epochs))

            if comparison.include_overdamped and i == 0:
                _append_result_group(combined, results, "overdamped", _base_plot_metrics)
            _append_result_group(combined, results, variant.label, _strip_momentum_prefix)
        results_by_width[width] = combined

    return results_by_width, replace(config, epochs=final_epochs)
