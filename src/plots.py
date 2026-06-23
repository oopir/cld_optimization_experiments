from itertools import cycle
from pathlib import Path
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
import matplotlib.ticker as mticker

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
    }
)

def _mean_std_across_seeds(results_by_seed, key):
    """Average one scalar history over seeds, truncating to epochs all seeds reached."""
    histories = [np.asarray(r[key]) for r in results_by_seed.values()]

    # Different seeds may have stopped at different epochs.
    min_len = min(h.shape[0] for h in histories)
    histories = [h[:min_len] for h in histories]

    arr = np.stack(histories, axis=0)

    return arr.mean(axis=0), arr.std(axis=0), min_len


def _has_history(results_by_seed, key):
    """Return whether every seed has a non-empty history for a plotted metric."""
    return all(key in metrics and len(metrics[key]) > 0 for metrics in results_by_seed.values())


def _warn_missing(run_name, panel_name, key):
    """Warn that a configurable metric was not tracked, so the panel is skipped."""
    warnings.warn(
        f"[plot] skipping {panel_name} for {run_name}; missing required metric {key!r}",
        RuntimeWarning,
        stacklevel=2,
    )


def _base_epoch_axis(results_by_seed, track_every):
    """Build the epoch axis for plotted metric histories, preferring checkpointed epoch_hist."""
    epoch_histories = [
        np.asarray(metrics["epoch_hist"])
        for metrics in results_by_seed.values()
        if "epoch_hist" in metrics
    ]
    if epoch_histories:
        min_len = min(h.shape[0] for h in epoch_histories)
        return epoch_histories[0][:min_len]

    lengths = []
    for metrics in results_by_seed.values():
        fallback = next((v for k, v in metrics.items() if k.endswith("_hist")), [])
        lengths.append(len(fallback))
    num_steps = min(lengths) if lengths else 0
    return np.arange(1, num_steps * track_every + 1, track_every)


def _mean_std_tuple_component(results_by_seed, key, component):
    """Average one component of tuple histories such as (L2, cosine) distances."""
    histories = [np.asarray(r[key]) for r in results_by_seed.values()]
    min_len = min(h.shape[0] for h in histories)
    arr = np.stack([h[:min_len] for h in histories], axis=0)
    return arr[:, :, component].mean(axis=0), arr[:, :, component].std(axis=0), min_len


def _plot_band(ax, x, mean, std, label, color, linestyle="-", lw=2.0):
    """Plot a mean history with a seed-std band."""
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=lw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color, linewidth=0.0)


def _plot_l2_cos_metric(results_by_seed, axes, base_x, key, axis_l2, axis_cos, label, color, linestyle="-"):
    """Plot distance histories stored as (L2, cosine), e.g. NTK drift or NN-vs-linearized."""
    l2_mean, l2_std, length = _mean_std_tuple_component(results_by_seed, key, component=0)
    l2_mean[0] = max(l2_mean[0], 1e-12)
    _plot_band(axes[axis_l2], base_x[:length], l2_mean, l2_std, label=label, color=color, linestyle=linestyle)

    co_mean, co_std, length = _mean_std_tuple_component(results_by_seed, key, component=1)
    _plot_band(axes[axis_cos], base_x[:length], co_mean, co_std, label=label, color=color, linestyle=linestyle)
    return {axis_l2, axis_cos}


def _save_individual_axes(fig, axes, axes_list, plot_output_dir):
    """Save each subplot as its own PDF in the configured plot_output_dir."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0] if fig.legends else None
    original_visibility = {ax: ax.get_visible() for ax in axes_list}
    legend_was_visible = legend.get_visible() if legend is not None else None

    bboxes_in = {}
    widths = []
    heights = []
    for name, ax in axes.items():
        bb_in = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        bboxes_in[name] = bb_in
        widths.append(bb_in.x1 - bb_in.x0)
        heights.append(bb_in.y1 - bb_in.y0)

    max_w = max(widths)
    max_h = max(heights)
    pad_lr = 0.03
    pad_top = 0.05
    pad_bottom = 0.0

    for name, ax in axes.items():
        for candidate in axes_list:
            candidate.set_visible(candidate is ax)
        if legend is not None:
            legend.set_visible(False)

        bb = bboxes_in[name]
        bbox_equal = Bbox.from_extents(
            bb.x0 - pad_lr,
            bb.y0 - pad_bottom,
            bb.x0 + max_w + pad_lr,
            bb.y1 + (max_h - (bb.y1 - bb.y0)) + pad_top,
        )
        fig.savefig(plot_output_dir / f"expr1_{name}.pdf", bbox_inches=bbox_equal)

    for ax in axes_list:
        ax.set_visible(original_visibility[ax])
    if legend is not None:
        legend.set_visible(legend_was_visible)


def plot_ex1_multiseed(results, epochs, track_every, use_linearized=True, plot_output_dir="plots"):
    """
    Plot one internally consistent experiment batch.

    Assumed metrics for the full panel set are jacobian_dist_hist,
    nn_lin_param_dist_hist, train_loss_hist, lin_train_loss_hist, and
    feat_gram_lambda_hist. Each missing panel is skipped with a warning so
    configurable metric runs can still produce the plots they support.
    Different history lengths from early stopping are okay; histories are
    truncated per run to epochs all seeds reached.
    """
    plot_output_dir = Path(plot_output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )

    fig = plt.figure(figsize=(8, 13.0))
    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)
    ax1l = plt.subplot(gs[0, 0])
    ax1r = plt.subplot(gs[0, 1])
    ax2l = plt.subplot(gs[1, 0])
    ax2r = plt.subplot(gs[1, 1])
    ax3l = plt.subplot(gs[2, 0])
    ax3r = plt.subplot(gs[2, 1])
    ax4l = plt.subplot(gs[3, 0])
    ax4r = plt.subplot(gs[3, 1])
    ax4r.set_axis_off()
    
    axes = {
        "jacobian_dist_l2": ax1l,
        "jacobian_dist_co": ax1r,
        "nn_to_lin_dist_l2": ax2l,
        "nn_to_lin_dist_co": ax2r,
        "train_loss": ax3l,
        "train_loss_with_lin": ax3r,
        "feat_gram_lambda": ax4l,
    }
    ylabels = {
        "jacobian_dist_l2": "Distance (L2, normalized)",
        "jacobian_dist_co": "Distance (cosine)",
        "nn_to_lin_dist_l2": "Distance (L2, normalized)",
        "nn_to_lin_dist_co": "Distance (cosine)",
        "feat_gram_lambda": r'$\lambda_{\min}$',
        "train_loss": "Training loss",
        "train_loss_with_lin": "Training loss",
     }
    log_axes = {"feat_gram_lambda"}

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plotted_axes = set()

    for run_name, run_results_by_seed in results.items():
        c = next(colors)
        base_x = _base_epoch_axis(run_results_by_seed, track_every)

        if has_jacobian_any:
            if _has_history(run_results_by_seed, "jacobian_dist_hist"):
                plotted_axes.update(
                    _plot_l2_cos_metric(
                        run_results_by_seed,
                        axes,
                        base_x,
                        key="jacobian_dist_hist",
                        axis_l2="jacobian_dist_l2",
                        axis_cos="jacobian_dist_co",
                        label=run_name,
                        color=c,
                    )
                )
            else:
                _warn_missing(run_name, "Jacobian drift", "jacobian_dist_hist")
            if _has_history(run_results_by_seed, "momentum_jacobian_dist_hist"):
                plotted_axes.update(
                    _plot_l2_cos_metric(
                        run_results_by_seed,
                        axes,
                        base_x,
                        key="momentum_jacobian_dist_hist",
                        axis_l2="jacobian_dist_l2",
                        axis_cos="jacobian_dist_co",
                        label=f"{run_name} momentum",
                        color=c,
                        linestyle=":",
                    )
                )

        if use_linearized:
            if _has_history(run_results_by_seed, "nn_lin_param_dist_hist"):
                plotted_axes.update(
                    _plot_l2_cos_metric(
                        run_results_by_seed,
                        axes,
                        base_x,
                        key="nn_lin_param_dist_hist",
                        axis_l2="nn_to_lin_dist_l2",
                        axis_cos="nn_to_lin_dist_co",
                        label=run_name,
                        color=c,
                    )
                )
            else:
                _warn_missing(run_name, "NN-vs-linearized distance", "nn_lin_param_dist_hist")
            if _has_history(run_results_by_seed, "momentum_nn_lin_param_dist_hist"):
                plotted_axes.update(
                    _plot_l2_cos_metric(
                        run_results_by_seed,
                        axes,
                        base_x,
                        key="momentum_nn_lin_param_dist_hist",
                        axis_l2="nn_to_lin_dist_l2",
                        axis_cos="nn_to_lin_dist_co",
                        label=f"{run_name} momentum",
                        color=c,
                        linestyle=":",
                    )
                )

        if _has_history(run_results_by_seed, "feat_gram_lambda_hist"):
            mean, std, L = _mean_std_across_seeds(run_results_by_seed, "feat_gram_lambda_hist")
            x = base_x[:L]
            _plot_band(axes["feat_gram_lambda"], x, mean, std, label=run_name, color=c)
            plotted_axes.add("feat_gram_lambda")
        else:
            _warn_missing(run_name, "feature Gram lambda", "feat_gram_lambda_hist")
        if _has_history(run_results_by_seed, "momentum_feat_gram_lambda_hist"):
            mean, std, L = _mean_std_across_seeds(run_results_by_seed, "momentum_feat_gram_lambda_hist")
            _plot_band(
                axes["feat_gram_lambda"], base_x[:L], mean, std,
                label=f"{run_name} momentum", color=c, linestyle=":"
            )
            plotted_axes.add("feat_gram_lambda")

        if _has_history(run_results_by_seed, "train_loss_hist"):
            mean, std, L = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
            x = base_x[:L]
            _plot_band(axes["train_loss"], x, mean, std, label=run_name, color=c, lw=1.5)
            _plot_band(axes["train_loss_with_lin"], x, mean, std, label=run_name, color=c, lw=1.5)
            plotted_axes.update({"train_loss", "train_loss_with_lin"})
        else:
            _warn_missing(run_name, "training loss", "train_loss_hist")
        if _has_history(run_results_by_seed, "momentum_train_loss_hist"):
            mean, std, L = _mean_std_across_seeds(run_results_by_seed, "momentum_train_loss_hist")
            x = base_x[:L]
            _plot_band(
                axes["train_loss"], x, mean, std,
                label=f"{run_name} momentum", color=c, linestyle=":", lw=1.5,
            )
            _plot_band(
                axes["train_loss_with_lin"], x, mean, std,
                label=f"{run_name} momentum", color=c, linestyle=":", lw=1.5,
            )
            plotted_axes.update({"train_loss", "train_loss_with_lin"})

        if use_linearized:
            if _has_history(run_results_by_seed, "lin_train_loss_hist"):
                mean, std, L = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
                x = base_x[:L]
                _plot_band(
                    axes["train_loss_with_lin"], x, mean, std,
                    label=f"{run_name} linear", color=c, linestyle="--", lw=1.5,
                )
                plotted_axes.add("train_loss_with_lin")
            else:
                _warn_missing(run_name, "linearized training loss", "lin_train_loss_hist")
            if _has_history(run_results_by_seed, "momentum_lin_train_loss_hist"):
                mean, std, L = _mean_std_across_seeds(run_results_by_seed, "momentum_lin_train_loss_hist")
                _plot_band(
                    axes["train_loss_with_lin"], base_x[:L], mean, std,
                    label=f"{run_name} momentum linear", color=c, linestyle="-.", lw=1.5,
                )
                plotted_axes.add("train_loss_with_lin")

    for k, ax in axes.items():
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabels[k])
        if k in {"train_loss", "train_loss_with_lin"}:
            ax.set_ylim(-0.05, 0.7)
        if k in log_axes:
            ax.set_yscale("log")
            if k == "feat_gram_lambda":
                ax.set_ylim(1.0e4, 2.2e4)
            scale = 1e4
            fmt = mticker.FuncFormatter(lambda y, _: f"{y / scale:g}")
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
            ax.text(0.0, 1.02, "1e4", transform=ax.transAxes, ha="left", va="bottom")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    for name, ax in axes.items():
        if name not in plotted_axes:
            ax.set_visible(False)

    if "jacobian_dist_l2" in plotted_axes:
        ax1l.legend(
            loc="center",
            bbox_to_anchor=(0.76, 0.30),
            frameon=False,
            fontsize=12,
        )
    if "nn_to_lin_dist_l2" in plotted_axes:
        ax2l.legend(
            loc="center",
            bbox_to_anchor=(0.76, 0.30),
            frameon=False,
            fontsize=12,
        )
    if "train_loss" in plotted_axes:
        ax3l.legend(loc="best", frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    axes_list = [ax1l, ax1r, ax2l, ax2r, ax3l, ax3r, ax4l, ax4r]
    axes_to_save = {name: ax for name, ax in axes.items() if name in plotted_axes}
    if axes_to_save:
        _save_individual_axes(fig, axes_to_save, axes_list, plot_output_dir)

    fig.savefig(plot_output_dir / "expr1_full.pdf", bbox_inches="tight")


def plot_test_error_vs_alpha(results, output_path="alpha_test_error.pdf"):
    """
    Plot final test error by alpha; assumes each plotted run tracks test_acc_hist.

    results: ResultsByLabel as produced by run_exp, with labels like
    'α=1e+00 β=inf' and values {seed -> metrics}. Runs without test_acc_hist
    are skipped with a warning.
    """
    xs = []
    ys = []

    for label, run_results_by_seed in results.items():
        if "α" not in label:
            continue

        # parse alpha from 'α=1e+00 β=...' -> take first token, then split on '='
        first_tok = label.split()[0]        # 'α=1e+00'
        alpha_str = first_tok.split("=", 1)[1]
        alpha = float(alpha_str)

        if not _has_history(run_results_by_seed, "test_acc_hist"):
            _warn_missing(label, "test error vs alpha", "test_acc_hist")
            continue

        # mean final test error across seeds (1 - test_acc)
        last_accs = []
        for metrics in run_results_by_seed.values():
            hist = metrics.get("test_acc_hist", None)
            if hist is not None and len(hist) > 0:
                last_accs.append(hist[-1])
        if not last_accs:
            continue

        mean_acc = float(np.mean(last_accs))
        err_pct = 100.0 * (1.0 - mean_acc)

        xs.append(alpha)
        ys.append(err_pct)

    if not xs:
        raise RuntimeError("No alpha-labelled runs found in results.")

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(xs, ys, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Test Error (%)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
