from itertools import cycle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox

from .data import load_1d_regression_data

# ICML-style defaults: small fonts, sufficient line width, high DPI
mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
    }
)

def _mean_std_across_seeds(results_by_seed, key):
    histories = [np.asarray(r[key]) for r in results_by_seed.values()]
    arr = np.stack(histories, axis=0)  # (n_seeds, T)
    return arr.mean(axis=0), arr.std(axis=0)

def _plot_band(ax, x, mean, std, label, color, lin=False):
    linestyle = "--" if lin else "-"
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.0)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color, linewidth=0.0)

def plot_ex1_multiseed(results, epochs, track_every):
    # check if any run actually tracked jacobian distances
    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )
    if not has_jacobian_any:
        raise RuntimeError("plot_ex1_multiseed expects Jacobian data")

    # ------------------------- figure config ------------------------- #
    # ('axes' dict is used later, so don't push this section to the end)
    # ICML-friendly multi-panel figure
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

    axes = {
        "jacobian_dist_l2": ax1l,
        "jacobian_dist_co": ax1r,
        "nn_to_lin_dist_l2": ax2l,
        "nn_to_lin_dist_co": ax2r,
        "feat_rel_dist": ax3l,
        "feat_cos_dist": ax3r,
        "feat_gram_lambda": ax4l,
        "train_loss": ax4r,
    }
    ylabels = {
        "jacobian_dist_l2": "relative Jacobian distance (L2)",
        "jacobian_dist_co": "Jacobian distance (cosine)",
        "nn_to_lin_dist_l2": "relative parameter distance (L2)",
        "nn_to_lin_dist_co": "parameter distance (cosine)",
        "feat_rel_dist": "relative feature distance (L2)",
        "feat_cos_dist": "feature distance (cosine)",
        "feat_gram_lambda": "λ_min of feature Gram matrix",
        "train_loss": "training loss",
     }
    log_axes = {"jacobian_dist_l2", "nn_to_lin_dist_l2", "feat_rel_dist", "feat_gram_lambda"}

    # ------------------------ actual plotting ------------------------ #  
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # infer x from epoch_hist if present; else fall back to track_every
    sample_beta_key = next(iter(results.keys()))
    sample_seed_key = next(iter(results[sample_beta_key].keys()))
    sample_metrics = results[sample_beta_key][sample_seed_key]

    if "epoch_hist" in sample_metrics:
        x = np.asarray(sample_metrics["epoch_hist"])
    else:
        x = np.arange(1, epochs + 1, track_every)
        
    for run_name, run_results_by_seed in results.items():
        c = next(colors)
        
        # jacobian distances
        jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
        jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)
        l2_mean = jac_arr[:, :, 0].mean(axis=0)
        l2_std  = jac_arr[:, :, 0].std(axis=0)
        l2_mean[0] = max(l2_mean[0], 1e-12)
        _plot_band(axes["jacobian_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
        co_mean = jac_arr[:, :, 1].mean(axis=0)
        co_std  = jac_arr[:, :, 1].std(axis=0)
        _plot_band(axes["jacobian_dist_co"], x, co_mean, co_std, label=run_name, color=c)
        
        # param distances
        param_histories = [np.asarray(r["nn_lin_param_dist_hist"]) for r in run_results_by_seed.values()]
        param_arr = np.stack(param_histories, axis=0)  # (n_seeds, T, 2)
        l2_mean = param_arr[:, :, 0].mean(axis=0)
        l2_std  = param_arr[:, :, 0].std(axis=0)
        l2_mean[0] = max(l2_mean[0], 1e-12)
        _plot_band(axes["nn_to_lin_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
        co_mean = param_arr[:, :, 1].mean(axis=0)
        co_std  = param_arr[:, :, 1].std(axis=0)
        _plot_band(axes["nn_to_lin_dist_co"], x, co_mean, co_std, label=run_name, color=c)

        # relative feature distance
        mean, std = _mean_std_across_seeds(run_results_by_seed, "feat_rel_dist_hist")
        mean[0] = max(mean[0], 1e-12)
        _plot_band(axes["feat_rel_dist"], x, mean, std, label=run_name, color=c)

        # cosine feature distance
        mean, std = _mean_std_across_seeds(run_results_by_seed, "feat_cos_dist_hist")
        _plot_band(axes["feat_cos_dist"], x, mean, std, label=run_name, color=c)

        # min eigenvalue of Gram(A_t)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "feat_gram_lambda_hist")
        _plot_band(axes["feat_gram_lambda"], x, mean, std, label=run_name, color=c)

        # accuracy/loss (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=f"{run_name} lin", color=c, lin=True)

    for k, ax in axes.items():
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabels[k])
        if k in log_axes:
            ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    handles, labels = axes["train_loss"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False,)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes_list = [ax1l, ax1r, ax2l, ax2r, ax3l, ax3r, ax4l, ax4r]
    legend = fig.legends[0] if fig.legends else None

    for name, ax in axes.items():
        # show only this axis
        for a in axes_list:
            a.set_visible(a is ax)
        # hide global legend for per-panel files
        if legend is not None:
            legend.set_visible(False)
        # tight bbox (includes labels), then extend only upwards
        bbox = ax.get_tightbbox(renderer)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())  # to inches
        bbox = Bbox.from_extents(bbox.x0 - 0.1, bbox.y0, bbox.x1 + 0.1, bbox.y1 + 0.30)
        fig.savefig(f"expr1_{name}.pdf", bbox_inches=bbox)

    # restore full figure (optional)
    for a in axes_list:
        a.set_visible(True)
    if legend is not None:
        legend.set_visible(True)

    fig.savefig(f"expr1_full.pdf", bbox_inches="tight")
