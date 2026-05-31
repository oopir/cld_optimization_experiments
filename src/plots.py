from itertools import cycle
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
    histories = [np.asarray(r[key]) for r in results_by_seed.values()]
    
    # truncate histories to the same length (different seeds might haved reached early stopping at different times)
    min_len = min(h.shape[0] for h in histories)
    histories = [h[:min_len] for h in histories]
    
    arr = np.stack(histories, axis=0)
    
    return arr.mean(axis=0), arr.std(axis=0), min_len

def _plot_band(ax, x, mean, std, label, color, lin=False, lw=2.0):
    linestyle = "--" if lin else "-"
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=lw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color, linewidth=0.0)

def plot_ex1_multiseed(results, epochs, track_every, use_linearized=True):
    """
    Plot one internally consistent experiment batch.

    This assumes metric availability is homogeneous across the plotted runs/seeds:
    if Jacobian plots are enabled, every plotted seed should have jacobian_dist_hist;
    if use_linearized=True, every plotted seed should have nn_lin_param_dist_hist
    and lin_train_loss_hist. Mixed old/new checkpoint sets are not handled gently.
    Different history lengths from early stopping are okay; we truncate per run.
    """
    # check if any run actually tracked jacobian distances
    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )
    # if not has_jacobian_any:
    #     raise RuntimeError("plot_ex1_multiseed expects Jacobian data")

    # ------------------------- figure config ------------------------- #
    # ('axes' dict is used later, so don't push this section to the end)
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
        "jacobian_dist_l2": "Distance (L2, normalized)",
        "jacobian_dist_co": "Distance (cosine)",
        "nn_to_lin_dist_l2": "Distance (L2, normalized)",
        "nn_to_lin_dist_co": "Distance (cosine)",
        "feat_rel_dist": "Distance (L2)",
        "feat_cos_dist": "Distance (cosine)",
        "feat_gram_lambda": r'$\lambda_{\min}$',
        "train_loss": "Training loss",
     }
    log_axes = {"feat_gram_lambda"}

    # ------------------------ actual plotting ------------------------ #
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for run_name, run_results_by_seed in results.items():
        c = next(colors)

        # per-run x: prefer epoch_hist if present, else infer from train_loss_hist length
        some_seed_metrics = next(iter(run_results_by_seed.values()))
        if "epoch_hist" in some_seed_metrics:
            base_x = np.asarray(some_seed_metrics["epoch_hist"])
        else:
            lengths = [len(r["train_loss_hist"]) for r in run_results_by_seed.values()]
            T_run = max(lengths) if lengths else 0
            base_x = np.arange(1, T_run * track_every + 1, track_every)

        # jacobian distances
        if has_jacobian_any:
            jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
            
            min_len = min(h.shape[0] for h in jac_histories)
            jac_histories = [h[:min_len] for h in jac_histories]
            x = base_x[:min_len]

            jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)
            l2_mean = jac_arr[:, :, 0].mean(axis=0)
            l2_std  = jac_arr[:, :, 0].std(axis=0)
            l2_mean[0] = max(l2_mean[0], 1e-12)
            _plot_band(axes["jacobian_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
            co_mean = jac_arr[:, :, 1].mean(axis=0)
            co_std  = jac_arr[:, :, 1].std(axis=0)
            _plot_band(axes["jacobian_dist_co"], x, co_mean, co_std, label=run_name, color=c)

        # param distances
        if use_linearized:
            param_histories = [np.asarray(r["nn_lin_param_dist_hist"]) for r in run_results_by_seed.values()]
            
            min_len = min(h.shape[0] for h in param_histories)
            param_histories = [h[:min_len] for h in param_histories]
            x = base_x[:min_len]
            
            param_arr = np.stack(param_histories, axis=0)  # (n_seeds, T, 2)
            l2_mean = param_arr[:, :, 0].mean(axis=0)
            l2_std  = param_arr[:, :, 0].std(axis=0)
            l2_mean[0] = max(l2_mean[0], 1e-12)
            _plot_band(axes["nn_to_lin_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
            co_mean = param_arr[:, :, 1].mean(axis=0)
            co_std  = param_arr[:, :, 1].std(axis=0)
            _plot_band(axes["nn_to_lin_dist_co"], x, co_mean, co_std, label=run_name, color=c)

        # relative feature distance
        mean, std, L = _mean_std_across_seeds(run_results_by_seed, "feat_rel_dist_hist")
        mean[0] = max(mean[0], 1e-12)
        x = base_x[:L]
        _plot_band(axes["feat_rel_dist"], x, mean, std, label=run_name, color=c)

        # cosine feature distance
        mean, std, L = _mean_std_across_seeds(run_results_by_seed, "feat_cos_dist_hist")
        x = base_x[:L]
        _plot_band(axes["feat_cos_dist"], x, mean, std, label=run_name, color=c)

        # min eigenvalue of Gram(A_t)
        mean, std, L = _mean_std_across_seeds(run_results_by_seed, "feat_gram_lambda_hist")
        x = base_x[:L]
        _plot_band(axes["feat_gram_lambda"], x, mean, std, label=run_name, color=c)

        # accuracy/loss (nonlinear vs linearized)
        mean, std, L = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
        x = base_x[:L]
        _plot_band(axes["train_loss"], x, mean, std, label=run_name, color=c, lw=1.5)
        # currently we don't show linearized model loss in any figure
        # if use_linearized:
        #     mean, std, L = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
        #     x = base_x[:L]
        #     _plot_band(axes["train_loss"], x, mean, std, label="linear", color=c, lin=True, lw=1.5)


    for k, ax in axes.items():
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabels[k])
        if k in log_axes:
            ax.set_yscale("log")
            scale = 1e4
            fmt = mticker.FuncFormatter(lambda y, _: f"{y / scale:g}")
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
            ax.text(0.0, 1.02, "1e4", transform=ax.transAxes, ha="left", va="bottom")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # handles, labels = axes["train_loss"].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False,)
    ax4r.legend(loc="best", frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes_list = [ax1l, ax1r, ax2l, ax2r, ax3l, ax3r, ax4l, ax4r]
    legend = fig.legends[0] if fig.legends else None

    # --- per-axis tight bboxes in inches ---
    bboxes_in = {}
    widths = []
    heights = []
    for name, ax in axes.items():
        bb = ax.get_tightbbox(renderer)
        bb_in = bb.transformed(fig.dpi_scale_trans.inverted())
        bboxes_in[name] = bb_in
        widths.append(bb_in.x1 - bb_in.x0)
        heights.append(bb_in.y1 - bb_in.y0)

    max_w = max(widths)
    max_h = max(heights)

    pad_lr = 0.03   # horizontal padding (both sides)
    pad_top = 0.05  # vertical padding at top
    pad_bottom = 0.0  # keep bottom essentially tight

    for name, ax in axes.items():
        # show only this axis
        for a in axes_list:
            a.set_visible(a is ax)
        # hide global legend
        if legend is not None:
            legend.set_visible(False)

        bb = bboxes_in[name]
        w = bb.x1 - bb.x0
        h = bb.y1 - bb.y0

        # bbox with:
        # - same left boundary across all axes (bb.x0 - pad_lr)
        # - same total width (max_w + 2*pad_lr)
        # - bottom ~tight (bb.y0 - pad_bottom)
        # - extra height added only at the top to reach max_h
        bbox_equal = Bbox.from_extents(
            bb.x0 - pad_lr,
            bb.y0 - pad_bottom,
            bb.x0 + max_w + pad_lr,
            bb.y1 + (max_h - h) + pad_top,
        )

        fig.savefig(f"expr1_{name}.pdf", bbox_inches=bbox_equal)

    # restore full figure (optional)
    for a in axes_list:
        a.set_visible(True)
    if legend is not None:
        legend.set_visible(True)

    fig.savefig(f"expr1_full.pdf", bbox_inches="tight")

def plot_test_error_vs_alpha(results, output_path="alpha_test_error.pdf"):
    """
    results: ResultsByLabel as produced by run_exp
             keys are labels like 'α=1e+00 β=inf', values are {seed -> metrics}
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
