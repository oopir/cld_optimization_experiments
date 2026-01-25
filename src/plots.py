from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .data import load_1d_regression_data

def _mean_std_across_seeds(results_by_seed, key):
    histories = [np.asarray(r[key]) for r in results_by_seed.values()]
    arr = np.stack(histories, axis=0)  # (n_seeds, T)
    return arr.mean(axis=0), arr.std(axis=0)

def _plot_band(ax, x, mean, std, label, color, lin=False):
    if lin:
        ax.plot(x, mean, label=label, color=color, linestyle="--")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    else:
        ax.plot(x, mean, label=label, color=color, linestyle="-")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

def plot_ex1_multiseed_short(results, epochs, track_every):
    # check if any run actually tracked jacobian distances
    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )
    if not has_jacobian_any:
        raise RuntimeError("plot_ex1_multiseed_short expects Jacobian data")

    # ------------------------- figure config ------------------------- #
    # ('axes' dict is used later, so don't push this section to the end)
    plt.figure(figsize=(8, 4))
    gs   = gridspec.GridSpec(1, 2)
    ax1l = plt.subplot(gs[0, 0])   # first  row left
    ax1r = plt.subplot(gs[0, 1])   # first  row right

    axes = {"train_loss": ax1l, "jacobian_dist_hist_co": ax1r}
    titles = {
        "train_loss": "train loss",
        "jacobian_dist_hist_co": "jacobian distance from init (cosine)",
    }
    log_axes = set()

    # ------------------------ actual plotting ------------------------ #  
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    x      = np.arange(1, epochs+1, track_every)

    for run_name, run_results_by_seed in results.items():
        c = next(colors)

        # accuracy/loss (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=f"{run_name} lin", color=c, lin=True)

        # jacobian distance
        jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
        jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)

        co_mean = jac_arr[:, :, 1].mean(axis=0)
        co_std  = jac_arr[:, :, 1].std(axis=0)
        _plot_band(axes["jacobian_dist_hist_co"], x, co_mean, co_std, label=run_name, color=c)

    for k, ax in axes.items():
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()
    plt.tight_layout()
    plt.show()

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
    plt.figure(figsize=(10, 20))
    gs   = gridspec.GridSpec(4, 2)
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
    titles = {
        "jacobian_dist_l2": "relative Jacobian distance from init (l2)",
        "jacobian_dist_co": "jacobian distance from init (cosine)",
        "nn_to_lin_dist_l2": "relative distance of nn params from lin params (l2)",
        "nn_to_lin_dist_co": "distance of nn params from lin params (cosine)",
        "feat_rel_dist": "relative feature distance from init (l2)",
        "feat_cos_dist": "feature distance from init (cosine)",
        "feat_gram_lambda": "λ_min of activation's Gram matrix",
        "train_loss": "train loss",
    }
    log_axes = {"jacobian_dist_l2", "nn_to_lin_dist_l2", "feat_rel_dist"}

    # ------------------------ actual plotting ------------------------ #  
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    x      = np.arange(1, epochs+1, track_every)

    for run_name, run_results_by_seed in results.items():
        c = next(colors)
        
        # jacobian distances
        jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
        jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)
        l2_mean = jac_arr[:, :, 0].mean(axis=0)
        l2_std  = jac_arr[:, :, 0].std(axis=0)
        _plot_band(axes["jacobian_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
        co_mean = jac_arr[:, :, 1].mean(axis=0)
        co_std  = jac_arr[:, :, 1].std(axis=0)
        _plot_band(axes["jacobian_dist_co"], x, co_mean, co_std, label=run_name, color=c)
        
        # param distances
        param_histories = [np.asarray(r["nn_lin_param_dist_hist"]) for r in run_results_by_seed.values()]
        param_arr = np.stack(param_histories, axis=0)  # (n_seeds, T, 2)
        l2_mean = param_arr[:, :, 0].mean(axis=0)
        l2_std  = param_arr[:, :, 0].std(axis=0)
        _plot_band(axes["nn_to_lin_dist_l2"], x, l2_mean, l2_std, label=run_name, color=c)
        co_mean = param_arr[:, :, 1].mean(axis=0)
        co_std  = param_arr[:, :, 1].std(axis=0)
        _plot_band(axes["nn_to_lin_dist_co"], x, co_mean, co_std, label=run_name, color=c)

        # relative feature distance
        mean, std = _mean_std_across_seeds(run_results_by_seed, "feat_rel_dist_hist")
        _plot_band(axes["feat_rel_dist"], x, mean, std, label=run_name, color=c)

        # relative feature distance
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
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_ex2_multiseed(results, epochs, track_every):
    # check if any run actually tracked jacobian distances
    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )

    if has_jacobian_any:
        plt.figure(figsize=(8, 12))
        gs   = gridspec.GridSpec(3, 2)
        ax1l = plt.subplot(gs[0, 0])   # first  row left
        ax1r = plt.subplot(gs[0, 1])   # first  row right
        ax2l = plt.subplot(gs[1, 0])   # second row left
        ax2r = plt.subplot(gs[1, 1])   # second row right
        ax3l = plt.subplot(gs[2, 0])   # third  row left
        ax3r = plt.subplot(gs[2, 1])   # third  row right
    else:
        # no jacobian tracking in any run → only 2 rows
        plt.figure(figsize=(8, 8))
        gs   = gridspec.GridSpec(2, 2)
        ax1l = plt.subplot(gs[0, 0])   # first  row left
        ax1r = plt.subplot(gs[0, 1])   # first  row right
        ax2l = plt.subplot(gs[1, 0])   # second row left
        ax2r = plt.subplot(gs[1, 1])   # second row right
        ax3l = ax3r = None

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    x = np.arange(track_every, epochs+1, track_every)

    for run_name, run_results_by_seed in results.items():
        c = next(colors)

        # distance from init (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "param_dist_hist")
        _plot_band(ax1l, x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_param_dist_hist")
        _plot_band(ax1l, x, mean, std, label=f"{run_name} lin", color=c, lin=True)
        # upper_bound_by_seed = np.asarray([r["param_dist_upper_bound"] for r in run_results_by_seed.values()])
        # ax1l.axhline(y=upper_bound_by_seed.mean(), linestyle='--', color='black')

        # loss (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
        _plot_band(ax1r, x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
        _plot_band(ax1r, x, mean, std, label=f"{run_name} lin", color=c, lin=True)

        # param norms
        mean, std = _mean_std_across_seeds(run_results_by_seed, "param_norm_fc1_hist")
        _plot_band(ax2l, x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_param_norm_fc1_hist")
        _plot_band(ax2l, x, mean, std, label=f"{run_name} lin", color=c, lin=True)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "param_norm_fc2_hist")
        _plot_band(ax2r, x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_param_norm_fc2_hist")
        _plot_band(ax2r, x, mean, std, label=f"{run_name} lin", color=c, lin=True)

        # jacobian distances (only for runs that actually have them)
        if has_jacobian_any and ax3l is not None and ax3r is not None:
            if all("jacobian_dist_hist" in r for r in run_results_by_seed.values()):
                jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
                jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)

                l2_mean = jac_arr[:, :, 0].mean(axis=0)
                l2_std  = jac_arr[:, :, 0].std(axis=0)
                _plot_band(ax3l, x, l2_mean, l2_std, label=run_name, color=c)

                co_mean = jac_arr[:, :, 1].mean(axis=0)
                co_std  = jac_arr[:, :, 1].std(axis=0)
                _plot_band(ax3r, x, co_mean, co_std, label=run_name, color=c)

    axes = {
        "dist_from_init": ax1l,
        "loss": ax1r,
        "param_norm_fc1": ax2l,
        "param_norm_fc2": ax2r,
    }
    if has_jacobian_any and ax3l is not None and ax3r is not None:
        axes["jacobian_dist_hist_l2"] = ax3l
        axes["jacobian_dist_hist_co"] = ax3r

    titles = {
        "dist_from_init": "param distance from init",
        "loss": "loss",
        "param_norm_fc1": "param norm (fc1)",
        "param_norm_fc2": "param norm (fc2)",
    }
    if has_jacobian_any and ax3l is not None and ax3r is not None:
        titles.update({
            "jacobian_dist_hist_l2": "jacobian distance from init (L2)",
            "jacobian_dist_hist_co": "jacobian distance from init (cosine)",
        })

    log_axes = {"dist_from_init", "param_norm_fc1", "param_norm_fc2"}
    if has_jacobian_any and ax3l is not None and ax3r is not None:
        log_axes.update({"jacobian_dist_hist_l2", "jacobian_dist_hist_co"})

    for k, ax in axes.items():
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_1d_regression_curves(x_plot, curves_by_beta):
    data = load_1d_regression_data(shuffle=False)
    x_plot_np = x_plot.cpu().numpy().ravel()
    X_train_np = data["X_train"].cpu().numpy().ravel()
    y_train_np = data["y_train"].cpu().numpy().ravel()
    y_target_np = np.interp(x_plot_np, X_train_np, y_train_np)

    plt.figure(figsize=(6, 4))

    plt.plot(x_plot_np, y_target_np, "k--", label="target")
    plt.scatter(X_train_np, y_train_np, c="k", s=20, zorder=3)
    
    for beta, fs in curves_by_beta.items():
        mean = fs.mean(axis=0)
        std = fs.std(axis=0)
        label = f"β={beta:.1e}"
        plt.plot(x_plot_np, mean, label=label)
        plt.fill_between(x_plot_np, mean - std, mean + std, alpha=0.2)

    plt.xlim(-1.5, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.show()