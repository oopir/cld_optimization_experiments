from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def plot_ex1_multiseed(results, epochs, track_every):
    # check if any run actually tracked jacobian distances
    has_jacobian_any = any(
        any("jacobian_dist_hist" in r for r in run_results_by_seed.values())
        for run_results_by_seed in results.values()
    )

    # ------------------------- figure config ------------------------- #
    # ('axes' dict is used later, so don't push this section to the end)
    if has_jacobian_any:
        plt.figure(figsize=(8, 12))
        gs   = gridspec.GridSpec(3, 2)
        ax1l = plt.subplot(gs[0, 0])   # first  row left
        ax1r = plt.subplot(gs[0, 1])   # first  row right
        ax2l = plt.subplot(gs[1, 0])   # second row left
        ax2r = plt.subplot(gs[1, 1])   # second row right
        ax3l = plt.subplot(gs[2, 0])   # third  row left
    else:
        plt.figure(figsize=(8, 8))
        gs   = gridspec.GridSpec(2, 2)
        ax1l = plt.subplot(gs[0, 0])   # first row left
        ax1r = plt.subplot(gs[0, 1])   # first row right
        ax2l = plt.subplot(gs[1, 0])   # second row left

    axes = {
        "dist_from_init": ax1l,
        "dist_from_lin": ax1r
    }
    if has_jacobian_any:
        axes["jacobian_dist_hist_l2"] = ax2l
        axes["jacobian_dist_hist_co"] = ax2r
        axes["train_loss"] = ax3l
    else:
        axes["train_loss"] = ax2l

    titles = {
        "dist_from_init": "param distance from init",
        "dist_from_lin": "param distance from linearized params",
        "train_loss": "train loss",
    }
    if has_jacobian_any:
        titles.update({
            "jacobian_dist_hist_l2": "jacobian distance from init (L2)",
            "jacobian_dist_hist_co": "jacobian distance from init (cosine)",
        })

    log_axes = set()
    # if has_jacobian_any:
    #     log_axes.update({"jacobian_dist_hist_l2", "jacobian_dist_hist_co"})

    # ------------------------ actual plotting ------------------------ #  
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    x      = np.arange(track_every, epochs+1, track_every)

    for run_name, run_results_by_seed in results.items():
        c = next(colors)

        # distance from init (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "param_dist_hist")
        _plot_band(axes["dist_from_init"], x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_param_dist_hist")
        _plot_band(axes["dist_from_init"], x, mean, std, label=f"{run_name} lin", color=c, lin=True)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "NN_to_lin_hist")
        _plot_band(axes["dist_from_lin"], x, mean, std, label=run_name, color=c)
        # the following was commented-out a while ago and needs updating
        # upper_bound_by_seed = np.asarray([r["param_dist_upper_bound"] for r in run_results_by_seed.values()])
        # ax1l.axhline(y=upper_bound_by_seed.mean(), linestyle='--', color='black')

        # accuracy/loss (nonlinear vs linearized)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=run_name, color=c)
        mean, std = _mean_std_across_seeds(run_results_by_seed, "lin_train_loss_hist")
        _plot_band(axes["train_loss"], x, mean, std, label=f"{run_name} lin", color=c, lin=True)

        # jacobian distances (only for runs that actually have them)
        if has_jacobian_any:
            if all("jacobian_dist_hist" in r for r in run_results_by_seed.values()):
                jac_histories = [np.asarray(r["jacobian_dist_hist"]) for r in run_results_by_seed.values()]
                jac_arr = np.stack(jac_histories, axis=0)  # (n_seeds, T, 2)

                l2_mean = jac_arr[:, :, 0].mean(axis=0)
                l2_std  = jac_arr[:, :, 0].std(axis=0)
                _plot_band(axes["jacobian_dist_hist_l2"], x, l2_mean, l2_std, label=run_name, color=c)

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
