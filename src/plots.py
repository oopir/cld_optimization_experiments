from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_ex1(results):
    plt.figure(figsize=(8, 4))
    gs   = gridspec.GridSpec(1, 2)
    ax1l = plt.subplot(gs[0, 0])   # first  row left
    ax1r = plt.subplot(gs[0, 1])   # first  row right

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for name, r in results.items():
        c = next(colors)

        ax1l.plot(r["param_dist_hist"], label=f"{name}", color=c)
        ax1l.axhline(y=r["param_dist_upper_bound"], linestyle='--', color=c)

        ax1r.plot(r["train_acc_hist"], label=f"{name}", color=c)
        ax1r.plot(r["test_acc_hist"], linestyle="--", color=c)

    axes = {
        "dist_from_init": ax1l,
        "accuracy": ax1r
    }
    titles = {
        "dist_from_init": "param distance from init",
        "accuracy": "accuracy"
    }
    log_axes = {"dist_from_init"}

    for k, ax in axes.items():
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_with_skip(ax, y, c, *, label, linestyle="-"):
    x = range(len(y))
    skip = 1
    if len(y) >= 1000:
        skip = len(y) // 100
    ax.plot(x[::skip], y[::skip], color=c, label=label, linestyle=linestyle)

def plot_ex2(results):
    plt.figure(figsize=(8, 12))
    gs   = gridspec.GridSpec(3, 2)
    ax1l = plt.subplot(gs[0, 0])   # first  row left
    ax1r = plt.subplot(gs[0, 1])   # first  row right
    ax2l = plt.subplot(gs[1, 0])   # second row left
    ax2r = plt.subplot(gs[1, 1])   # second row right
    ax3l = plt.subplot(gs[2, 0])   # third  row left
    ax3r = plt.subplot(gs[2, 0])   # third  row right

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for name, r in results.items():
        c = next(colors)

        ax1l.plot(r["param_dist_hist"], label=f"{name}", color=c)
        ax1l.plot(r["lin_param_dist_hist"], linestyle="--", label=f"{name} lin", color=c)

        ax1r.plot(r["train_loss_hist"], label=f"{name}", color=c)
        ax1r.plot(r["lin_train_loss_hist"], linestyle="--", label=f"{name} lin", color=c)

        plot_with_skip(ax2l, r["param_norm_fc1_hist"], c, label=f"{name} (fc1)")
        plot_with_skip(ax2l, r["lin_param_norm_fc1_hist"], c, label=f"{name} lin (fc1)", linestyle="--")
        plot_with_skip(ax2r, r["param_norm_fc2_hist"], c, label=f"{name} (fc2)")
        plot_with_skip(ax2r, r["lin_param_norm_fc2_hist"], c, label=f"{name} lin (fc2)", linestyle="--")

        jac_dist_hist_l2, jac_dist_hist_co = zip(*r["jacobian_dist_hist"])
        ax3l.plot(jac_dist_hist_l2, label=f"{name}", color=c)
        ax3r.plot(jac_dist_hist_co, label=f"{name}", color=c)

    axes = {
        "dist_from_init": ax1l,
        "loss": ax1r,
        "param_norm_fc1": ax2l,
        "param_norm_fc2": ax2r,
        "jacobian_dist_hist_l2": ax3l,
        "jacobian_dist_hist_co": ax3r,
    }
    titles = {
        "dist_from_init": "param distance from init",
        "loss": "loss",
        "param_norm_fc1": "param norm (fc1)",
        "param_norm_fc2": "param norm (fc2)",
        "jacobian_dist_hist_l2": "jacobian distance from init (L2)",
        "jacobian_dist_hist_co": "jacobian distance from init (cosine)"
    }
    log_axes = {"dist_from_init", "param_norm_fc1", "param_norm_fc2", "jacobian_dist_hist_l2", "jacobian_dist_hist_co"}

    for k, ax in axes.items():
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.show()
    