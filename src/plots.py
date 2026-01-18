import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import cycle

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

def plot_ex2(results):
    plt.figure(figsize=(8, 8))
    gs   = gridspec.GridSpec(2, 2)
    ax1l = plt.subplot(gs[0, 0])   # first  row left
    ax1r = plt.subplot(gs[0, 1])   # first  row right
    ax2l = plt.subplot(gs[1, 0])   # second row left
    ax2r = plt.subplot(gs[1, 1])   # second row right

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for name, r in results.items():
        c = next(colors)

        ax1l.plot(r["param_dist_hist"], label=f"{name}", color=c)
        ax1l.plot(r["lin_param_dist_hist"], linestyle="--", label=f"{name} lin", color=c)

        ax1r.plot(r["train_loss_hist"], label=f"{name}", color=c)
        ax1r.plot(r["lin_train_loss_hist"], linestyle="--", label=f"{name} lin", color=c)

        skip = len(r["param_norm_hist"]) // 100
        ax2l.plot(skip, r["param_norm_hist"][::skip], label=f"{name}", color=c)
        ax2l.plot(skip, r["lin_param_norm_hist"][::skip], linestyle="--", label=f"{name} lin", color=c)

        ax2r.plot(r["jacobian_dist_hist"], label=f"{name}", color=c)

    axes = {
        "dist_from_init": ax1l,
        "loss": ax1r,
        "param_norm": ax2l,
        "jacobian_dist_hist": ax2r
    }
    titles = {
        "dist_from_init": "param distance from init",
        "loss": "loss",
        "param_norm": "param norm",
        "jacobian_dist_hist": "jacobian distance from init"
    }
    log_axes = {"dist_from_init", "jacobian_dist_hist"}

    for k, ax in axes.items():
        ax.set_title(titles[k])
        ax.set_xlabel("epoch")
        if k in log_axes:
            ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.show()