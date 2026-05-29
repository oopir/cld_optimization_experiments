#!/usr/bin/env python3
import argparse, math, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths and imports "
        f"resolve consistently:\n  cd {REPO_ROOT}\n  python scripts/plot_feat_gram_per_seed.py ..."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_checkpoint

DISTANCE_COMPONENTS = {"l2": 0, "cosine": 1}


def metric_history(metrics, metric, distance_type):
    y = np.asarray(metrics[metric])
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 2:
        # Distance histories are stored as (normalized L2, cosine) per epoch.
        if distance_type is None:
            raise ValueError(
                f"{metric!r} has two distance components; pass --distance-type "
                "{l2,cosine}."
            )
        return y[:, DISTANCE_COMPONENTS[distance_type]]
    raise ValueError(
        f"{metric!r} must be a scalar history or a two-component distance history; "
        f"got array shape {y.shape}."
    )


def plot_panels(results, cfg, by, metric, out, yscale, lw, ylabel, title, distance_type):
    labels = list(results)
    seeds = sorted({s for by_seed in results.values() for s in by_seed})
    panels = labels if by == "beta" else seeds

    if by == "beta":
        rows, cols = 2, 2
    else:
        cols = min(3, len(panels))
        rows = math.ceil(len(panels) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), sharex=True, sharey=True, constrained_layout=False)
    axs = np.ravel([axs])

    all_x = []
    for ax, panel in zip(axs, panels):
        # Each panel fixes either one beta/alpha-beta label or one seed.
        if by == "beta":
            curves = [(f"seed {s}", results[panel][s]) for s in seeds if s in results[panel]]
            panel_title = str(panel)
            fig_title = f"{title} - plot for each beta"
        elif by == "seed":
            curves = [(label, results[label][panel]) for label in labels if panel in results[label]]
            panel_title = f"seed {panel}"
            fig_title = f"{title} - plot for each seed"
        else:
            raise ValueError("by must be 'beta' or 'seed'")

        for name, metrics in curves:
            if metric not in metrics:
                continue
            y = metric_history(metrics, metric, distance_type)
            if "epoch_hist" in metrics:
                x = np.asarray(metrics["epoch_hist"])[:len(y)]
            else:
                x = np.arange(len(y)) * cfg.track_every
            all_x.append(x)
            ax.plot(x, y, lw=lw, alpha=0.9, label=name)

        ax.set_title(panel_title)
        ax.set_yscale(yscale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

    # Remove blank subplots when the grid has more axes than panels.
    for ax in axs[len(panels):]:
        ax.remove()

    # Keep all panels on the same epoch range for easier visual comparison.
    if all_x:
        xmax = max(float(x[-1]) for x in all_x if len(x))
        for ax in axs[:len(panels)]:
            ax.set_xlim(left=0, right=1.01 * xmax)

    handles, names = axs[0].get_legend_handles_labels()

    fig.suptitle(fig_title, y=0.985)
    fig.legend(
        handles, names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(6, len(names)),
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.09,
        left=0.10,
        right=0.98,
        hspace=0.38,
        wspace=0.20,
    )

    fig.savefig(out, bbox_inches="tight", dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--metric", default="feat_gram_lambda_hist")
    p.add_argument("--distance-type", choices=sorted(DISTANCE_COMPONENTS), default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--yscale", choices=["log", "linear"], default=None)
    p.add_argument("--linewidth", type=float, default=2.0)
    p.add_argument("--beta-linewidth", type=float, default=1.4)
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    results, cfg = load_checkpoint(Path(args.ckpt).expanduser())

    # Fail before creating figures if the metric is missing or ambiguous.
    found_metric = False
    for by_seed in results.values():
        for metrics in by_seed.values():
            if args.metric not in metrics:
                continue
            found_metric = True
            metric_history(metrics, args.metric, args.distance_type)
    if not found_metric:
        raise ValueError(f"Metric {args.metric!r} was not found in any run/seed.")

    metric_stem = args.metric[:-5] if args.metric.endswith("_hist") else args.metric
    readable_metric = metric_stem.replace("_", " ")
    out_prefix = args.out_prefix or metric_stem
    if args.distance_type is not None:
        out_prefix = f"{out_prefix}_{args.distance_type}"

    if args.distance_type is None:
        default_label = r"$\lambda_{\min}$" if args.metric == "feat_gram_lambda_hist" else readable_metric
    else:
        suffix = "normalized L2" if args.distance_type == "l2" else "cosine"
        default_label = f"{readable_metric} ({suffix})"

    title = args.title or default_label
    ylabel = args.ylabel or default_label
    yscale = args.yscale or ("log" if args.metric == "feat_gram_lambda_hist" else "linear")

    plot_panels(
        results,
        cfg,
        "beta",
        args.metric,
        f"{out_prefix}_by_beta.png",
        yscale,
        args.beta_linewidth,
        ylabel,
        title,
        args.distance_type,
    )
    plot_panels(
        results,
        cfg,
        "seed",
        args.metric,
        f"{out_prefix}_by_seed.png",
        yscale,
        args.linewidth,
        ylabel,
        title,
        args.distance_type,
    )


if __name__ == "__main__":
    main()
