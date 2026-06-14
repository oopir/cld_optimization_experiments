import numpy as np
import matplotlib.pyplot as plt

from ..lazy_training_test.plotting import _has_history, _warn_missing


def plot_test_error_vs_alpha(results, output_path="alpha_test_error.pdf"):
    """
    Plot final test error by alpha; assumes each plotted run tracks test_acc_hist.

    results: ResultsByLabel as produced by the alpha sweep, with labels like
    'α=1e+00 β=inf' and values {seed -> metrics}. Runs without test_acc_hist
    are skipped with a warning.
    """
    xs = []
    ys = []

    for label, run_results_by_seed in results.items():
        if "α" not in label:
            continue

        first_tok = label.split()[0]
        alpha_str = first_tok.split("=", 1)[1]
        alpha = float(alpha_str)

        if not _has_history(run_results_by_seed, "test_acc_hist"):
            _warn_missing(label, "test error vs alpha", "test_acc_hist")
            continue

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
