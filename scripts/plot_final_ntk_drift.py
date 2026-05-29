#!/usr/bin/env python3
import argparse, math, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths and imports "
        f"resolve consistently:\n  cd {REPO_ROOT}\n  python scripts/plot_final_ntk_drift.py ..."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_checkpoint
from src.data import load_digits_data, load_mnist_data
from src.linearized import compute_param_jacobians
from src.model import TwoLayerNet
from src.utils import select_idle_gpus_for_experiment


def fnum(x):
    """Convert beta-like values, including inf strings, to float."""
    if isinstance(x, str) and x.lower() in {"inf", "infty", "infinity", "∞"}:
        return float("inf")
    return float(x)


def run_label(alpha_for_label, beta, n):
    """Return the result-dict label used by the experiment code."""
    beta = fnum(beta)
    beta_part = "inf" if math.isinf(beta) else f"β={int(beta // n)}n"
    return beta_part if alpha_for_label is None else f"α={alpha_for_label:.0e} {beta_part}"


def pick_alpha(config, alpha):
    """Choose the alpha used for model reconstruction."""
    alphas = list(getattr(config, "alphas", []) or [])

    if not alphas:
        if alpha is not None:
            raise ValueError("--alpha was given, but checkpoint has no alpha range.")
        return 1.0

    if len(alphas) > 1 and alpha is None:
        raise ValueError(f"Checkpoint has multiple alphas {alphas}; pass --alpha.")

    chosen = float(alphas[0]) if alpha is None else float(alpha)
    if not any(math.isclose(chosen, float(a), rel_tol=1e-12, abs_tol=1e-12) for a in alphas):
        raise ValueError(f"--alpha={chosen} is not in checkpoint alphas {alphas}.")

    return chosen


def load_data(config, seed, device):
    """Recreate the training data for one seed."""
    kwargs = dict(n=config.n, random_labels=config.random_labels, device=device, seed=int(seed))
    dataset = getattr(config, "dataset", "digits")

    if dataset == "digits":
        return load_digits_data(**kwargs)
    if dataset == "mnist":
        return load_mnist_data(**kwargs, reserve_last=getattr(config, "reserve_last", 1000))

    raise ValueError(f"Unknown dataset={dataset!r}.")


def load_model(state, data, config, alpha, device):
    """Recreate a TwoLayerNet and load one saved state dict."""
    model = TwoLayerNet(
        d_in=data["d_in"], m=config.m, d_out=data["d_out"], init_type=config.init_type, alpha=alpha,
    ).to(device)

    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    return model


def dist(A, B, eps=1e-12):
    """Return normalized L2 drift and cosine drift between two tensors."""
    A, B = A.reshape(-1).double(), B.reshape(-1).double()
    nA, nB = torch.linalg.vector_norm(A), torch.linalg.vector_norm(B)

    l2 = torch.linalg.vector_norm(A - B) / (nB + eps)
    cos = torch.dot(A, B) / (nA * nB + eps)

    return float(l2), float(1.0 - torch.clamp(cos, -1.0, 1.0))


def seed_row(metrics, config, seed, alpha, device, probe_size):
    """Compute final Jacobian and NTK drifts for one seed."""
    def jacobian(model, X):
        """Return the flattened-output Jacobian as one matrix."""
        return torch.cat(compute_param_jacobians(model, X), dim=1)

    data = load_data(config, seed, device)
    X = data["X_train"] if probe_size is None else data["X_train"][:probe_size]

    init = load_model(metrics["init_model_state_dict"], data, config, alpha, device)
    final = load_model(metrics["model_state_dict"], data, config, alpha, device)

    J0, J1 = jacobian(init, X), jacobian(final, X)
    K0, K1 = J0 @ J0.T, J1 @ J1.T

    jac_l2, jac_cos = dist(J1, J0)
    ntk_l2, ntk_cos = dist(K1, K0)

    return {
        "jac_l2": jac_l2,
        "jac_cos": jac_cos,
        "ntk_l2": ntk_l2,
        "ntk_cos": ntk_cos,
    }


def x_positions(betas, n):
    """Map beta/n values to finite plot positions and tick labels."""
    vals = np.array([fnum(b) / n for b in betas], dtype=float)
    finite = vals[np.isfinite(vals)]

    # Plot finite beta/n values at their true positions; replace beta=inf by a finite
    # placeholder just to the right of the largest finite value, labeled as ∞.
    if len(finite) == 0:
        pos = np.arange(len(vals), dtype=float)
    else:
        pos = vals.copy()
        if np.any(~np.isfinite(pos)):
            unique = np.unique(np.sort(finite))
            gap = np.median(np.diff(unique)) if len(unique) > 1 else max(1.0, finite.max())
            pos[~np.isfinite(pos)] = finite.max() + gap

    labels = [r"$\infty$" if not np.isfinite(v) else f"{v:g}" for v in vals]
    return pos, labels


def mean_std(rows, key):
    """Compute mean and standard deviation of one metric over seeds."""
    vals = np.array([r[key] for r in rows], dtype=float)
    return vals.mean(), vals.std()


def plot_pdf(path, betas, grouped, n, kind):
    """Save one PDF plot for either cosine or normalized L2 drift."""
    x, labels = x_positions(betas, n)
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)

    for prefix, name in [("jac", "Jacobian drift"), ("ntk", "NTK drift")]:
        means, stds = zip(*(mean_std(rows, f"{prefix}_{kind}") for rows in grouped))
        ax.scatter(x, means, label=name)
        ax.plot(x, means)
        ax.errorbar(x, means, yerr=stds, fmt="none", capsize=3)

    ax.set_title(
        "Final Jacobian and NTK cosine drift"
        if kind == "cos"
        else "Final Jacobian and NTK normalized L2 drift"
    )
    ax.set_xlabel(r"$\beta/n$")
    ax.set_ylabel("distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def resolve_device(device_mode):
    """Return a torch device string from a user-facing cpu/gpu mode."""
    if device_mode == "cpu":
        return "cpu"

    gpu_ids = select_idle_gpus_for_experiment(device="cuda", util_threshold=1)
    if gpu_ids == [None]:
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return f"cuda:{gpu_ids[0]}"


def main():
    """Parse arguments, compute drifts, and save PDF figure(s)."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--outdir", default=None)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--probe-size", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--distance", choices=["l2", "cos", "both"], default="cos")
    args = p.parse_args()
    device = resolve_device(args.device)
    print(f"using device: {device}")

    results, config = load_checkpoint(args.checkpoint)
    has_alpha_sweep = bool(getattr(config, "alphas", []) or [])
    alpha = pick_alpha(config, args.alpha)

    betas = list(config.betas)
    grouped = []

    for beta in betas:
        label = run_label(alpha if has_alpha_sweep else None, beta, config.n)
        if label not in results:
            raise KeyError(f"Missing result label {label!r}. Available: {list(results)}")

        rows = [
            seed_row(metrics, config, seed, alpha, device, args.probe_size)
            for seed, metrics in results[label].items()
        ]
        grouped.append(rows)

    outdir = Path(args.outdir or Path(args.checkpoint).resolve().parent)
    outdir.mkdir(parents=True, exist_ok=True)

    kinds = ["l2", "cos"] if args.distance == "both" else [args.distance]
    alpha_tag = "" if not has_alpha_sweep else f"_alpha_{alpha:.0e}".replace("+", "")

    for kind in kinds:
        suffix = "cosine" if kind == "cos" else "normalized_l2"
        path = outdir / f"final_jacobian_ntk_drifts{alpha_tag}_{suffix}.pdf"
        plot_pdf(path, betas, grouped, config.n, kind)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
