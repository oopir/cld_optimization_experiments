#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(f"Run this script from the repository root:\n  cd {REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None
os.environ["PYTHONPATH"] = (
    str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
)
SAVED_RE = re.compile(r"Saved checkpoint:\s*(.+)")


def checkpoint_path(path: Path) -> Path:
    if path.suffix != ".log":
        return path.expanduser()
    matches = SAVED_RE.findall(path.read_text())
    if not matches:
        raise ValueError(f"No 'Saved checkpoint:' line found in {path}")
    return Path(matches[-1]).expanduser()


def label_for(alpha, beta, n):
    prefix = f"α={alpha:.0e} " if alpha is not None else ""
    return prefix + (
        "inf" if math.isinf(float(beta)) else f"β={int(float(beta) // n)}n"
    )


def forward_layers(weights, X, torch):
    z, a, value = [], [], X
    for weight in weights:
        zi = value.matmul(weight.t())
        value = torch.tanh(zi)
        z.append(zi)
        a.append(value)
    return z, a


def deltas(weights, output, z, torch):
    out = [None] * len(weights)
    out[-1] = (1 - torch.tanh(z[-1]).pow(2)).unsqueeze(-1) * output.t().unsqueeze(0)
    for idx in range(len(weights) - 2, -1, -1):
        message = torch.einsum("buc,uk->bkc", out[idx + 1], weights[idx + 1])
        out[idx] = (1 - torch.tanh(z[idx]).pow(2)).unsqueeze(-1) * message
    return out


def matrix_distance(left, right, torch):
    left, right = left.reshape(-1).double(), right.reshape(-1).double()
    l2 = torch.linalg.vector_norm(left - right) / (
        torch.linalg.vector_norm(right) + 1e-12
    )
    cosine = torch.dot(left, right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-12
    )
    return float(l2), float(1 - torch.clamp(cosine, -1, 1))


def ntk(weights, output, X, d_out, torch):
    z, activations = forward_layers(weights, X, torch)
    ds = deltas(weights, output, z, torch)
    size = X.shape[0] * d_out
    kernel = torch.zeros((size, size), device=X.device)
    for idx, delta in enumerate(ds):
        previous = X if idx == 0 else activations[idx - 1]
        previous_dot = previous.matmul(previous.t())
        block = (
            torch.einsum("buc,xuv->bcxv", delta, delta) * previous_dot[:, None, :, None]
        )
        kernel += block.reshape(size, size)
    gram = activations[-1].matmul(activations[-1].t())
    for cls in range(d_out):
        kernel[cls::d_out, cls::d_out] += gram
    return kernel


def seed_row(loaded, label, seed, alpha, device, probe_size, torch):
    from src.checkpoint import apply_resume_state_to_model, load_resume_state
    from src.model import MLP
    from src.training import load_data_for_seed
    from src.metrics import streamed_jacobian_drift

    resume_state = load_resume_state(loaded, label, seed)
    data = load_data_for_seed(loaded.config, seed, device)
    model = MLP(data["d_in"], data["d_out"], loaded.config, device, alpha, seed)
    apply_resume_state_to_model(model, resume_state, device)
    X = data["X_train"] if probe_size is None else data["X_train"][:probe_size]
    jac_l2, jac_cos = streamed_jacobian_drift(model, X, batch_size=X.shape[0])
    curr = ntk(list(model.hidden), model.output, X, data["d_out"], torch)
    initial = ntk(list(model.init_hidden), model.init_output, X, data["d_out"], torch)
    ntk_l2, ntk_cos = matrix_distance(curr, initial, torch)
    return {"jac_l2": jac_l2, "jac_cos": jac_cos, "ntk_l2": ntk_l2, "ntk_cos": ntk_cos}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot final Jacobian/NTK drift from single-device checkpoints."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--probe-size", type=int, default=None)
    parser.add_argument("--distance", choices=["l2", "cos", "both"], default="cos")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha to analyze when the checkpoint sweeps alpha.",
    )
    parser.add_argument("--exclude-beta-inf", action="store_true")
    args = parser.parse_args()
    if args.probe_size is not None and args.probe_size < 1:
        parser.error("--probe-size must be >= 1")
    return args


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from src.checkpoint import load_checkpoint_with_metadata

    device = torch.device(
        "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    )
    loaded = [
        load_checkpoint_with_metadata(checkpoint_path(path)) for path in args.inputs
    ]
    reference = loaded[0]
    reference_config = asdict(reference.config)
    reference_config.pop("seeds", None)
    reference_config.pop("gpu_indices", None)
    for item in loaded:
        current = asdict(item.config)
        current.pop("seeds", None)
        current.pop("gpu_indices", None)
        if current != reference_config:
            raise ValueError(
                "Input checkpoint configurations differ outside their seeds."
            )

    betas = [
        beta
        for beta in (reference.config.betas or [math.inf])
        if not (args.exclude_beta_inf and math.isinf(float(beta)))
    ]
    if not betas:
        raise ValueError("No beta values remain after filtering.")

    alphas = list(reference.config.alphas or [])
    if args.alpha is not None and not alphas:
        raise ValueError("--alpha was given, but checkpoint has no alpha sweep.")
    if len(alphas) > 1 and args.alpha is None:
        raise ValueError(f"Checkpoint has multiple alphas {alphas}; pass --alpha.")
    alpha = args.alpha if args.alpha is not None else (float(alphas[0]) if alphas else 1.0)
    if alphas and not any(math.isclose(float(alpha), float(value)) for value in alphas):
        raise ValueError(f"--alpha={alpha} is not in checkpoint alphas {alphas}.")
    label_alpha = alpha if alphas else None

    rows = {}
    for beta in betas:
        label = label_for(label_alpha, beta, reference.config.n)
        rows[label] = []
        for item in loaded:
            if label not in item.results:
                raise KeyError(f"Missing result label {label!r}. Available: {list(item.results)}")
            rows[label].extend(
                seed_row(item, label, int(seed), float(alpha), device, args.probe_size, torch)
                for seed in item.results[label]
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = ["l2", "cos"] if args.distance == "both" else [args.distance]
    x = list(range(len(betas)))
    labels = [
        (
            r"$\infty$"
            if math.isinf(float(beta))
            else f"{float(beta) / reference.config.n:g}"
        )
        for beta in betas
    ]
    for kind in kinds:
        fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
        for prefix, name in (("jac", "Jacobian drift"), ("ntk", "NTK drift")):
            means = [
                np.mean(
                    [
                        row[f"{prefix}_{kind}"]
                        for row in rows[label_for(label_alpha, beta, reference.config.n)]
                    ]
                )
                for beta in betas
            ]
            stds = [
                np.std(
                    [
                        row[f"{prefix}_{kind}"]
                        for row in rows[label_for(label_alpha, beta, reference.config.n)]
                    ]
                )
                for beta in betas
            ]
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=name)
        ax.set(xlabel=r"$\beta/n$", ylabel="distance", xticks=x, xticklabels=labels)
        ax.legend(frameon=False)
        suffix = "cosine" if kind == "cos" else "normalized_l2"
        path = outdir / f"final_jacobian_ntk_drifts_{suffix}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
