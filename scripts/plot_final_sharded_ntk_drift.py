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
    raise SystemExit(
        "Run this script from the repository root:\n"
        f"  cd {REPO_ROOT}\n"
        "  python3 scripts/plot_final_sharded_ntk_drift.py tmp0.log tmp1.log ..."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

SAVED_RE = re.compile(r"Saved checkpoint:\s*(.+)")


def checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix != ".log":
        return path
    matches = SAVED_RE.findall(path.read_text())
    if not matches:
        raise ValueError(f"No 'Saved checkpoint:' line found in {path}")
    return Path(matches[-1]).expanduser()


def config_cmp(config) -> dict:
    data = asdict(config)
    data.pop("seeds", None)
    data.pop("gpu_indices", None)
    return data


def fnum(x) -> float:
    if isinstance(x, str) and x.lower() in {"inf", "infty", "infinity", "∞"}:
        return math.inf
    return float(x)


def pick_alpha(config, alpha):
    alphas = list(getattr(config, "alphas", []) or [])
    if not alphas:
        if alpha is not None:
            raise ValueError("--alpha was given, but checkpoint has no alpha sweep.")
        return 1.0
    if len(alphas) > 1 and alpha is None:
        raise ValueError(f"Checkpoint has multiple alphas {alphas}; pass --alpha.")
    chosen = float(alphas[0]) if alpha is None else float(alpha)
    if not any(math.isclose(chosen, float(a), rel_tol=1e-12, abs_tol=1e-12) for a in alphas):
        raise ValueError(f"--alpha={chosen} is not in checkpoint alphas {alphas}.")
    return chosen


def x_positions(betas, n):
    vals = np.array([fnum(b) / n for b in betas], dtype=float)
    finite = vals[np.isfinite(vals)]
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
    vals = np.array([r[key] for r in rows], dtype=float)
    return vals.mean(), vals.std()


def resolve_device(mode):
    if mode == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if mode == "gpu":
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
    return torch.device("cpu")


def load_data(config, seed, device):
    from sharded.training import load_data_for_seed

    return load_data_for_seed(config, int(seed), device)


def rank_infos(root: Path, metrics: dict):
    rel = metrics.get("state_shard_dir")
    if not rel:
        raise ValueError("This checkpoint has no state_shard_dir; use sharded_state checkpoints or logs.")

    state_dir = root / rel
    infos = []
    for rank_dir in sorted(state_dir.glob("rank_*")):
        manifest_path = rank_dir / "manifest.pt"
        if manifest_path.is_file():
            infos.append((rank_dir, torch.load(manifest_path, map_location="cpu", weights_only=False)))
    if not infos:
        raise FileNotFoundError(f"No rank shard manifests found under {state_dir}")
    return infos


def load_tensor(info, name, device):
    rank_dir, manifest = info
    filename = manifest["tensor_files"][name]
    return torch.load(rank_dir / filename, map_location=device, weights_only=False)


def forward_locals(infos, X, config, prefix, device):
    a_prev = X
    z_layers, a_layers = [], []
    for layer in range(config.L):
        z_locals, a_locals = [], []
        for info in infos:
            weight = load_tensor(info, f"{prefix}hidden_{layer:03d}", device)
            z = a_prev.matmul(weight.t())
            a = torch.tanh(z)
            z_locals.append(z)
            a_locals.append(a)
            del weight
        z_layers.append(z_locals)
        a_layers.append(a_locals)
        if layer < config.L - 1:
            a_prev = torch.cat(a_locals, dim=1)
    return z_layers, a_layers


def backprop_deltas(infos, z_layers, config, prefix, device):
    deltas = [None] * config.L
    last = []
    for info, z in zip(infos, z_layers[-1]):
        output = load_tensor(info, f"{prefix}output", device)
        a_prime = 1.0 - torch.tanh(z).pow(2)
        last.append(a_prime.unsqueeze(-1) * output.t().unsqueeze(0))
        del output
    deltas[-1] = last

    for layer in range(config.L - 2, -1, -1):
        n, d_out = z_layers[layer + 1][0].shape[0], deltas[layer + 1][0].shape[-1]
        m = sum(z.shape[1] for z in z_layers[layer])
        msg_full = torch.zeros((n, m, d_out), device=device)
        for info, delta_next in zip(infos, deltas[layer + 1]):
            weight_next = load_tensor(info, f"{prefix}hidden_{layer + 1:03d}", device)
            msg_full += torch.einsum("buc,uk->bkc", delta_next, weight_next)
            del weight_next

        current = []
        start = 0
        for z in z_layers[layer]:
            end = start + z.shape[1]
            a_prime = 1.0 - torch.tanh(z).pow(2)
            current.append(a_prime.unsqueeze(-1) * msg_full[:, start:end, :])
            start = end
        deltas[layer] = current
    return deltas


def hidden_ntk_matrix(deltas, prev_dot):
    blocks = []
    for delta in deltas:
        blocks.append(torch.einsum("buc,xuv->bcxv", delta, delta))
    K = sum(blocks) * prev_dot[:, None, :, None]
    return K.reshape(K.shape[0] * K.shape[1], K.shape[2] * K.shape[3])


def output_ntk_matrix(a_locals, d_out, device):
    gram = sum(a.matmul(a.t()) for a in a_locals)
    n = gram.shape[0]
    K = torch.zeros((n, d_out, n, d_out), device=device)
    for c in range(d_out):
        K[:, c, :, c] = gram
    return K.reshape(n * d_out, n * d_out)


def jacobian_totals(curr, init, X, device):
    _, a_curr, d_curr, d_out = curr
    _, a_init, d_init, _ = init
    totals = torch.zeros(4, device=device, dtype=torch.float64)

    for layer in range(len(d_curr)):
        if layer == 0:
            curr_prev_sq = init_prev_sq = cross_prev = X.pow(2).sum(dim=1, dtype=torch.float64)
        else:
            ac = torch.cat(a_curr[layer - 1], dim=1)
            ai = torch.cat(a_init[layer - 1], dim=1)
            curr_prev_sq = ac.pow(2).sum(dim=1, dtype=torch.float64)
            init_prev_sq = ai.pow(2).sum(dim=1, dtype=torch.float64)
            cross_prev = (ac * ai).sum(dim=1, dtype=torch.float64)

        for dc, di in zip(d_curr[layer], d_init[layer]):
            dc = dc.to(torch.float64)
            di = di.to(torch.float64)
            totals[0] += (dc.pow(2).sum(dim=(1, 2)) * curr_prev_sq).sum()
            totals[1] += (di.pow(2).sum(dim=(1, 2)) * init_prev_sq).sum()
            totals[2] += ((dc * di).sum(dim=(1, 2)) * cross_prev).sum()

    ac = torch.cat(a_curr[-1], dim=1).to(torch.float64)
    ai = torch.cat(a_init[-1], dim=1).to(torch.float64)
    totals[0] += d_out * ac.pow(2).sum()
    totals[1] += d_out * ai.pow(2).sum()
    totals[2] += d_out * (ac * ai).sum()
    totals[3] = totals[0] + totals[1] - 2.0 * totals[2]
    return totals


def matrix_dist(A, B):
    A, B = A.reshape(-1).double(), B.reshape(-1).double()
    norm_a = torch.linalg.vector_norm(A)
    norm_b = torch.linalg.vector_norm(B)
    l2 = torch.linalg.vector_norm(A - B) / (norm_b + 1e-12)
    cos = torch.dot(A, B) / (norm_a * norm_b + 1e-12)
    return float(l2), float(1.0 - torch.clamp(cos, -1.0, 1.0))


def seed_row(root, metrics, config, seed, alpha, device, probe_size):
    data = load_data(config, seed, device)
    X = data["X_train"] if probe_size is None else data["X_train"][:probe_size]
    infos = rank_infos(root, metrics)

    curr_z, curr_a = forward_locals(infos, X, config, "", device)
    init_z, init_a = forward_locals(infos, X, config, "init_", device)
    curr_d = backprop_deltas(infos, curr_z, config, "", device)
    init_d = backprop_deltas(infos, init_z, config, "init_", device)

    curr = (curr_z, curr_a, curr_d, data["d_out"])
    init = (init_z, init_a, init_d, data["d_out"])
    curr_norm, init_norm, dot, diff = jacobian_totals(curr, init, X, device)
    jac_l2 = float(torch.sqrt(torch.clamp(diff, min=0.0)) / (torch.sqrt(init_norm) + 1e-12))
    jac_cos = float(1.0 - torch.clamp(dot / (torch.sqrt(curr_norm) * torch.sqrt(init_norm) + 1e-12), -1.0, 1.0))

    K_curr = torch.zeros((X.shape[0] * data["d_out"], X.shape[0] * data["d_out"]), device=device)
    K_init = torch.zeros_like(K_curr)
    for layer in range(config.L):
        if layer == 0:
            curr_prev_dot = init_prev_dot = X.matmul(X.t())
        else:
            curr_prev = torch.cat(curr_a[layer - 1], dim=1)
            init_prev = torch.cat(init_a[layer - 1], dim=1)
            curr_prev_dot = curr_prev.matmul(curr_prev.t())
            init_prev_dot = init_prev.matmul(init_prev.t())
        K_curr += hidden_ntk_matrix(curr_d[layer], curr_prev_dot)
        K_init += hidden_ntk_matrix(init_d[layer], init_prev_dot)
    K_curr += output_ntk_matrix(curr_a[-1], data["d_out"], device)
    K_init += output_ntk_matrix(init_a[-1], data["d_out"], device)
    ntk_l2, ntk_cos = matrix_dist(K_curr, K_init)

    return {"jac_l2": jac_l2, "jac_cos": jac_cos, "ntk_l2": ntk_l2, "ntk_cos": ntk_cos}


def plot_pdf(path, betas, grouped, n, kind):
    x, labels = x_positions(betas, n)
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for prefix, name in [("jac", "Jacobian drift"), ("ntk", "NTK drift")]:
        means, stds = zip(*(mean_std(rows, f"{prefix}_{kind}") for rows in grouped))
        ax.scatter(x, means, label=name)
        ax.plot(x, means)
        ax.errorbar(x, means, yerr=stds, fmt="none", capsize=3)

    ax.set_title("Final Jacobian and NTK cosine drift" if kind == "cos" else "Final Jacobian and NTK normalized L2 drift")
    ax.set_xlabel(r"$\beta/n$")
    ax.set_ylabel("distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot final Jacobian/NTK drift from sharded checkpoints.")
    p.add_argument("inputs", nargs="+", type=Path, help="Sharded checkpoint directories/results.pt files or run logs.")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    p.add_argument("--probe-size", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--distance", choices=["l2", "cos", "both"], default="cos")
    p.add_argument("--exclude-beta-inf", action="store_true")
    args = p.parse_args()
    if args.probe_size is not None and args.probe_size < 1:
        p.error("--probe-size must be >= 1")
    return args


def main():
    args = parse_args()

    global np, torch, plt
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sharded.checkpoint import load_checkpoint_with_metadata
    from sharded.exp import label_from_alpha_beta

    device = resolve_device(args.device)
    print(f"using device: {device}")

    paths = [checkpoint_path(path) for path in args.inputs]
    loaded = [load_checkpoint_with_metadata(path) for path in paths]
    ref = loaded[0]
    ref_cmp = config_cmp(ref.config)
    alpha = pick_alpha(ref.config, args.alpha)
    has_alpha_sweep = bool(getattr(ref.config, "alphas", []) or [])

    cases = {}
    for item in loaded:
        if config_cmp(item.config) != ref_cmp:
            raise ValueError(f"Config mismatch outside seeds: {item.payload_path}")
        for label, per_seed in item.results.items():
            cases.setdefault(label, [])
            for seed, metrics in per_seed.items():
                cases[label].append((item.path, int(seed), metrics))

    config_betas = list(ref.config.betas or [math.inf])
    betas = [b for b in config_betas if not (args.exclude_beta_inf and math.isinf(float(b)))]
    if not betas:
        raise ValueError("No beta values remain after filtering.")
    grouped = []
    for beta in betas:
        label = label_from_alpha_beta(alpha=alpha if has_alpha_sweep else None, beta=float(beta), n=ref.config.n)
        if label not in cases:
            raise KeyError(f"Missing result label {label!r}. Available: {list(cases)}")
        rows = [
            seed_row(root, metrics, ref.config, seed, alpha, device, args.probe_size)
            for root, seed, metrics in cases[label]
        ]
        grouped.append(rows)

    outdir = Path(args.outdir or paths[0].resolve().parent)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = ["l2", "cos"] if args.distance == "both" else [args.distance]
    alpha_tag = "" if not has_alpha_sweep else f"_alpha_{alpha:.0e}".replace("+", "")
    for kind in kinds:
        suffix = "cosine" if kind == "cos" else "normalized_l2"
        path = outdir / f"final_sharded_jacobian_ntk_drifts{alpha_tag}_{suffix}.pdf"
        plot_pdf(path, betas, grouped, ref.config.n, kind)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
