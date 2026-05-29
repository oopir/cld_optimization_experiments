import argparse
import sys
import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths and imports "
        f"resolve consistently:\n  cd {REPO_ROOT}\n  python scripts/generate_prediction_vectors.py ..."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import torch
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.datasets._openml",
)

from src.config import ExpConfig, load_checkpoint
from src.model import TwoLayerNet
from src.linearized import linearized_forward
from src.utils import select_idle_gpus_for_experiment


def _get_unused_digits(config: ExpConfig, num_points: int = 100, device: str = "cpu", sample_seed: int = 0):
    """
    construct `num_points` digits points that were not used in *any training set* of any run 
    in the given ExpConfig. points that appear in a test set are treated as unused.
    """
    # get the full digits dataset, with the same preprocessing as in load_digits_data
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0 
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1])
    X = X.astype(np.float32)
    y = digits.target.astype(np.int64)

    n_total = X.shape[0]
    all_idx = np.arange(n_total)
    used_mask = np.zeros(n_total, dtype=bool)

    # mark data used in training as used
    for seed in config.seeds:
        idx_train, _, y_train, _ = train_test_split(all_idx, y, train_size=config.n, stratify=y, random_state=seed)
        used_mask[idx_train] = True
    unused_idx = all_idx[~used_mask]
    print(f"Total points: {n_total}, used (train only): {used_mask.sum()}, unused: {unused_idx.shape[0]}")

    # choose `num_points` unused points at random
    if unused_idx.shape[0] < num_points:
        raise ValueError(f"Requested {num_points} unused points, but only {unused_idx.shape[0]} are available.")
    rng = np.random.default_rng(sample_seed)
    chosen_idx = rng.choice(unused_idx, size=num_points, replace=False)
    X_sel = torch.tensor(X[chosen_idx], device=device)
    y_sel = torch.tensor(y[chosen_idx], device=device)
    return chosen_idx, X_sel, y_sel


def _get_unused_mnist(
    config: ExpConfig,
    num_points: int = 100,
    device: str = "cpu",
    reserve_last=1000,
    sample_seed: int = 0,
):
    # download data from web
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32)
    y = mnist["target"].astype(np.int64)
    
    # preprocessing
    X = X / 255.0
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1])
    X = X.astype(np.float32)
    
    # choose `num_points` unused points from the last `reserved_last` datapoints 
    n_total  = X.shape[0]
    last_idx = np.arange(n_total - reserve_last, n_total)
    rng = np.random.default_rng(sample_seed)
    chosen_idx = rng.choice(last_idx, size=num_points, replace=False)
    X_sel = torch.tensor(X[chosen_idx], device=device)
    y_sel = torch.tensor(y[chosen_idx], device=device)

    return chosen_idx, X_sel, y_sel


def get_oos_points(
    config: ExpConfig,
    num_points: int,
    device: str,
    sample_seed: int,
):
    dataset = getattr(config, "dataset", "digits")
    if dataset == "digits":
        return _get_unused_digits(config, num_points=num_points, device=device, sample_seed=sample_seed)
    if dataset == "mnist":
        return _get_unused_mnist(
            config,
            num_points=num_points,
            device=device,
            reserve_last=getattr(config, "reserve_last", 1000),
            sample_seed=sample_seed,
        )
    raise ValueError(f"Unsupported dataset={dataset!r}.")


def compute_oos_predictions(
    ckpt_path: str,
    num_points: int = 100,
    device: str = "cpu",
    save: bool = True,
    save_dir: Optional[str] = None,
    sample_seed: int = 0,
):
    """
    For every (nn_model, linearized_model) pair in an Exp1 checkpoint:
      - compute predictions on out-of-sample points,
      - flatten outputs into vectors,
      - optionally save everything to disk.
    """
    results, config = load_checkpoint(ckpt_path)
    idx_oos, X_oos, y_oos = get_oos_points(
        config,
        num_points=num_points,
        device=device,
        sample_seed=sample_seed,
    )

    all_preds = {}
    for k, by_seed in results.items():
        preds_of_k = {}
        for seed, metrics in by_seed.items():
            if "lin_params_state" not in metrics:
                continue

            init_state = metrics["init_model_state_dict"]
            model_state = metrics["model_state_dict"]
            lin_state = metrics["lin_params_state"]

            fc1_w = init_state["fc1.weight"]
            fc2_w = init_state["fc2.weight"]
            d_in = fc1_w.shape[1]
            m = fc1_w.shape[0]
            d_out = fc2_w.shape[0]

            nn_model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=config.init_type).to(device)
            nn_model.load_state_dict(model_state)
            nn_model.eval()
            with torch.no_grad():
                nn_out = nn_model(X_oos).detach().cpu().reshape(-1)

            # move lin_state and init_state to 'device'
            base_params_dict = {name: init_state[name].to(device) for name, _ in nn_model.named_parameters()}
            lin_params = [p.to(device) for p in lin_state]
            with torch.no_grad():
                lin_out = linearized_forward(nn_model, base_params_dict, lin_params, X_oos)
                lin_out = lin_out.detach().cpu().reshape(-1)

            preds_of_k[int(seed)] = {"nn": nn_out.numpy(), "lin": lin_out.numpy()}
        all_preds[k] = preds_of_k

    payload = {
        "ckpt_path": ckpt_path,
        "config": config,
        "indices_oos": idx_oos,
        "X_oos": X_oos.cpu().numpy(),
        "y_oos": y_oos.cpu().numpy(),
        "predictions": all_preds,
    }

    if save:
        if save_dir is None:
            save_dir = os.path.dirname(ckpt_path)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_base = os.path.basename(ckpt_path)
        ckpt_stem, _ = os.path.splitext(ckpt_base)
        out_path = os.path.join(save_dir, f"{ckpt_stem}_oos_preds.pt")
        torch.save(payload, out_path)
        print(f"Saved out-of-sample predictions to {out_path}")

    return payload


def load_prediction_payload(predictions_path: str):
    payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise ValueError(f"{predictions_path} is not a prediction payload saved by this script.")
    return payload


def compute_beta_distance_matrix(payload, beta_key, metric: str = "l2"):
    """
    Build pairwise distance matrix between all prediction vectors for a given beta.
    For each seed we have two vectors: nn, lin.
    """
    preds_all = payload["predictions"]
    if beta_key not in preds_all:
        raise KeyError(f"beta_key {beta_key!r} not found in payload['predictions']")

    by_seed = preds_all[beta_key]

    vectors = []
    labels = []

    for seed in sorted(by_seed.keys()):
        entry = by_seed[seed]
        v_nn = np.asarray(entry["nn"]).reshape(-1)
        vectors.append(v_nn)
        labels.append(f"seed{int(seed)}_nn")
    for seed in sorted(by_seed.keys()):
        entry = by_seed[seed]
        v_lin = np.asarray(entry["lin"]).reshape(-1)
        vectors.append(v_lin)
        labels.append(f"seed{int(seed)}_lin")

    n_vec = len(vectors)
    D = np.zeros((n_vec, n_vec), dtype=float)

    if metric == "l2":
        for i in range(n_vec):
            for j in range(n_vec):
                diff = vectors[i] - vectors[j]
                D[i, j] = np.linalg.norm(diff)
    elif metric == "cosine":
        norms = [np.linalg.norm(v) for v in vectors]
        for i in range(n_vec):
            for j in range(n_vec):
                num = float(np.dot(vectors[i], vectors[j]))
                den = (norms[i] * norms[j]) + 1e-12
                D[i, j] = 1.0 - num / den
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return D, labels


def plot_beta_distance_heatmap(D, labels, ckpt_path, beta_key, metric, save_dir: Optional[str] = None):
    n_vec = D.shape[0]

    # root directory to place the checkpoint-specific folder in
    if save_dir is None:
        save_dir = os.path.dirname(ckpt_path)

    ckpt_base = os.path.basename(ckpt_path)
    ckpt_stem, _ = os.path.splitext(ckpt_base)
    beta_tag = str(beta_key).replace("β", "b")

    # dedicated folder per checkpoint
    ckpt_fig_dir = os.path.join(save_dir, ckpt_stem)
    os.makedirs(ckpt_fig_dir, exist_ok=True)

    out_png = os.path.join(ckpt_fig_dir, f"{beta_tag}_{metric}_dist_heatmap.png")

    fig_width = 0.6 * n_vec + 2.0
    fig_height = 0.6 * n_vec + 2.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(D)

    ax.set_xticks(np.arange(n_vec))
    ax.set_yticks(np.arange(n_vec))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(f"{metric} prediction distances for {beta_key}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved distance heatmap for {beta_key} to {out_png}")


def plot_all_distance_heatmaps(payload, ckpt_path, metrics=("l2", "cosine"), save_dir: Optional[str] = None):
    preds_all = payload["predictions"]
    # beta_keys = sorted(preds_all.keys())
    alpha_beta_keys = preds_all.keys()
    n_betas = len(alpha_beta_keys)
    n_metrics = len(metrics)

    if save_dir is None:
        save_dir = os.path.dirname(ckpt_path)

    ckpt_base = os.path.basename(ckpt_path)
    ckpt_stem, _ = os.path.splitext(ckpt_base)
    ckpt_fig_dir = os.path.join(save_dir, ckpt_stem)
    os.makedirs(ckpt_fig_dir, exist_ok=True)

    out_png = os.path.join(ckpt_fig_dir, "all_dist_heatmaps.png")

    fig, axes = plt.subplots(n_metrics, n_betas, figsize=(3*n_betas, 3*n_metrics), constrained_layout=True)
    axes = np.array(axes, ndmin=2)

    for mi, metric in enumerate(metrics):
        # precompute all matrices for this metric to share color scale
        D_dict = {}
        d_min, d_max = np.inf, -np.inf
        for beta_key in alpha_beta_keys:
            D, labels = compute_beta_distance_matrix(payload, beta_key, metric=metric)
            D_dict[beta_key] = (D, labels)
            d_min, d_max = min(d_min, float(D.min())), max(d_max, float(D.max()))

        for bi, beta_key in enumerate(alpha_beta_keys):
            # get data
            D, labels = D_dict[beta_key]
            
            # set subplot
            ax = axes[mi, bi]
            im = ax.imshow(D)
            ax.set_aspect("equal")

            if mi == n_metrics - 1:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            if bi == 0:
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, fontsize=6)
                ax.set_ylabel(metric, fontsize=9)
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])

            if mi == 0:
                ax.set_title(f"{beta_key}", fontsize=9)

        # one colorbar per metric row
        fig.colorbar(im, ax=ax, location="right", shrink=0.8)

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved all distance heatmaps to {out_png}")


def resolve_device(device_mode):
    """Return a torch device string from a user-facing cpu/gpu mode."""
    if device_mode == "cpu":
        return "cpu"

    gpu_ids = select_idle_gpus_for_experiment(device="cuda", util_threshold=1)
    if gpu_ids == [None]:
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        return "cpu"
    return f"cuda:{gpu_ids[0]}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate out-of-sample prediction vectors and distance heatmaps."
    )
    p.add_argument("checkpoint", help="Checkpoint to process.")
    p.add_argument("--outdir", default=None, help="Output directory. Defaults to the checkpoint directory.")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--num-points", type=int, default=100)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--metrics", nargs="+", choices=["l2", "cosine"], default=["l2", "cosine"])
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save the generated prediction payload next to the figures.",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Existing prediction payload to plot instead of recomputing predictions.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip heatmap generation.")
    p.add_argument(
        "--per-label-plots",
        action="store_true",
        help="Also save one heatmap per beta/alpha-beta label and metric.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = str(Path(args.checkpoint).expanduser().resolve())
    save_dir = str(Path(args.outdir).expanduser().resolve()) if args.outdir is not None else os.path.dirname(ckpt_path)
    os.makedirs(save_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    if args.predictions is None:
        payload = compute_oos_predictions(
            ckpt_path=ckpt_path,
            num_points=args.num_points,
            device=device,
            save=args.save_predictions,
            save_dir=save_dir,
            sample_seed=args.sample_seed,
        )
    else:
        payload = load_prediction_payload(str(Path(args.predictions).expanduser().resolve()))
        ckpt_path = payload.get("ckpt_path", ckpt_path)

    if args.no_plots:
        return

    metrics = tuple(args.metrics)
    if args.per_label_plots:
        for beta_key in payload["predictions"].keys():
            for metric in metrics:
                D, labels = compute_beta_distance_matrix(payload, beta_key, metric=metric)
                plot_beta_distance_heatmap(D, labels, ckpt_path, beta_key, metric, save_dir=save_dir)
    plot_all_distance_heatmaps(payload, ckpt_path, metrics=metrics, save_dir=save_dir)


if __name__ == "__main__":
    main()
