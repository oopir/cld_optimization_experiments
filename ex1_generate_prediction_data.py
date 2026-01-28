import sys
import os
from pathlib import Path
from typing import Optional

ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.metric_checkpoints import Exp1Config, load_exp1_checkpoint
from src.model import TwoLayerNet
from src.linearized import linearized_forward

def _ex1_get_unused_digits(config: Exp1Config, num_points: int = 100, device: str = "cpu"):
    """
    Construct `num_points` digits points that were not used in *any training set*
    of any run in the given Exp1Config.
    Points that ever appear in a test set are treated as UNUSED.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # scale to [0,1]
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1])  # ||x|| = sqrt(d)
    X = X.astype(np.float32)
    y = digits.target.astype(np.int64)

    n_total = X.shape[0]
    all_idx = np.arange(n_total)
    used_mask = np.zeros(n_total, dtype=bool)

    # Only mark TRAIN points as used; test points are considered unused by design.
    for seed in config.seeds:
        idx_train, _, y_train, _ = train_test_split(
            all_idx,
            y,
            train_size=config.n,
            stratify=y,
            random_state=seed,
        )
        used_mask[idx_train] = True

    unused_idx = all_idx[~used_mask]
    print(f"Total points: {n_total}, used (train only): {used_mask.sum()}, unused: {unused_idx.shape[0]}")

    if unused_idx.shape[0] < num_points:
        raise ValueError(
            f"Requested {num_points} unused points, but only {unused_idx.shape[0]} are available."
        )

    chosen_idx = np.random.choice(unused_idx, size=num_points, replace=False)
    X_sel = torch.tensor(X[chosen_idx], device=device)
    y_sel = torch.tensor(y[chosen_idx], device=device)
    return chosen_idx, X_sel, y_sel


def compute_ex1_oos_predictions(
    ckpt_path: str,
    num_points: int = 100,
    device: str = "cpu",
    save: bool = True,
    save_dir: Optional[str] = None,
):
    """
    For every (nn_model, linearized_model) pair in an Exp1 checkpoint:
      - compute predictions on 100 digits points unseen in any run,
      - flatten outputs into vectors,
      - optionally save everything to disk.
    """
    results, config = load_exp1_checkpoint(ckpt_path)
    idx_oos, X_oos, y_oos = _ex1_get_unused_digits(config, num_points=num_points, device=device)

    all_preds = {}
    for beta_key, by_seed in results.items():
        beta_preds = {}
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

            lin_model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=config.init_type).to(device)
            base_params_dict = {
                name: init_state[name].to(device)
                for name, _ in lin_model.named_parameters()
            }
            lin_params = [p.to(device) for p in lin_state]
            with torch.no_grad():
                lin_out = linearized_forward(lin_model, base_params_dict, lin_params, X_oos)
                lin_out = lin_out.detach().cpu().reshape(-1)

            beta_preds[int(seed)] = {
                "nn": nn_out.numpy(),
                "lin": lin_out.numpy(),
            }
        all_preds[beta_key] = beta_preds

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
        ckpt_base = os.path.basename(ckpt_path)
        ckpt_stem, _ = os.path.splitext(ckpt_base)
        out_path = os.path.join(save_dir, f"{ckpt_stem}_oos_preds.pt")
        torch.save(payload, out_path)
        print(f"Saved out-of-sample predictions to {out_path}")

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
        v_lin = np.asarray(entry["lin"]).reshape(-1)
        vectors.append(v_nn)
        labels.append(f"seed{int(seed)}_nn")
        vectors.append(v_lin)
        labels.append(f"seed{int(seed)}_lin")

    n_vec = len(vectors)
    D = np.zeros((n_vec, n_vec), dtype=float)

    if metric == "l2":
        for i in range(n_vec):
            for j in range(n_vec):
                diff = vectors[i] - vectors[j]
                D[i, j] = np.linalg.norm(diff)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return D, labels


def plot_beta_distance_heatmap(D, labels, ckpt_path, beta_key, save_dir: Optional[str] = None):
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

    out_png = os.path.join(ckpt_fig_dir, f"{beta_tag}_dist_heatmap.png")

    fig_width = 0.6 * n_vec + 2.0
    fig_height = 0.6 * n_vec + 2.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(D)

    ax.set_xticks(np.arange(n_vec))
    ax.set_yticks(np.arange(n_vec))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(f"Prediction distances for {beta_key}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved distance heatmap for {beta_key} to {out_png}")



def main():
    ckpt_dir = "/home/ofirg/cld_checkpoints/expr1"
    ckpt_name = "exp1_digits_20260128_155418.pt"
    ckpt_path = os.path.join(ckpt_dir , ckpt_name)
    payload = compute_ex1_oos_predictions(
        ckpt_path=ckpt_path,
        device="cuda",
        save=True,
        save_dir=ckpt_dir,
    )
    # build and plot distance heatmaps for each beta
    for beta_key in payload["predictions"].keys():
        D, labels = compute_beta_distance_matrix(payload, beta_key, metric="l2")
        plot_beta_distance_heatmap(D, labels, payload["ckpt_path"], beta_key, save_dir=ckpt_dir)
 

if __name__ == "__main__":
    main()