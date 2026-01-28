import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import torch
from sklearn.datasets import load_digits
from src.data import load_digits_data

from src.metric_checkpoints import Exp1Config, load_exp1_checkpoint
from src.model import TwoLayerNet
from src.linearized import linearized_forward


def _ex1_get_unused_digits(config: Exp1Config, num_points: int = 100, device: str = "cpu"):
    """
    Construct `num_points` digits points that were not used in *any* (train or test)
    split of any run in the given Exp1Config.
    Uses the real `load_digits_data` to exactly reproduce splits.
    """
    # Full digits dataset, with the *same* preprocessing as in load_digits_data
    digits = load_digits()
    X_all = digits.data.astype(np.float32) / 16.0
    X_all = X_all - np.mean(X_all, axis=1, keepdims=True)
    X_all = X_all / np.linalg.norm(X_all, axis=1, keepdims=True) * np.sqrt(X_all.shape[1])
    X_all = X_all.astype(np.float32)
    y_all = digits.target.astype(np.int64)

    n_total = X_all.shape[0]

    # Map each preprocessed sample to its index using exact bytes as key
    keys_all = [x.tobytes() for x in X_all]
    key_to_idx = {k: i for i, k in enumerate(keys_all)}

    used_mask = np.zeros(n_total, dtype=bool)

    # Recreate the actual train/test splits via your own loader
    for seed in config.seeds:
        data = load_digits_data(
            n=config.n,
            random_labels=config.random_labels,
            device="cpu",          # keep on CPU for mapping
            seed=seed,
        )
        X_train = data["X_train"].cpu().numpy().astype(np.float32)
        X_test = data["X_test"].cpu().numpy().astype(np.float32)

        for X_split in (X_train, X_test):
            for x in X_split:
                idx = key_to_idx[x.tobytes()]
                used_mask[idx] = True

    all_idx = np.arange(n_total)
    unused_idx = all_idx[~used_mask]

    print(f"Total points: {n_total}, used: {used_mask.sum()}, unused: {unused_idx.shape[0]}")

    if unused_idx.shape[0] < num_points:
        raise ValueError(
            f"Requested {num_points} unused points, but only {unused_idx.shape[0]} are available."
        )

    chosen_idx = np.random.choice(unused_idx, size=num_points, replace=False)
    X_sel = torch.tensor(X_all[chosen_idx], device=device)
    y_sel = torch.tensor(y_all[chosen_idx], device=device)
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

def main():
    ckpt_dir = "/home/ofirg/cld_checkpoints/expr1"
    ckpt_name = "exp1_digits_20260127_135649.pt"
    ckpt_path = os.path.join(ckpt_dir , ckpt_name)
    payload = compute_ex1_oos_predictions(
        ckpt_path=ckpt_path,
        device="cuda",
        save=True,
        save_dir=ckpt_dir,
    )

if __name__ == "__main__":
    main()