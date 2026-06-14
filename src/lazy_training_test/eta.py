from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import itertools
import math

import numpy as np
import yaml

from ..base.parallel import resolve_worker_gpu_ids, worker_device
from ..config import ExpConfig


def _format_scalar_for_key(x: float) -> str:
    if isinstance(x, str):
        return x
    if math.isinf(float(x)):
        return "inf"
    x_float = float(x)
    if x_float.is_integer():
        return str(int(x_float))
    return str(x_float)


def _load_eta_table(path: Path) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.is_file():
        print(f"[eta] WARNING: eta_table_path '{path}' not found; using defaults.")
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f)
        if data is None:
            print(f"[eta] WARNING: loading eta from '{path}' failed; using defaults.")
            return {}
    return data


def _resolve_eta(config: ExpConfig, alpha: float, beta: float) -> float:
    mode = getattr(config, "eta_mode", "scalar")
    table_path = getattr(config, "eta_table_path", None)
    default_eta = getattr(config, "eta_default", None)
    if default_eta is None:
        default_eta = config.eta

    if mode == "scalar" or table_path is None:
        return config.eta

    table = _load_eta_table(table_path)

    if mode == "per_beta":
        per_beta = table.get("per_beta", {})
        key = _format_scalar_for_key(beta)
        if key in per_beta:
            return float(per_beta[key])
        print(f"[eta] WARNING: missing beta={key}; using eta_default={default_eta}")
        return float(default_eta)

    if mode == "per_alpha_beta":
        per_ab = table.get("per_alpha_beta", {})
        a_str = _format_scalar_for_key(alpha)
        b_str = _format_scalar_for_key(beta)
        key = f"alpha={a_str},beta={b_str}"
        if key in per_ab:
            return float(per_ab[key])
        return float(default_eta)

    print(f"[eta] WARNING: unknown mode '{mode}'; using scalar eta={config.eta}")
    return config.eta


def _tune_eta_for_pair(
    alpha: float,
    beta: float,
    device: str,
    base_config: ExpConfig,
    eta_grid: List[float],
    tuning_epochs: int,
    seed: int,
    metric_name: str,
    goal: str,
) -> Tuple[float, float, float, float]:
    from .core import _train_single_alpha_beta

    cfg = replace(base_config)
    cfg.device = device
    cfg.epochs = tuning_epochs
    cfg.seeds = [seed]
    cfg.eta_mode = "scalar"
    cfg.eta_table_path = None
    cfg.eta_default = None
    cfg.early_stop_metric = None
    cfg.early_stop_value = None
    cfg.early_stop_goal = "min"
    if cfg.tracked_metrics is not None and metric_name not in cfg.tracked_metrics:
        cfg.tracked_metrics = list(cfg.tracked_metrics) + [metric_name]

    hist_key = f"{metric_name}_hist"
    best_eta: Optional[float] = None
    best_score: Optional[float] = None

    for eta in eta_grid:
        cfg.eta = eta
        seed_metrics = _train_single_alpha_beta(cfg, alpha=alpha, beta=beta, gpu_ids=None)

        vals = []
        for metrics in seed_metrics.values():
            hist = metrics.get(hist_key)
            if hist:
                vals.append(float(hist[-1]))
        if not vals:
            raise RuntimeError(f"[eta_tuning] Metric '{hist_key}' missing for eta={eta}")

        score = float(sum(vals) / len(vals))
        if best_score is None:
            best_score = score
            best_eta = eta
        elif goal == "min" and score < best_score:
            best_score = score
            best_eta = eta
        elif goal == "max" and score > best_score:
            best_score = score
            best_eta = eta

    if best_eta is None:
        raise RuntimeError(f"[eta_tuning] Failed to pick eta for alpha={alpha}, beta={beta}")

    return alpha, beta, best_eta, best_score


def tune_eta_for_exp(base_config: ExpConfig, tuning_cfg: Mapping[str, Any], gpu_ids: Optional[List[int]]) -> None:
    """
    Run short training runs over an eta grid for each beta or (alpha,beta),
    pick the best eta according to the requested metric, and write a YAML table.
    """
    eta_grid = tuning_cfg.get("eta_grid")
    if not eta_grid:
        raise ValueError("eta_tuning.eta_grid must be provided and non-empty")
    output_path = Path(tuning_cfg["output_path"]).expanduser()
    force = bool(tuning_cfg.get("force", False))
    if output_path.exists() and not force:
        print(f"[eta_tuning] Table '{output_path}' already exists, skipping (force=False).")
        return
    if not base_config.betas:
        raise ValueError("ExpConfig.betas is empty; nothing to tune over.")

    mode = tuning_cfg.get("mode", "per_beta")
    tuning_epochs = int(tuning_cfg.get("tuning_epochs", base_config.epochs))
    tuning_seeds = list(tuning_cfg.get("tuning_seeds", base_config.seeds))
    metric_name = tuning_cfg.get("metric", "train_loss")
    goal = tuning_cfg.get("goal", "min")
    alphas = base_config.alphas or [1.0]
    betas = base_config.betas or [np.inf]

    partial_args = (base_config, eta_grid, tuning_epochs, tuning_seeds[0], metric_name, goal)

    table_per_beta: Dict[str, float] = {}
    table_per_ab: Dict[str, float] = {}

    if mode == "per_beta":
        pairs = [(alphas[0], b) for b in betas]
    elif mode == "per_alpha_beta":
        pairs = list(itertools.product(alphas, betas))
    else:
        raise ValueError(f"Unknown eta tuning mode: {mode}")

    base_device = base_config.device
    gpu_ids = resolve_worker_gpu_ids(base_device, gpu_ids)

    if len(pairs) == 1:
        alpha, beta = pairs[0]
        dev_str = worker_device(base_device, gpu_ids, 0)
        a, b, best_eta, best_score = _tune_eta_for_pair(alpha, beta, dev_str, *partial_args)
        print(f"[eta_tuning] Best eta for alpha={a}, beta={b}: {best_eta:.1e} (score={best_score:.3f})")

        if mode == "per_beta":
            table_per_beta[_format_scalar_for_key(b)] = best_eta
        else:
            a_str = _format_scalar_for_key(a)
            b_str = _format_scalar_for_key(b)
            table_per_ab[f"alpha={a_str},beta={b_str}"] = best_eta
    else:
        max_workers = len(pairs) if gpu_ids[0] is None else min(len(pairs), len(gpu_ids))
        for batch_start in range(0, len(pairs), max_workers):
            batch_pairs = pairs[batch_start : batch_start + max_workers]
            with ProcessPoolExecutor(max_workers=len(batch_pairs)) as pool:
                futures = []
                for i, (alpha, beta) in enumerate(batch_pairs):
                    dev_str = worker_device(base_device, gpu_ids, i)
                    futures.append(pool.submit(_tune_eta_for_pair, alpha, beta, dev_str, *partial_args))
                for fut in futures:
                    a, b, best_eta, best_score = fut.result()
                    print(f"[eta_tuning] Best eta for alpha={a}, beta={b}: {best_eta} (score={best_score})")
                    if mode == "per_beta":
                        table_per_beta[_format_scalar_for_key(b)] = best_eta
                    else:
                        a_str = _format_scalar_for_key(a)
                        b_str = _format_scalar_for_key(b)
                        table_per_ab[f"alpha={a_str},beta={b_str}"] = best_eta

    data: Dict[str, Any] = {}
    if mode == "per_beta":
        data["per_beta"] = table_per_beta
    else:
        data["per_alpha_beta"] = table_per_ab

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(data, f)

    print(f"[eta_tuning] Saved tuned etas to '{output_path}'", flush=True)
