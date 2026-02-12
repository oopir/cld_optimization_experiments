from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable
import math
import itertools
import yaml 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch

from .metric_checkpoints import Exp1Config, save_exp1_checkpoint, load_exp1_checkpoint
from .training import train_multiseed

Metrics = Dict[str, Any]
ResultsByLabel = Dict[str, Dict[int, Metrics]]

EXP1_CHECKPOINT_PREFIX = "exp1_"


@dataclass
class Exp1RunOpts:
    ckpt_dir:         Path
    save_ckpt:        bool = False # for saving progress after training
    load_ckpt:        bool = False # for plotting/extending an existing ckpt  
    load_ckpt_name:   Path | None = None
    resume_from_ckpt: bool = False
    new_total_epochs: int | None = None
    config_overrides: Optional[List[str]] = None


def _format_scalar_for_key(x: float) -> str:
    if isinstance(x, str):
        return x
    if math.isinf(float(x)):
        return "inf"
    x_float = float(x)
    if x_float.is_integer():
        return str(int(x_float))
    return str(x_float)


def build_from_config_mapping(cfg: dict) -> tuple[Exp1Config, Exp1RunOpts]:
    """
    cfg can be either:
      {
        "experiment": { ...fields of Exp1Config... },
        "run":        { ...fields of Exp1RunOpts... }
      }
    or a flat mapping of Exp1Config fields only.
    """
    exp_section = cfg.get("experiment")
    run_section = cfg.get("run")

    if exp_section is None:
        exp_kwargs = cfg
        run_kwargs = {}
    else:
        exp_kwargs = exp_section
        run_kwargs = run_section or {}
    exp_config = Exp1Config(**exp_kwargs)

    if "ckpt_dir" in run_kwargs:
        run_kwargs["ckpt_dir"] = Path(run_kwargs["ckpt_dir"]).expanduser()
    if "load_ckpt_name" in run_kwargs:
        run_kwargs["load_ckpt_name"] = Path(run_kwargs["load_ckpt_name"]).expanduser()
    run_opts = Exp1RunOpts(**run_kwargs)

    return exp_config, run_opts


def _apply_config_overrides(base: Exp1Config, override_src: Exp1Config, override_keys: Optional[Iterable[str]]) -> Exp1Config:
    if not override_keys:
        return base
    for k in override_keys:
        if k not in [
            "eta", 
            "regularization_scale", 
            "same_noise", 
            "jac_probe_size", 
            "device", 
            "print_every"
        ]:
            raise ValueError(f"Error: overriding {k} is not supported yet.")
    
    base_dict = base.__dict__
    src_dict = override_src.__dict__
    valid_keys = set(base_dict.keys())
    kwargs = {k: src_dict[k] for k in override_keys if (k in valid_keys and k in src_dict and k != "epochs")}

    if not kwargs:
        return base
    return replace(base, **kwargs)


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
    return data


def _resolve_eta(config: Exp1Config, alpha: float, beta: float) -> float:
    mode = getattr(config, "eta_mode", "scalar")
    table_path = getattr(config, "eta_table_path", None)
    default_eta = getattr(config, "eta_default", None)
    if default_eta is None:
        default_eta = config.eta

    # print(f"[eta_debug] mode={mode}, table_path={table_path}")
    # print(f"[eta_debug] alpha={alpha} (type={type(alpha)}), beta={beta} (type={type(beta)})")


    if mode == "scalar" or table_path is None:
        # print(f"[eta_debug] scalar mode or no table -> eta={config.eta}")
        return config.eta

    table = _load_eta_table(table_path)
    # print(f"[eta_debug] loaded table top-level keys: {list(table.keys())}")

    if mode == "per_beta":
        per_beta = table.get("per_beta", {})
        # print(f"[eta_debug] per_beta keys: {list(per_beta.keys())}")
        key1 = str(beta)
        key2 = str(float(beta))
        # print(f"[eta_debug] trying per_beta keys: '{key1}', '{key2}'")
        if key1 in per_beta:
            val = float(per_beta[key1])
            # print(f"[eta_debug] HIT key1 -> eta={val}")
            return val
        if key2 in per_beta:
            val = float(per_beta[key2])
            # print(f"[eta_debug] HIT key2 -> eta={val}")
            return val
        print(f"[eta_debug] MISS -> fallback eta_default={default_eta}")
        return float(default_eta)

    if mode == "per_alpha_beta":
        per_ab = table.get("per_alpha_beta", {})
        a_str = _format_scalar_for_key(alpha)
        b_str = _format_scalar_for_key(beta)
        key = f"alpha={a_str},beta={b_str}"
        # optional: debug once
        # print(f"[eta_debug] per_alpha_beta keys: {list(per_ab.keys())}")
        # print(f"[eta_debug] lookup key: {key!r}")
        if key in per_ab:
            return float(per_ab[key])
        return float(default_eta)

    print(f"[eta_debug] unknown mode '{mode}' -> scalar eta={config.eta}")
    return config.eta


def _print_exp_config(
    exp_config: Exp1Config,
    prev_config: Exp1Config | None = None,
    override_keys: Iterable[str] | None = None,
) -> None:
    print("configuration:")

    curr = asdict(exp_config)
    prev = asdict(prev_config) if prev_config is not None else {}
    override_keys = set(override_keys or ())

    for k, v in curr.items():
        if prev_config is not None and k in override_keys and k in prev and prev[k] != v:
            print(f"  {k}: {v} (previously {prev[k]})")
        else:
            print(f"  {k}: {v}")


def label_from_alpha_beta(alpha=None, beta=None, n=None):
    label = ""
    if alpha is not None:
        label += f"α={alpha:.0e} "
    if n is None:
        label += f"β={beta}"
    else:
        label += "β=" + ("inf" if math.isinf(beta) else f"{beta // n}n")
    return label


def _train_single_alpha_beta(
    config: Exp1Config,
    alpha: float = 1.0,
    beta: float = np.inf,
    gpu_ids: Optional[List[int]] = None,
    resume_paths: Optional[Dict[int, str]] = None,
    epoch_offset: int = 0,
) -> Dict[int, Metrics]:
    common = config.train_kwargs()
    common["gpu_ids"] = gpu_ids
    common["resume_paths"] = resume_paths
    common["epoch_offset"] = epoch_offset
    common["eta"] = _resolve_eta(config, alpha=alpha, beta=beta)

    results_by_seed: Dict[int, Metrics] = train_multiseed(alpha=alpha, beta=beta, **common)
    return results_by_seed


def _write_base_ckpt_data_for_beta_to_disk(
    label: str,
    base_seed_metrics: Mapping[int, Mapping[str, Any]],
    resume_root: Path,
) -> Dict[int, str]:
    """
    populate a tmp directory with the data needed to resume training from some saved state
    (initial model parameters, last model (NN + linearized) parameters, randomness state)
    """
    dirr = resume_root / label.replace("α=", "alpha_").replace("β=", "beta_")
    dirr.mkdir(parents=True, exist_ok=True)

    resume_paths: Dict[int, str] = {}

    for seed, metrics in base_seed_metrics.items():
        payload = {
            "init_model_state_dict": metrics["init_model_state_dict"],
            "start_model_state_dict": metrics["model_state_dict"],
            "start_lin_params": metrics.get("lin_params_state"),
            "rng_state": metrics.get("rng_state"),
        }
        path = dirr / f"seed_{seed}.pt"
        torch.save(payload, path)
        resume_paths[seed] = str(path)

    return resume_paths


def _tune_eta_for_pair(
    alpha: float,
    beta: float,
    device: Optional[int],
    base_config: Exp1Config,
    eta_grid: List[float],
    tuning_epochs: int,
    seed: int,
    metric_name: str,
    goal: str,
) -> Tuple[float, float, float, float]:

    cfg = replace(base_config) # each run gets its own clone config to modify
    cfg.device = device
    cfg.epochs = tuning_epochs
    cfg.seeds = [seed]
    cfg.eta_mode = "scalar"
    cfg.eta_table_path = None
    cfg.eta_default = None

    hist_key = f"{metric_name}_hist"
    best_eta: Optional[float] = None
    best_score: Optional[float] = None

    for eta in eta_grid:
        cfg.eta = eta
        seed_metrics = _train_single_alpha_beta(cfg, alpha=alpha, beta=beta, gpu_ids=None)
        
        vals = []
        for m in seed_metrics.values():
            hist = m.get(hist_key)
            if hist:
                vals.append(float(hist[-1]))  # last logged value
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


def tune_eta_for_exp1(base_config: Exp1Config, tuning_cfg: Mapping[str, Any], gpu_ids: Optional[List[int]]) -> None:
    """
    Run short training runs over an eta grid for each beta or (alpha,beta),
    pick the best eta according to the requested metric, and write a YAML table.

    tuning_cfg keys (all optional except eta_grid and output_path):
        mode: "per_beta" or "per_alpha_beta"  (default: "per_beta")
        eta_grid: list of floats (required)
        tuning_epochs: int (default: base_config.epochs)
        tuning_seeds: list[int] (default: base_config.seeds)
        metric: base metric name, e.g. "train_loss" (default: "train_loss")
        goal: "min" or "max" (default: "min")
        output_path: str (required)
        force: bool (default: False)
    """
    # some input validation
    eta_grid = tuning_cfg.get("eta_grid")
    if not eta_grid:
        raise ValueError("eta_tuning.eta_grid must be provided and non-empty")
    output_path = Path(tuning_cfg["output_path"]).expanduser()
    force = bool(tuning_cfg.get("force", False))   
    if output_path.exists() and not force:
        print(f"[eta_tuning] Table '{output_path}' already exists, skipping (force=False).")
        return
    if not base_config.betas:
        raise ValueError("Exp1Config.betas is empty; nothing to tune over.")

    # get config values (or default alternatives)
    mode          = tuning_cfg.get("mode", "per_beta")
    tuning_epochs = int(tuning_cfg.get("tuning_epochs", base_config.epochs))
    tuning_seeds  = list(tuning_cfg.get("tuning_seeds", base_config.seeds))
    metric_name   = tuning_cfg.get("metric", "train_loss")  # base name -> "<name>_hist"
    goal          = tuning_cfg.get("goal", "min")
    alphas        = base_config.alphas or [1.0]
    betas         = base_config.betas

    partial_args  = (base_config, eta_grid, tuning_epochs, tuning_seeds[0], metric_name, goal)

    table_per_beta: Dict[str, float] = {}
    table_per_ab: Dict[str, float] = {}

    # generate (alpha, beta) pairs to iterate over
    if mode == "per_beta":
        pairs = [(alphas[0], b) for b in betas]
    elif mode == "per_alpha_beta":
        pairs = list(itertools.product(alphas, betas))
    else:
        raise ValueError(f"Unknown eta tuning mode: {mode}")

    # GPU settings
    device = base_config.device
    base_device = device
    if gpu_ids is None:
        # this section was never tested, because we always pass GPUs determined by utils.py
        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                idx = int(device.split(":", 1)[1])
                gpu_ids = [idx]
            else:
                num_gpus = torch.cuda.device_count()
                gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [0]
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
        else:
            gpu_ids = [None]
    else:
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
        
    # if there's only one pair, no need to open new processes
    if len(pairs) == 1:
        alpha, beta = pairs[0]
        dev_str = (base_device if gpu_ids[0] is None else f"cuda:{gpu_ids[0]}")
        
        a, b, best_eta, best_score = _tune_eta_for_pair(alpha, beta, dev_str, *partial_args)
        print(f"[eta_tuning] Best eta for alpha={a}, beta={b}: {best_eta:.1e} (score={best_score:.3f})")
        
        if mode == "per_beta":
            b_str = _format_scalar_for_key(b)
            table_per_beta[b_str] = best_eta
        else:
            a_str = _format_scalar_for_key(a)
            b_str = _format_scalar_for_key(b)
            table_per_ab[f"alpha={a_str},beta={b_str}"] = best_eta
    # otherwise, work in parallel 
    else:
        max_workers = len(pairs) if gpu_ids[0] is None else min(len(pairs), len(gpu_ids))
        for batch_start in range(0, len(pairs), max_workers):
            batch_pairs = pairs[batch_start : batch_start + max_workers]
            with ProcessPoolExecutor(max_workers=len(batch_pairs)) as pool:
                futures = []
                for i, (alpha, beta) in enumerate(batch_pairs):
                    dev_str = base_device if (gpu_ids[0] is None) else f"cuda:{gpu_ids[i % len(gpu_ids)]}"
                    futures.append(pool.submit(_tune_eta_for_pair, alpha, beta, dev_str, *partial_args))
                for fut in futures:
                    a, b, best_eta, best_score = fut.result()
                    print(f"[eta_tuning] Best eta for alpha={a}, beta={b}: {best_eta} (score={best_score})")
                    if mode == "per_beta":
                        b_str = _format_scalar_for_key(b)
                        table_per_beta[b_str] = best_eta
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


def _train_over_range(
    config: Exp1Config,
    alpha_range: list = [],
    beta_range: list = [],
    gpu_ids: Optional[List[int]] = None,
    resume_root = None,
    base_results = None,
    epoch_offset: int = 0,
) -> Dict[int, Metrics]: 
    
    results: ResultsByLabel = {}
    
    if len(alpha_range) == 0:
        for beta in beta_range:
            label = label_from_alpha_beta(beta=beta, n=config.n)
            resume_paths = None
            if base_results != None:
                resume_paths = _write_base_ckpt_data_for_beta_to_disk(label, base_results[label], resume_root)
            metrics = _train_single_alpha_beta(
                config, 
                beta=beta, 
                gpu_ids=gpu_ids, 
                resume_paths=resume_paths, 
                epoch_offset=epoch_offset
            )
            results[label] = metrics
    else:
        for alpha in alpha_range:
            for beta in beta_range:
                label = label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
                resume_paths = None
                if base_results != None:
                    resume_paths = _write_base_ckpt_data_for_beta_to_disk(label, base_results[label], resume_root)
                metrics = _train_single_alpha_beta(
                    config, 
                    alpha=alpha, 
                    beta=beta, 
                    gpu_ids=gpu_ids, 
                    resume_paths=resume_paths, 
                    epoch_offset=epoch_offset
                )
                results[label] = metrics
    
    return results


def _merge_metrics(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    base_keys = set(base.keys())
    extra_keys = set(extra.keys())
    if base_keys != extra_keys:
        raise ValueError(f"Metric keys differ between base and extra runs: {base_keys ^ extra_keys}")

    merged: Dict[str, Any] = {}

    for k in base_keys:
        b = base[k]
        e = extra[k]

        if k.endswith("_hist"):
            if b is None:
                merged[k] = e
            elif e is None:
                merged[k] = b
            elif isinstance(b, list) and isinstance(e, list):
                merged[k] = b + e
            elif torch.is_tensor(b) and torch.is_tensor(e):
                merged[k] = torch.cat([b, e], dim=0)
            elif isinstance(b, np.ndarray) and isinstance(e, np.ndarray):
                merged[k] = np.concatenate([b, e], axis=0)
            else:
                raise ValueError("Metric histogram concatenation failed.")
        elif k == "init_model_state_dict":
            merged[k] = b
        elif k in {"model_state_dict", "lin_params_state"}:
            if b is not None and e is None:
                raise ValueError(f"New run is missing {k} while base run has it.")
            merged[k] = e if e is not None else b
        else:
            merged[k] = e

    return merged


def resume_from_ckpt(
    base_results: ResultsByLabel,
    config: Exp1Config,
    new_epochs: int,
    gpu_ids: List[int],
    tmp_dir: Path,
) -> Tuple[ResultsByLabel, Exp1Config]:
    # some validation
    if new_epochs <= config.epochs:
        raise ValueError(f"new_epochs ({new_epochs}) must be > existing epochs ({config.epochs})")
    if len(config.alphas) == 0:
        expected_labels = [label_from_alpha_beta(beta=beta, n=config.n) for beta in config.betas]
    else:
        expected_labels = [
            label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
            for alpha in config.alphas
            for beta in config.betas
        ]
    if set(base_results.keys()) != set(expected_labels):
        raise ValueError("Checkpoint alpha/betas do not match config.alphas/config.betas.")

    # create tmp directory for passing data to child processes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_root = tmp_dir / f"exp1_resume_{timestamp}"
    resume_root.mkdir(parents=True, exist_ok=True)

    # train
    print(f"extending to a new total of {new_epochs} epochs...")
    extra_cfg = replace(config, epochs=new_epochs)
    new_results: ResultsByLabel = _train_over_range(extra_cfg, [], config.betas, gpu_ids, resume_root, base_results, config.epochs)

    # merge base + new
    merged_results: ResultsByLabel = {}
    for label in expected_labels:
        base_seed_metrics  = base_results[label]
        extra_seed_metrics = new_results[label]

        merged_seed_metrics: Dict[int, Metrics] = {}
        for seed in config.seeds:
            merged_seed_metrics[seed] = _merge_metrics(base_seed_metrics[seed], extra_seed_metrics[seed])
        merged_results[label] = merged_seed_metrics

    new_config = replace(config, epochs=new_epochs)
    return merged_results, new_config


def run_exp1(config: Exp1Config, run_opts: Exp1RunOpts, gpu_ids: List[int],) -> Tuple[ResultsByLabel, Exp1Config]:
    run_opts.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if run_opts.load_ckpt:
        # load ckpt
        load_ckpt_path = run_opts.ckpt_dir / run_opts.load_ckpt_name
        base_results, base_config = load_exp1_checkpoint(str(load_ckpt_path))
        print(f"Loaded checkpoint: {load_ckpt_path}")
        
        # resume run if needed
        if run_opts.resume_from_ckpt:
            if run_opts.new_total_epochs is None:
                raise ValueError("new_total_epochs must be set when resume_from_ckpt is True")
            
            exp_config = _apply_config_overrides(base_config, config, run_opts.config_overrides)

            override_keys = (run_opts.config_overrides or {})
            _print_exp_config(exp_config, prev_config=base_config, override_keys=override_keys)

            results, exp_config = resume_from_ckpt(
                base_results=base_results,
                config=exp_config,
                new_epochs=run_opts.new_total_epochs,
                gpu_ids=gpu_ids,
                tmp_dir=run_opts.ckpt_dir,
            )
        # otherwise, go with existing ckpt data
        else:
            exp_config = base_config
            _print_exp_config(exp_config)
            results = base_results 
    else:
        exp_config = config
        _print_exp_config(exp_config)
        results = _train_over_range(config, [], config.betas, gpu_ids)

    if run_opts.save_ckpt:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = run_opts.ckpt_dir / f"{EXP1_CHECKPOINT_PREFIX}_{exp_config.dataset}_{timestamp}.pt"
        save_exp1_checkpoint(str(ckpt_path), results, exp_config)
        print(f"Saved checkpoint: {ckpt_path}")

    return results, exp_config


# required for plotting the ICML-deadline ckpt
def infer_effective_track_every(results: ResultsByLabel, config: Exp1Config) -> int:
    beta_key = next(iter(results))
    seed_key = next(iter(results[beta_key]))
    hist = results[beta_key][seed_key]["train_loss_hist"]
    L = len(hist)

    eff = config.track_every
    if L <= 1:
        return eff

    E = config.epochs
    low = (E - 1) // L + 1
    high = (E - 1) // (L - 1)

    if low <= high and not low <= eff <= high:
        eff = low

    return eff
