from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable
import math

import numpy as np
import torch

from .metric_checkpoints import Exp1Config, save_exp1_checkpoint, load_exp1_checkpoint
from .training import train_multiseed

Metrics = Dict[str, Any]
ResultsByBeta = Dict[str, Dict[int, Metrics]]

EXP1_CHECKPOINT_PREFIX = "exp1_digits_"


@dataclass
class Exp1RunOpts:
    ckpt_dir:         Path
    save_ckpt:        bool = False # for saving progress after training
    load_ckpt:        bool = False # for plotting/extending an existing ckpt  
    load_ckpt_name:   Path | None = None
    resume_from_ckpt: bool = False
    new_total_epochs: int | None = None
    config_overrides: Optional[List[str]] = None

def label_from_beta(beta, n=None):
    if n is None:
        return f"β={beta}"
    else:
        return "β=" + ("inf" if math.isinf(beta) else f"{beta // n}n")


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


def _write_base_ckpt_data_for_beta_to_disk(
    label: str,
    beta_results: Mapping[int, Mapping[str, Any]],
    resume_root: Path,
) -> Dict[int, str]:
    """
    populate a tmp directory with the data needed to resume training from some saved state
    (initial model parameters, last model (NN + linearized) parameters, randomness state)
    """
    beta_dir = resume_root / label.replace("β=", "beta_")
    beta_dir.mkdir(parents=True, exist_ok=True)

    resume_paths: Dict[int, str] = {}

    for seed, metrics in beta_results.items():
        payload = {
            "init_model_state_dict": metrics["init_model_state_dict"],
            "start_model_state_dict": metrics["model_state_dict"],
            "start_lin_params": metrics.get("lin_params_state"),
            "rng_state": metrics.get("rng_state"),
        }
        path = beta_dir / f"seed_{seed}.pt"
        torch.save(payload, path)
        resume_paths[seed] = str(path)

    return resume_paths


def _train_single_beta(
    config: Exp1Config,
    beta: float,
    gpu_ids: Optional[List[int]],
    resume_paths: Optional[Dict[int, str]] = None,
    epoch_offset: int = 0,
) -> Dict[int, Metrics]:
    common = config.train_kwargs()
    common["gpu_ids"] = gpu_ids
    common["resume_paths"] = resume_paths
    common["epoch_offset"] = epoch_offset
    results_by_seed: Dict[int, Metrics] = train_multiseed(beta=beta, **common)
    return results_by_seed


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
    base_results: ResultsByBeta,
    config: Exp1Config,
    new_epochs: int,
    gpu_ids: List[int],
    tmp_dir: Path,
) -> Tuple[ResultsByBeta, Exp1Config]:
    # some validation
    if new_epochs <= config.epochs:
        raise ValueError(f"new_epochs ({new_epochs}) must be > existing epochs ({config.epochs})")
    expected_labels = [label_from_beta(beta, config.n) for beta in config.betas]
    if set(base_results.keys()) != set(expected_labels):
        raise ValueError("Checkpoint betas do not match config.betas.")

    # create tmp directory for passing data to child processes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_root = tmp_dir / f"exp1_resume_{timestamp}"
    resume_root.mkdir(parents=True, exist_ok=True)

    # train
    print(f"extending to a new total of {new_epochs} epochs...")
    extra_cfg = replace(config, epochs=new_epochs)
    new_results: ResultsByBeta = {}

    for beta in config.betas:
        label = label_from_beta(beta, config.n)
        base_seed_metrics = base_results[label]

        if set(base_seed_metrics.keys()) != set(config.seeds):
            raise ValueError(f"Checkpoint seeds for {label} do not match config.seeds.")

        resume_paths = _write_base_ckpt_data_for_beta_to_disk(label, base_seed_metrics, resume_root)
        extra_seed_metrics = _train_single_beta(extra_cfg, beta, gpu_ids, resume_paths=resume_paths, epoch_offset=config.epochs)
        new_results[label] = extra_seed_metrics

    # merge base + extra
    merged_results: ResultsByBeta = {}
    for beta in config.betas:
        label = label_from_beta(beta, config.n)
        base_seed_metrics  = base_results[label]
        extra_seed_metrics = new_results[label]

        merged_seed_metrics: Dict[int, Metrics] = {}
        for seed in config.seeds:
            merged_seed_metrics[seed] = _merge_metrics(base_seed_metrics[seed], extra_seed_metrics[seed])
        merged_results[label] = merged_seed_metrics

    new_config = replace(config, epochs=new_epochs)
    return merged_results, new_config


def run_exp1(config: Exp1Config, run_opts: Exp1RunOpts, gpu_ids: List[int],) -> Tuple[ResultsByBeta, Exp1Config]:
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
        results = {
            label_from_beta(beta, config.n): _train_single_beta(config=config, beta=beta, gpu_ids=gpu_ids)
            for beta in config.betas
        }

    if run_opts.save_ckpt:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = run_opts.ckpt_dir / f"{EXP1_CHECKPOINT_PREFIX}{timestamp}.pt"
        save_exp1_checkpoint(str(ckpt_path), results, exp_config)
        print(f"Saved checkpoint: {ckpt_path}")

    return results, exp_config


# required for plotting the ICML-deadline ckpt
def infer_effective_track_every(results: ResultsByBeta, config: Exp1Config) -> int:
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
