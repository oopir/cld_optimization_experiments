from __future__ import annotations

from dataclasses import replace
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import (
    apply_config_overrides,
    save_model_state,
    infer_last_epoch_from_results,
    load_checkpoint_with_metadata,
    load_resume_state,
    merge_results,
    resolve_checkpoint_path,
    save_checkpoint,
    timestamped_checkpoint_path,
    validate_resume_request,
)
from .config import ExpConfig, RunOpts, resolve_metric_plan, validate_config
from .training import train_one


def label_from_alpha_beta(alpha=None, beta=None, n=None) -> str:
    label = ""
    if alpha is not None:
        label += f"α={alpha:.0e} "
    if beta == np.inf or math.isinf(float(beta)):
        label += "inf"
    elif n is None:
        label += f"β={int(beta)}"
    else:
        label += f"β={int(beta // n)}n"
    return label


def iter_alpha_beta_pairs(config: ExpConfig) -> list[tuple[float | None, float]]:
    betas = list(config.betas or [])
    alphas = list(config.alphas or [])
    if not betas:
        betas = [math.inf]
    if not alphas:
        return [(None, beta) for beta in betas]
    return list(itertools.product(alphas, betas))


def _labels_for_config(config: ExpConfig) -> list[str]:
    return [
        label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
        for alpha, beta in iter_alpha_beta_pairs(config)
    ]


def _print_config(config: ExpConfig) -> None:
    print("configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print(
        f"  effective_tracked_metrics: {list(resolve_metric_plan(config).tracked_metrics)}"
    )


def _train_over_config(
    config: ExpConfig,
    metric_plan,
    device: torch.device,
    loaded=None,
    ckpt_path: Path | None = None,
    capture_states: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    results = {}
    states = {} if capture_states else None
    for alpha_opt, beta in iter_alpha_beta_pairs(config):
        alpha = 1.0 if alpha_opt is None else float(alpha_opt)
        label = label_from_alpha_beta(alpha=alpha_opt, beta=beta, n=config.n)
        results[label] = {}
        if states is not None:
            states[label] = {}
        for seed in config.seeds:
            seed_int = int(seed)
            resume_state = (load_resume_state(loaded, label, seed_int) if loaded is not None else None)
            metrics, model = train_one(
                config=config,
                metric_plan=metric_plan,
                alpha=alpha,
                beta=float(beta),
                seed=seed_int,
                device=device,
                resume_state=resume_state,
            )
            results[label][seed_int] = metrics
            if states is not None:
                if ckpt_path is None:
                    raise ValueError("ckpt_path must be set when capture_states=True.")
                states[label][seed_int] = save_model_state(
                    ckpt_path,
                    label,
                    seed_int,
                    model,
                    metrics,
                    device,
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results, states


def _save(
    ckpt_path: Path | None,
    results: dict,
    config: ExpConfig,
    states: dict | None,
) -> None:
    if ckpt_path is None:
        return
    checkpoint_states = states if config.checkpoint_state == "resumable_state" else None
    save_checkpoint(
        ckpt_path,
        results,
        checkpoint_states,
        config,
        list(resolve_metric_plan(config).tracked_metrics),
    )
    print(f"Saved checkpoint: {ckpt_path}", flush=True)


def run_exp(config: ExpConfig, run_opts: RunOpts, device: torch.device) -> tuple[dict[str, Any], Path | None]:
    if run_opts.resume_from_ckpt and not run_opts.load_ckpt:
        raise ValueError("load_ckpt must be true when resume_from_ckpt is true.")
    if run_opts.load_ckpt:
        load_path = resolve_checkpoint_path(run_opts.ckpt_dir, run_opts.load_ckpt_name)
        loaded = load_checkpoint_with_metadata(load_path)
        print(f"Loaded checkpoint: {loaded.path}")
        if not run_opts.resume_from_ckpt:
            return loaded.results, loaded.path
        if run_opts.new_total_epochs is None:
            raise ValueError("new_total_epochs must be set when resume_from_ckpt is true.")
        resume_config = apply_config_overrides(
            loaded.config,
            config,
            run_opts.config_overrides,
        )
        validate_config(resume_config)
        expected_labels = _labels_for_config(resume_config)
        validate_resume_request(
            resume_config,
            loaded,
            expected_labels,
            int(run_opts.new_total_epochs),
        )
        train_config = replace(resume_config, epochs=int(run_opts.new_total_epochs))
        metric_plan = resolve_metric_plan(train_config)
        _print_config(
            train_config,
        )
        ckpt_path = (
            timestamped_checkpoint_path(
                run_opts.ckpt_dir,
                config.seeds[0],
                train_config.checkpoint_state,
            )
            if run_opts.save_ckpt
            else None
        )
        extra_results, extra_states = _train_over_config(
            train_config,
            metric_plan,
            device,
            loaded=loaded,
            ckpt_path=ckpt_path,
            capture_states=(
                run_opts.save_ckpt and train_config.checkpoint_state == "resumable_state"
            ),
        )
        merged = merge_results(loaded.results, extra_results, train_config.seeds, expected_labels)
        states = extra_states
        final_epochs = infer_last_epoch_from_results(merged, int(run_opts.new_total_epochs))
        final_config = replace(train_config, epochs=final_epochs)
        _save(ckpt_path, merged, final_config, states)
        return merged, ckpt_path
    validate_config(config)
    _print_config(config)
    ckpt_path = (
        timestamped_checkpoint_path(
            run_opts.ckpt_dir,
            config.seeds[0],
            config.checkpoint_state,
        )
        if run_opts.save_ckpt
        else None
    )
    results, states = _train_over_config(
        config,
        resolve_metric_plan(config),
        device,
        ckpt_path=ckpt_path,
        capture_states=(
            run_opts.save_ckpt and config.checkpoint_state == "resumable_state"
        ),
    )
    final_epochs = infer_last_epoch_from_results(results, config.epochs)
    final_config = replace(config, epochs=final_epochs)
    _save(ckpt_path, results, final_config, states)
    return results, ckpt_path
