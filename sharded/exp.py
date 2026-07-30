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
    infer_last_epoch_from_results,
    load_checkpoint_with_metadata,
    load_state_shard,
    merge_results,
    resolve_checkpoint_path,
    save_metrics_checkpoint,
    save_state_shard,
    timestamped_checkpoint_path,
    validate_resume_request,
)
from .config import ShardedExpConfig, ShardedRunOpts, resolve_metric_plan, validate_config
from .distributed import DistContext, barrier, rank0_print
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


def iter_alpha_beta_pairs(config: ShardedExpConfig) -> list[tuple[float | None, float]]:
    betas = list(config.betas or [])
    alphas = list(config.alphas or [])
    if not betas:
        betas = [math.inf]
    if not alphas:
        return [(None, beta) for beta in betas]
    return list(itertools.product(alphas, betas))


def _labels_for_config(config: ShardedExpConfig) -> list[str]:
    return [
        label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
        for alpha, beta in iter_alpha_beta_pairs(config)
    ]


def _print_config(
    config: ShardedExpConfig,
    ctx: DistContext,
    prev_config: ShardedExpConfig | None = None,
    override_keys: list[str] | None = None,
) -> None:
    if not ctx.is_rank0:
        return
    rank0_print(ctx, "configuration:")
    prev = prev_config.__dict__ if prev_config is not None else {}
    override_key_set = set(override_keys or [])
    for key, value in config.__dict__.items():
        if key in override_key_set and key in prev and prev[key] != value:
            rank0_print(ctx, f"  {key}: {value} (previously {prev[key]})")
        else:
            rank0_print(ctx, f"  {key}: {value}")
    rank0_print(ctx, f"  effective_tracked_metrics: {list(resolve_metric_plan(config).tracked_metrics)}")


def _maybe_empty_cache(ctx: DistContext) -> None:
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def _create_output_checkpoint(
    config: ShardedExpConfig,
    run_opts: ShardedRunOpts,
    ctx: DistContext,
) -> Path | None:
    if not run_opts.save_ckpt:
        return None

    run_opts.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = timestamped_checkpoint_path(run_opts.ckpt_dir, config.dataset, config.checkpoint_state)
    if config.checkpoint_state == "sharded_state" and ctx.is_rank0:
        ckpt_path.mkdir(parents=True, exist_ok=True)
    barrier()
    return ckpt_path


def _train_over_config(
    config: ShardedExpConfig,
    metric_plan,
    ctx: DistContext,
    ckpt_path: Path | None = None,
    resume_root: Path | None = None,
    base_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for alpha_opt, beta in iter_alpha_beta_pairs(config):
        alpha = 1.0 if alpha_opt is None else float(alpha_opt)
        label = label_from_alpha_beta(alpha=alpha_opt, beta=beta, n=config.n)
        if ctx.is_rank0:
            results[label] = {}

        for seed in config.seeds:
            resume_state = None
            seed_int = int(seed)
            if base_results is not None:
                if resume_root is None:
                    raise ValueError("resume_root must be set when base_results is provided.")
                resume_state = load_state_shard(resume_root, base_results[label][seed_int], ctx)

            metrics, model = train_one(
                config=config,
                metric_plan=metric_plan,
                alpha=alpha,
                beta=float(beta),
                seed=seed_int,
                ctx=ctx,
                resume_state=resume_state,
            )

            if ckpt_path is not None and config.checkpoint_state == "sharded_state":
                state_rel = save_state_shard(ckpt_path, label, seed_int, model, ctx, metrics)
                metrics["state_shard_dir"] = state_rel

            if ctx.is_rank0:
                results[label][seed_int] = metrics

            del model
            _maybe_empty_cache(ctx)
            barrier()

    return results


def _save_final_checkpoint(
    ckpt_path: Path | None,
    results: dict[str, Any],
    config: ShardedExpConfig,
    ctx: DistContext,
) -> None:
    if ckpt_path is None:
        return
    metric_plan = resolve_metric_plan(config)
    save_metrics_checkpoint(
        ckpt_path,
        results,
        config,
        list(metric_plan.tracked_metrics),
        ctx,
    )
    rank0_print(ctx, f"Saved checkpoint: {ckpt_path}", flush=True)


def run_exp(config: ShardedExpConfig, run_opts: ShardedRunOpts, ctx: DistContext) -> tuple[dict[str, Any], Path | None]:
    if run_opts.resume_from_ckpt and not run_opts.load_ckpt:
        raise ValueError("load_ckpt must be true when resume_from_ckpt is true.")

    if run_opts.load_ckpt:
        load_path = resolve_checkpoint_path(run_opts.ckpt_dir, run_opts.load_ckpt_name)
        loaded = load_checkpoint_with_metadata(load_path)
        rank0_print(ctx, f"Loaded checkpoint: {loaded.payload_path}", flush=True)

        if not run_opts.resume_from_ckpt:
            _print_config(loaded.config, ctx)
            return loaded.results if ctx.is_rank0 else {}, loaded.path

        if run_opts.new_total_epochs is None:
            raise ValueError("new_total_epochs must be set when resume_from_ckpt is true.")

        resume_config = apply_config_overrides(
            loaded.config,
            config,
            run_opts.config_overrides,
        )
        validate_config(resume_config, ctx.world_size)
        expected_labels = _labels_for_config(resume_config)
        validate_resume_request(
            resume_config,
            loaded.metadata,
            loaded.results,
            expected_labels,
            ctx.world_size,
            int(run_opts.new_total_epochs),
        )
        train_config = replace(resume_config, epochs=int(run_opts.new_total_epochs))
        metric_plan = resolve_metric_plan(train_config)
        _print_config(
            train_config,
            ctx,
            prev_config=loaded.config,
            override_keys=run_opts.config_overrides or [],
        )
        rank0_print(ctx, f"extending to a new total of {run_opts.new_total_epochs} epochs...", flush=True)

        ckpt_path = _create_output_checkpoint(train_config, run_opts, ctx)
        extra_results = _train_over_config(
            train_config,
            metric_plan,
            ctx,
            ckpt_path=ckpt_path,
            resume_root=loaded.path,
            base_results=loaded.results,
        )

        if ctx.is_rank0:
            merged = merge_results(loaded.results, extra_results, train_config.seeds, expected_labels)
            final_epochs = infer_last_epoch_from_results(merged, int(run_opts.new_total_epochs))
            final_config = replace(train_config, epochs=final_epochs)
        else:
            merged = {}
            final_config = train_config

        _save_final_checkpoint(ckpt_path, merged, final_config, ctx)
        return merged, ckpt_path

    validate_config(config, ctx.world_size)
    metric_plan = resolve_metric_plan(config)
    _print_config(config, ctx)

    ckpt_path = _create_output_checkpoint(config, run_opts, ctx)
    results = _train_over_config(config, metric_plan, ctx, ckpt_path=ckpt_path)
    if ctx.is_rank0:
        final_epochs = infer_last_epoch_from_results(results, config.epochs)
        final_config = replace(config, epochs=final_epochs)
    else:
        final_config = config

    _save_final_checkpoint(ckpt_path, results, final_config, ctx)
    return results, ckpt_path
