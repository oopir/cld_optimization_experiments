from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import save_metrics_checkpoint, save_state_shard, timestamped_checkpoint_path
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


def _print_config(config: ShardedExpConfig, ctx: DistContext) -> None:
    if not ctx.is_rank0:
        return
    rank0_print(ctx, "configuration:")
    for key, value in config.__dict__.items():
        rank0_print(ctx, f"  {key}: {value}")
    rank0_print(ctx, f"  effective_tracked_metrics: {list(resolve_metric_plan(config).tracked_metrics)}")


def _maybe_empty_cache(ctx: DistContext) -> None:
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def run_exp(config: ShardedExpConfig, run_opts: ShardedRunOpts, ctx: DistContext) -> tuple[dict[str, Any], Path | None]:
    validate_config(config, ctx.world_size)
    metric_plan = resolve_metric_plan(config)
    _print_config(config, ctx)

    ckpt_path = None
    if run_opts.save_ckpt:
        run_opts.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = timestamped_checkpoint_path(run_opts.ckpt_dir, config.dataset, config.checkpoint_state)
        if config.checkpoint_state == "sharded_state" and ctx.is_rank0:
            ckpt_path.mkdir(parents=True, exist_ok=True)
        barrier()

    results: dict[str, Any] = {}
    for alpha_opt, beta in iter_alpha_beta_pairs(config):
        alpha = 1.0 if alpha_opt is None else float(alpha_opt)
        label = label_from_alpha_beta(alpha=alpha_opt, beta=beta, n=config.n)
        if ctx.is_rank0:
            results[label] = {}

        for seed in config.seeds:
            metrics, model = train_one(
                config=config,
                metric_plan=metric_plan,
                alpha=alpha,
                beta=float(beta),
                seed=int(seed),
                ctx=ctx,
            )

            if ckpt_path is not None and config.checkpoint_state == "sharded_state":
                state_rel = save_state_shard(ckpt_path, label, int(seed), model, ctx)
                if ctx.is_rank0:
                    metrics["state_shard_dir"] = state_rel

            if ctx.is_rank0:
                results[label][int(seed)] = metrics

            del model
            _maybe_empty_cache(ctx)
            barrier()

    if ckpt_path is not None:
        save_metrics_checkpoint(
            ckpt_path,
            results,
            config,
            list(metric_plan.tracked_metrics),
            ctx,
        )
        rank0_print(ctx, f"Saved checkpoint: {ckpt_path}", flush=True)

    return results, ckpt_path

