from __future__ import annotations

import math
import random

import numpy as np
import torch

from src.data import load_digits_data, load_mnist_data

from .checkpoint import ShardedResumeState, apply_resume_state_to_model
from .config import ShardedExpConfig, ShardedMetricPlan, resolve_eta
from .distributed import DistContext, rank0_print
from .metrics import loss_fn, record_metrics
from .model import ShardedMLP


def load_data_for_seed(config: ShardedExpConfig, seed: int, device: torch.device) -> dict:
    if config.dataset == "digits":
        return load_digits_data(
            n=config.n,
            random_labels=config.random_labels,
            device=str(device),
            seed=seed,
        )
    if config.dataset == "mnist":
        return load_mnist_data(
            n=config.n,
            random_labels=config.random_labels,
            device=str(device),
            seed=seed,
            reserve_last=config.reserve_last,
        )
    raise ValueError(f"Unsupported dataset={config.dataset!r}")


def estimate_local_parameter_count(config: ShardedExpConfig, d_in: int, d_out: int, world_size: int) -> int:
    local_m = config.m // world_size
    hidden = local_m * d_in
    hidden += max(0, config.L - 1) * local_m * config.m
    output = d_out * local_m
    return hidden + output


def validate_memory_budget(config: ShardedExpConfig, d_in: int, d_out: int, ctx: DistContext) -> None:
    if ctx.device.type != "cuda":
        return
    local_params = estimate_local_parameter_count(config, d_in, d_out, ctx.world_size)
    bytes_per_param = torch.finfo(torch.float32).bits // 8
    # Base params, init params, linearized params, two gradient sets, plus slack.
    estimated_peak = local_params * bytes_per_param * 6
    total_memory = torch.cuda.get_device_properties(ctx.device).total_memory
    if estimated_peak > 0.9 * total_memory:
        raise RuntimeError(
            "Estimated sharded model state is too large for this GPU. "
            f"local_params={local_params:,}, estimated_peak={estimated_peak / 2**30:.1f} GiB, "
            f"gpu_memory={total_memory / 2**30:.1f} GiB. "
            "Reduce m/L or increase the number of ranks."
        )


def init_metric_store(metric_plan: ShardedMetricPlan) -> dict:
    metrics = {f"{name}_hist": [] for name in metric_plan.tracked_metrics}
    metrics["epoch_hist"] = []
    metrics["tracked_metrics"] = list(metric_plan.tracked_metrics)
    metrics["stopped_early"] = False
    return metrics


def set_reproducible_seed(seed: int, ctx: DistContext) -> None:
    rank_seed = int(seed) + ctx.rank * 1_000_003
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    np.random.seed(rank_seed % (2**32 - 1))
    random.seed(rank_seed)


def _zero_grads(params: list[torch.Tensor]) -> None:
    for param in params:
        param.grad = None


def _apply_one_param_update(
    param: torch.Tensor,
    lam: float,
    beta: float,
    eta: float,
    regularization_scale: float,
    noise_scale: float,
    generator: torch.Generator,
    chunk_elements: int,
) -> None:
    if param.grad is not None:
        param.add_(param.grad, alpha=-eta)

    if not math.isinf(float(beta)):
        param.add_(param, alpha=-eta * (regularization_scale / beta) * lam)

    if noise_scale == 0.0:
        return

    flat = param.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        end = min(start + chunk_elements, flat.numel())
        noise = torch.randn(
            (end - start,),
            device=param.device,
            dtype=param.dtype,
            generator=generator,
        )
        flat[start:end].add_(noise, alpha=noise_scale)


@torch.no_grad()
def apply_langevin_step(model: ShardedMLP, beta: float, eta: float, regularization_scale: float, same_noise: bool) -> None:
    noise_scale = 0.0 if math.isinf(float(beta)) else math.sqrt(2.0 * eta / beta)
    lambdas = model.lambdas()

    if same_noise:
        for base, lin, lam in zip(model.base_parameters_list(), model.lin_parameters_list(), lambdas):
            if base.grad is not None:
                base.add_(base.grad, alpha=-eta)
            if lin.grad is not None:
                lin.add_(lin.grad, alpha=-eta)
            if not math.isinf(float(beta)):
                coeff = -eta * (regularization_scale / beta) * lam
                base.add_(base, alpha=coeff)
                lin.add_(lin, alpha=coeff)
            if noise_scale == 0.0:
                continue
            base_flat = base.view(-1)
            lin_flat = lin.view(-1)
            for start in range(0, base_flat.numel(), model.noise_chunk_elements):
                end = min(start + model.noise_chunk_elements, base_flat.numel())
                noise = torch.randn(
                    (end - start,),
                    device=base.device,
                    dtype=base.dtype,
                    generator=model.noise_gen,
                )
                base_flat[start:end].add_(noise, alpha=noise_scale)
                lin_flat[start:end].add_(noise, alpha=noise_scale)
        return

    for param, lam in zip(model.base_parameters_list(), lambdas):
        _apply_one_param_update(
            param,
            lam,
            beta,
            eta,
            regularization_scale,
            noise_scale,
            model.noise_gen,
            model.noise_chunk_elements,
        )
    for param, lam in zip(model.lin_parameters_list(), lambdas):
        _apply_one_param_update(
            param,
            lam,
            beta,
            eta,
            regularization_scale,
            noise_scale,
            model.noise_gen,
            model.noise_chunk_elements,
        )


def _should_record(epoch: int, track_every: int) -> bool:
    return track_every == 1 or epoch % track_every == 1


def _should_print(epoch: int, print_every: int) -> bool:
    return print_every == 1 or epoch % print_every == 1


def _should_stop_early(config: ShardedExpConfig, stats: dict, lin_stats: dict) -> tuple[bool, float | None]:
    if config.early_stop_metric is None or config.early_stop_value is None:
        return False, None
    values = {**stats, **lin_stats}
    if config.early_stop_metric not in values:
        return False, None
    current = float(values[config.early_stop_metric])
    if config.early_stop_goal == "min":
        return current <= config.early_stop_value, current
    return current >= config.early_stop_value, current


def train_one(
    config: ShardedExpConfig,
    metric_plan: ShardedMetricPlan,
    alpha: float,
    beta: float,
    seed: int,
    ctx: DistContext,
    resume_state: ShardedResumeState | None = None,
) -> tuple[dict, ShardedMLP]:
    set_reproducible_seed(seed, ctx)
    data = load_data_for_seed(config, seed, ctx.device)
    validate_memory_budget(config, data["d_in"], data["d_out"], ctx)
    model = ShardedMLP(
        d_in=data["d_in"],
        d_out=data["d_out"],
        config=config,
        ctx=ctx,
        alpha=alpha,
        seed=seed,
    ).to(ctx.device)

    epoch_offset = 0
    if resume_state is not None:
        apply_resume_state_to_model(model, resume_state)
        if resume_state.last_epoch is not None:
            epoch_offset = int(resume_state.last_epoch)

    param_norm0_value = model.param_norm0()
    metrics = init_metric_store(metric_plan)
    eta = resolve_eta(config, alpha=alpha, beta=beta)
    end_epoch = epoch_offset if (resume_state is not None and resume_state.stopped_early) else config.epochs

    rank0_print(
        ctx,
        f"training alpha={alpha}, beta={beta}, seed={seed}, "
        f"L={config.L}, m={config.m}, eta={eta}, from epoch={epoch_offset + 1}",
        flush=True,
    )

    last_epoch = epoch_offset
    for epoch in range(epoch_offset + 1, end_epoch + 1):
        last_epoch = epoch
        if _should_record(epoch, config.track_every):
            model.clear_grads("all")
            stats, lin_stats = record_metrics(
                metrics,
                model,
                data,
                metric_plan,
                epoch,
                param_norm0_value,
                jacobian_batch_size=config.jac_probe_size,
            )

            if _should_print(epoch, config.print_every):
                rank0_print(
                    ctx,
                    f"epoch {epoch:8d} | loss {stats['train_loss']:.4f} | "
                    f"train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}",
                    flush=True,
                )

            should_stop, current = _should_stop_early(config, stats, lin_stats)
            if should_stop:
                metrics["stopped_early"] = True
                rank0_print(
                    ctx,
                    f"early stopping at epoch={epoch}, {config.early_stop_metric}={current:.4g}",
                    flush=True,
                )
                break

        model.clear_grads("all")
        outputs = model(data["X_train"])
        train_loss = loss_fn(outputs, data["y_train_one_hot"])
        (train_loss / float(ctx.world_size)).backward()

        if config.use_linearized:
            lin_outputs = model.linearized_forward(data["X_train"])
            lin_loss = loss_fn(lin_outputs, data["y_train_one_hot"])
            (lin_loss / float(ctx.world_size)).backward()

        step_beta = beta
        if config.noise_free_after_epoch is not None and epoch > config.noise_free_after_epoch:
            step_beta = math.inf
        apply_langevin_step(
            model,
            beta=step_beta,
            eta=eta,
            regularization_scale=config.regularization_scale,
            same_noise=config.same_noise and config.use_linearized,
        )

    metrics["last_epoch"] = last_epoch
    if resume_state is not None and resume_state.stopped_early:
        metrics["stopped_early"] = True
    model.clear_grads("all")
    return metrics, model
