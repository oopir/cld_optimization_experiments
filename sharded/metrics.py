from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import BASE_HISTORY_METRICS, LIN_HISTORY_METRICS, ShardedMetricPlan
from .distributed import all_reduce_sum
from .model import ShardedMLP
from .ops import all_gather_cat


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(outputs, targets)


@torch.no_grad()
def classification_metrics(
    model: ShardedMLP,
    data: dict,
    metric_plan: ShardedMetricPlan,
    linearized: bool = False,
) -> dict:
    metrics = metric_plan.compute_metrics
    prefix = "lin_" if linearized else ""
    stats = {}

    need_train = f"{prefix}train_loss" in metrics or f"{prefix}train_acc" in metrics
    need_test = f"{prefix}test_acc" in metrics

    if need_train:
        outputs = model.linearized_forward(data["X_train"]) if linearized else model(data["X_train"])
        pred = outputs.argmax(dim=1)
        if f"{prefix}train_acc" in metrics:
            stats[f"{prefix}train_acc"] = (pred == data["y_train"]).float().mean().item()
        if f"{prefix}train_loss" in metrics:
            stats[f"{prefix}train_loss"] = loss_fn(outputs, data["y_train_one_hot"]).item()

    if need_test:
        outputs = model.linearized_forward(data["X_test"]) if linearized else model(data["X_test"])
        pred = outputs.argmax(dim=1)
        stats[f"{prefix}test_acc"] = (pred == data["y_test"]).float().mean().item()

    return stats


@torch.no_grad()
def feature_gram_lambda(model: ShardedMLP, X_train: torch.Tensor) -> float:
    first_local = model.first_hidden_local(X_train)
    gram_local = first_local.matmul(first_local.t()).to(dtype=torch.float64)
    gram = all_reduce_sum(gram_local)
    gram = 0.5 * (gram + gram.t())
    try:
        return float(torch.linalg.eigvalsh(gram)[0].item())
    except Exception:
        return float("nan")


@torch.no_grad()
def param_dist(model: ShardedMLP, params_a: list[torch.Tensor], params_b: list[torch.Tensor], normalize_by=None):
    local_sq = torch.zeros((), device=model.ctx.device, dtype=torch.float64)
    for a, b in zip(params_a, params_b):
        local_sq += (a.detach() - b.detach()).pow(2).sum(dtype=torch.float64)
    total_sq = float(all_reduce_sum(local_sq).item())
    value = math.sqrt(total_sq)
    if normalize_by is not None:
        value /= float(normalize_by) + 1e-12
    return value


@torch.no_grad()
def nn_lin_param_dist(model: ShardedMLP, normalize_by: float) -> tuple[float, float]:
    local = torch.zeros(4, device=model.ctx.device, dtype=torch.float64)
    for p_nn, p_lin in zip(model.base_parameters_list(), model.lin_parameters_list()):
        nn = p_nn.detach()
        lin = p_lin.detach()
        diff = nn - lin
        local[0] += diff.pow(2).sum(dtype=torch.float64)
        local[1] += (nn * lin).sum(dtype=torch.float64)
        local[2] += nn.pow(2).sum(dtype=torch.float64)
        local[3] += lin.pow(2).sum(dtype=torch.float64)

    total_sq, dot, norm_nn_sq, norm_lin_sq = all_reduce_sum(local).tolist()
    l2 = math.sqrt(total_sq) / (float(normalize_by) + 1e-12)
    denom = math.sqrt(norm_nn_sq) * math.sqrt(norm_lin_sq) + 1e-12
    cos_sim = max(-1.0, min(1.0, dot / denom))
    return l2, 1.0 - cos_sim


@torch.no_grad()
def _hidden_forward_locals(
    hidden_weights: list[torch.Tensor],
    X: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    z_locals = []
    a_locals = []
    a_prev_full = X

    for idx, weight in enumerate(hidden_weights):
        z_local = a_prev_full.matmul(weight.t())
        a_local = torch.tanh(z_local)
        z_locals.append(z_local)
        a_locals.append(a_local)
        if idx < len(hidden_weights) - 1:
            a_prev_full = all_gather_cat(a_local, dim=1)

    return z_locals, a_locals


@torch.no_grad()
def _backprop_output_sensitivities(
    hidden_weights: list[torch.Tensor],
    output_weight: torch.Tensor,
    z_locals: list[torch.Tensor],
    model: ShardedMLP,
) -> list[torch.Tensor]:
    """Return local d f_class / d z_l for every hidden layer.

    Each tensor has shape (batch, local_m, d_out). The output_scale is applied
    once when the final scalar reductions are accumulated.
    """
    deltas: list[torch.Tensor | None] = [None] * len(hidden_weights)
    last_a_prime = 1.0 - torch.tanh(z_locals[-1]).pow(2)
    deltas[-1] = last_a_prime.unsqueeze(-1) * output_weight.t().unsqueeze(0)

    for idx in range(len(hidden_weights) - 2, -1, -1):
        next_delta = deltas[idx + 1]
        assert next_delta is not None
        msg_local = torch.einsum("buc,uk->bkc", next_delta, hidden_weights[idx + 1])
        msg_full = all_reduce_sum(msg_local)
        msg_shard = msg_full[:, model.local_start : model.local_start + model.local_m, :]
        a_prime = 1.0 - torch.tanh(z_locals[idx]).pow(2)
        deltas[idx] = a_prime.unsqueeze(-1) * msg_shard

    return [delta for delta in deltas if delta is not None]


@torch.no_grad()
def _prev_activation_stats(
    X_batch: torch.Tensor,
    curr_activations: list[torch.Tensor],
    init_activations: list[torch.Tensor],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layer_idx == 0:
        norm = X_batch.pow(2).sum(dim=1, dtype=torch.float64)
        return norm, norm, norm

    curr_prev = curr_activations[layer_idx - 1]
    init_prev = init_activations[layer_idx - 1]
    local = torch.stack(
        (
            curr_prev.pow(2).sum(dim=1, dtype=torch.float64),
            init_prev.pow(2).sum(dim=1, dtype=torch.float64),
            (curr_prev * init_prev).sum(dim=1, dtype=torch.float64),
        )
    )
    total = all_reduce_sum(local)
    return total[0], total[1], total[2]


@torch.no_grad()
def streamed_jacobian_drift(
    model: ShardedMLP,
    X_train: torch.Tensor,
    batch_size: int = 1,
) -> tuple[float, float]:
    """Compute exact full-Jacobian drift from streamed scalar reductions."""
    totals = torch.zeros(4, device=model.ctx.device, dtype=torch.float64)

    hidden_curr = list(model.hidden)
    hidden_init = list(model.init_hidden)
    scale_sq = float(model.output_scale) ** 2

    for start in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[start : start + batch_size]

        z_curr, a_curr = _hidden_forward_locals(hidden_curr, X_batch)
        z_init, a_init = _hidden_forward_locals(hidden_init, X_batch)
        delta_curr = _backprop_output_sensitivities(hidden_curr, model.output, z_curr, model)
        delta_init = _backprop_output_sensitivities(hidden_init, model.init_output, z_init, model)

        for layer_idx, (dc, di) in enumerate(zip(delta_curr, delta_init)):
            prev_curr_sq, prev_init_sq, prev_dot = _prev_activation_stats(
                X_batch,
                a_curr,
                a_init,
                layer_idx,
            )

            dc_sq = dc.pow(2).sum(dim=1, dtype=torch.float64)
            di_sq = di.pow(2).sum(dim=1, dtype=torch.float64)
            dc_di = (dc * di).sum(dim=1, dtype=torch.float64)

            norm_curr = (dc_sq * prev_curr_sq[:, None]).sum()
            norm_init = (di_sq * prev_init_sq[:, None]).sum()
            dot = (dc_di * prev_dot[:, None]).sum()
            totals[1] += scale_sq * dot
            totals[2] += scale_sq * norm_curr
            totals[3] += scale_sq * norm_init
            totals[0] += scale_sq * (norm_curr + norm_init - 2.0 * dot)

        out_curr_sq = a_curr[-1].pow(2).sum(dim=1, dtype=torch.float64)
        out_init_sq = a_init[-1].pow(2).sum(dim=1, dtype=torch.float64)
        out_dot = (a_curr[-1] * a_init[-1]).sum(dim=1, dtype=torch.float64)
        output_classes = model.d_out
        out_norm_curr = output_classes * out_curr_sq.sum()
        out_norm_init = output_classes * out_init_sq.sum()
        out_dot_total = output_classes * out_dot.sum()
        totals[1] += scale_sq * out_dot_total
        totals[2] += scale_sq * out_norm_curr
        totals[3] += scale_sq * out_norm_init
        totals[0] += scale_sq * (out_norm_curr + out_norm_init - 2.0 * out_dot_total)

    total_sq, dot, norm_curr_sq, norm_init_sq = all_reduce_sum(totals).tolist()
    total_sq = max(0.0, total_sq)
    l2 = math.sqrt(total_sq) / (math.sqrt(norm_init_sq) + 1e-12)
    denom = math.sqrt(norm_curr_sq) * math.sqrt(norm_init_sq) + 1e-12
    cos_sim = max(-1.0, min(1.0, dot / denom))
    return l2, 1.0 - cos_sim


def record_metrics(
    metrics: dict,
    model: ShardedMLP,
    data: dict,
    metric_plan: ShardedMetricPlan,
    epoch: int,
    param_norm0_value: float,
    jacobian_batch_size: int = 1,
) -> tuple[dict, dict]:
    metrics["epoch_hist"].append(epoch)
    stats = classification_metrics(model, data, metric_plan, linearized=False)

    if "param_dist" in metric_plan.compute_metrics:
        stats["param_dist"] = param_dist(model, model.base_parameters_list(), model.init_tensors_list())
    if "feat_gram_lambda" in metric_plan.compute_metrics:
        stats["feat_gram_lambda"] = feature_gram_lambda(model, data["X_train"])

    lin_stats = {}
    if metric_plan.needs_linearized_metrics:
        lin_stats = classification_metrics(model, data, metric_plan, linearized=True)
        if "lin_param_dist" in metric_plan.compute_metrics:
            lin_stats["lin_param_dist"] = param_dist(
                model,
                model.lin_parameters_list(),
                model.init_tensors_list(),
            )
        if "nn_lin_param_dist" in metric_plan.compute_metrics:
            lin_stats["nn_lin_param_dist"] = nn_lin_param_dist(model, normalize_by=param_norm0_value)

    if "jacobian_dist" in metric_plan.compute_metrics:
        stats["jacobian_dist"] = streamed_jacobian_drift(
            model,
            data["X_train"],
            batch_size=jacobian_batch_size,
        )

    combined = {**stats, **lin_stats}
    for name in BASE_HISTORY_METRICS + LIN_HISTORY_METRICS + ("nn_lin_param_dist", "jacobian_dist"):
        if name in metric_plan.history_metrics:
            metrics[f"{name}_hist"].append(combined[name])

    return stats, lin_stats
