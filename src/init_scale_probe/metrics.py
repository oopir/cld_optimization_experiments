import math
from typing import Dict, Sequence

import torch


BINARY_CORE_METRICS = (
    "mean_abs_output",
    "mean_preactivation_norm",
    "mean_preactivation_norm_normalized",
    "mean_abs_residual",
    "empirical_loss",
    "mean_output_grad_norm",
    "empirical_loss_grad_norm",
)

BINARY_FORWARD_METRICS = (
    "mean_abs_output",
    "mean_preactivation_norm",
    "mean_preactivation_norm_normalized",
    "mean_abs_residual",
    "empirical_loss",
)

BINARY_OUTPUT_GRAD_METRICS = (
    "mean_output_grad_norm",
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_fc2_normalized",
)

BINARY_EMPIRICAL_LOSS_GRAD_METRICS = (
    "empirical_loss_grad_norm",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
    "empirical_loss_grad_norm_fc2_normalized",
)

BINARY_LAYERWISE_METRICS = (
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_fc2_normalized",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
    "empirical_loss_grad_norm_fc2_normalized",
)

BINARY_PARAMETER_METRICS = (
    "fc1_weight_fro_norm",
    "fc1_weight_fro_norm_normalized",
    "fc1_weight_spectral_norm",
    "fc1_weight_spectral_norm_normalized",
)

BINARY_ALL_METRICS = BINARY_CORE_METRICS + BINARY_LAYERWISE_METRICS + BINARY_PARAMETER_METRICS
BINARY_GRADIENT_METRICS = BINARY_OUTPUT_GRAD_METRICS + BINARY_EMPIRICAL_LOSS_GRAD_METRICS


@torch.no_grad()
def _compute_binary_forward_metrics(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """Compute scalar-output binary probe forward metrics."""
    metric_names = set(metric_names)
    n = X.shape[0]
    totals = {name: 0.0 for name in metric_names}
    sqrt_m = math.sqrt(float(model.m))
    preactivation_metric_names = {"mean_preactivation_norm", "mean_preactivation_norm_normalized"}
    output_metric_names = {"mean_abs_output", "mean_abs_residual", "empirical_loss"}
    needs_preactivation_norm = bool(metric_names & preactivation_metric_names)
    needs_output = bool(metric_names & output_metric_names)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        yb = y[start:end]

        z = model.fc1(xb)
        if needs_preactivation_norm:
            preactivation_norm_sum = float(torch.linalg.vector_norm(z, dim=1).sum().item())
            if "mean_preactivation_norm" in metric_names:
                totals["mean_preactivation_norm"] += preactivation_norm_sum
            if "mean_preactivation_norm_normalized" in metric_names:
                totals["mean_preactivation_norm_normalized"] += preactivation_norm_sum / sqrt_m
        if not needs_output:
            continue

        out = model.fc2(torch.tanh(z))
        if model.init_type == "alpha" and model.alpha != 0:
            out = out / model.alpha
        out = out.view(-1, 1)

        if "mean_abs_output" in metric_names:
            totals["mean_abs_output"] += float(out.abs().sum().item())
        residual = out - yb
        if "mean_abs_residual" in metric_names:
            totals["mean_abs_residual"] += float(residual.abs().sum().item())
        if "empirical_loss" in metric_names:
            totals["empirical_loss"] += float(residual.pow(2).sum().item())

    return {name: value / n for name, value in totals.items()}


@torch.no_grad()
def _compute_binary_parameter_metrics(
    model,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """Compute scalar probe parameter metrics at the current model state."""
    metric_names = set(metric_names)
    W1 = model.fc1.weight.detach()
    sqrt_m = math.sqrt(float(model.m))
    spectral_scale = 1.0 + math.sqrt(float(model.m) / float(model.d_in))
    out: Dict[str, float] = {}
    needs_fro = bool(metric_names & {"fc1_weight_fro_norm", "fc1_weight_fro_norm_normalized"})
    needs_spectral = bool(metric_names & {"fc1_weight_spectral_norm", "fc1_weight_spectral_norm_normalized"})
    if needs_fro:
        fro_norm = float(torch.linalg.matrix_norm(W1, ord="fro").item())
        if "fc1_weight_fro_norm" in metric_names:
            out["fc1_weight_fro_norm"] = fro_norm
        if "fc1_weight_fro_norm_normalized" in metric_names:
            out["fc1_weight_fro_norm_normalized"] = fro_norm / sqrt_m
    if needs_spectral:
        spectral_norm = float(torch.linalg.matrix_norm(W1, ord=2).item())
        if "fc1_weight_spectral_norm" in metric_names:
            out["fc1_weight_spectral_norm"] = spectral_norm
        if "fc1_weight_spectral_norm_normalized" in metric_names:
            out["fc1_weight_spectral_norm_normalized"] = spectral_norm / spectral_scale
    return out


@torch.no_grad()
def _compute_binary_gradient_metrics(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """
    Compute exact scalar-output parameter-gradient norms for the current model.

    For f(x) = scale * fc2(tanh(fc1(x))) with no biases:
      ||grad_fc2 f(x)||^2 = scale^2 * ||h||^2
      ||grad_fc1 f(x)||^2 = scale^2 * ||x||^2 *
          sum_j fc2_j^2 * (1 - h_j^2)^2
    """
    W1 = model.fc1.weight.detach()
    W2 = model.fc2.weight.detach().view(-1)
    scale = 1.0 / float(model.alpha) if model.init_type == "alpha" and model.alpha != 0 else 1.0
    scale_sq = scale * scale
    sqrt_m = math.sqrt(float(model.m))
    n = X.shape[0]
    metric_names = set(metric_names)
    needs_output_grad = bool(metric_names & set(BINARY_OUTPUT_GRAD_METRICS))
    needs_empirical_loss_grad = bool(metric_names & set(BINARY_EMPIRICAL_LOSS_GRAD_METRICS))

    emp_fc1_sum = torch.zeros_like(W1) if needs_empirical_loss_grad else None
    emp_fc2_sum = torch.zeros_like(W2) if needs_empirical_loss_grad else None

    mean_metrics = tuple(metric_names & set(BINARY_OUTPUT_GRAD_METRICS))
    totals = {name: 0.0 for name in mean_metrics}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]

        z = xb @ W1.T
        h = torch.tanh(z)

        one_minus_h2 = 1.0 - h.pow(2)
        fc1_out_grad_sq = None
        fc2_out_grad_sq = None
        total_out_grad_sq = None

        if needs_output_grad:
            x_norm_sq = xb.pow(2).sum(dim=1)
            fc2_out_grad_sq = scale_sq * h.pow(2).sum(dim=1)
            fc1_weighted_sq = (W2.pow(2).view(1, -1) * one_minus_h2.pow(2)).sum(dim=1)
            fc1_out_grad_sq = scale_sq * x_norm_sq * fc1_weighted_sq
            total_out_grad_sq = fc1_out_grad_sq + fc2_out_grad_sq

        if needs_output_grad:
            needs_fc2_norm = bool(metric_names & {"mean_output_grad_norm_fc2", "mean_output_grad_norm_fc2_normalized"})
            if "mean_output_grad_norm" in metric_names:
                totals["mean_output_grad_norm"] += float(torch.sqrt(total_out_grad_sq).sum().item())
            if "mean_output_grad_norm_fc1" in metric_names:
                totals["mean_output_grad_norm_fc1"] += float(torch.sqrt(fc1_out_grad_sq).sum().item())
            if needs_fc2_norm:
                fc2_out_grad_norm_sum = float(torch.sqrt(fc2_out_grad_sq).sum().item())
                if "mean_output_grad_norm_fc2" in metric_names:
                    totals["mean_output_grad_norm_fc2"] += fc2_out_grad_norm_sum
                if "mean_output_grad_norm_fc2_normalized" in metric_names:
                    totals["mean_output_grad_norm_fc2_normalized"] += fc2_out_grad_norm_sum / sqrt_m

        if needs_empirical_loss_grad:
            yb = y[start:end].view(-1)
            residual = scale * (h @ W2).view(-1) - yb
            coeff = 2.0 * residual * scale
            emp_fc2_sum += (coeff.view(-1, 1) * h).sum(dim=0)
            emp_fc1_batch = coeff.view(-1, 1) * W2.view(1, -1) * one_minus_h2
            emp_fc1_sum += emp_fc1_batch.T @ xb

    out = {name: value / n for name, value in totals.items()}

    if needs_empirical_loss_grad:
        emp_fc1 = emp_fc1_sum / n
        emp_fc2 = emp_fc2_sum / n
        emp_fc1_norm = float(torch.linalg.vector_norm(emp_fc1).item())
        emp_fc2_norm = float(torch.linalg.vector_norm(emp_fc2).item())
        emp_norm = math.sqrt(emp_fc1_norm * emp_fc1_norm + emp_fc2_norm * emp_fc2_norm)

        if "empirical_loss_grad_norm_fc1" in metric_names:
            out["empirical_loss_grad_norm_fc1"] = emp_fc1_norm
        if "empirical_loss_grad_norm_fc2" in metric_names:
            out["empirical_loss_grad_norm_fc2"] = emp_fc2_norm
        if "empirical_loss_grad_norm_fc2_normalized" in metric_names:
            out["empirical_loss_grad_norm_fc2_normalized"] = emp_fc2_norm / sqrt_m
        if "empirical_loss_grad_norm" in metric_names:
            out["empirical_loss_grad_norm"] = emp_norm

    return out


def get_binary_probe_stats(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    metric_names: Sequence[str],
    batch_size: int = 1024,
    jacobian_batch_size: int = 256,
) -> Dict[str, float]:
    """Compute the requested binary init-scale probe metrics."""
    metric_names = list(metric_names)
    forward_metrics = [name for name in metric_names if name in BINARY_FORWARD_METRICS]
    gradient_metrics = [name for name in metric_names if name in BINARY_GRADIENT_METRICS]
    parameter_metrics = [name for name in metric_names if name in BINARY_PARAMETER_METRICS]

    metrics: Dict[str, float] = {}
    if parameter_metrics:
        metrics.update(_compute_binary_parameter_metrics(model, metric_names=parameter_metrics))
    if forward_metrics:
        metrics.update(_compute_binary_forward_metrics(model, X, y, batch_size=batch_size, metric_names=forward_metrics))
    if gradient_metrics:
        metrics.update(
            _compute_binary_gradient_metrics(
                model,
                X,
                y,
                batch_size=jacobian_batch_size,
                metric_names=gradient_metrics,
            )
        )
    return metrics
