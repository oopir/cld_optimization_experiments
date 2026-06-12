import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .model import loss_fn
from .linearized import linearized_forward, compute_param_jacobians
from .metric_config import BASE_METRIC_NAMES, LIN_METRIC_NAMES

BINARY_CORE_METRICS = (
    "mean_abs_output",
    "mean_output_sq",
    "mean_preactivation_norm",
    "mean_abs_residual",
    "empirical_loss",
    "mean_output_grad_norm",
    "mean_output_grad_norm_sq",
    "empirical_loss_grad_norm",
)

BINARY_FORWARD_METRICS = (
    "mean_abs_output",
    "mean_output_sq",
    "mean_preactivation_norm",
    "mean_abs_residual",
    "empirical_loss",
)

BINARY_OUTPUT_GRAD_METRICS = (
    "mean_output_grad_norm",
    "mean_output_grad_norm_sq",
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_sq_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_sq_fc2",
)

BINARY_EMPIRICAL_LOSS_GRAD_METRICS = (
    "empirical_loss_grad_norm",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
)

BINARY_LAYERWISE_METRICS = (
    "mean_output_grad_norm_fc1",
    "mean_output_grad_norm_sq_fc1",
    "mean_output_grad_norm_fc2",
    "mean_output_grad_norm_sq_fc2",
    "empirical_loss_grad_norm_fc1",
    "empirical_loss_grad_norm_fc2",
)

BINARY_PARAMETER_METRICS = (
    "fc1_weight_fro_norm",
    "fc1_weight_spectral_norm",
)

BINARY_ALL_METRICS = BINARY_CORE_METRICS + BINARY_LAYERWISE_METRICS + BINARY_PARAMETER_METRICS
BINARY_GRADIENT_METRICS = BINARY_OUTPUT_GRAD_METRICS + BINARY_EMPIRICAL_LOSS_GRAD_METRICS

def _param_dist(params, params0):
    """Compute parameter displacement from the matching initialization tensors."""
    return torch.sqrt(sum((p - p0).pow(2).sum() for p, p0 in zip(params, params0))).item()


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
    output_metric_names = {"mean_abs_output", "mean_output_sq", "mean_abs_residual", "empirical_loss"}
    needs_output = bool(metric_names & output_metric_names)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        yb = y[start:end]

        z = model.fc1(xb)
        if "mean_preactivation_norm" in metric_names:
            totals["mean_preactivation_norm"] += float(torch.linalg.vector_norm(z, dim=1).sum().item())
        if not needs_output:
            continue

        out = model.fc2(torch.tanh(z))
        if model.init_type == "alpha" and model.alpha != 0:
            out = out / model.alpha
        out = out.view(-1, 1)

        if "mean_abs_output" in metric_names:
            totals["mean_abs_output"] += float(out.abs().sum().item())
        if "mean_output_sq" in metric_names:
            totals["mean_output_sq"] += float(out.pow(2).sum().item())
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
    out: Dict[str, float] = {}
    if "fc1_weight_fro_norm" in metric_names:
        out["fc1_weight_fro_norm"] = float(torch.linalg.matrix_norm(W1, ord="fro").item())
    if "fc1_weight_spectral_norm" in metric_names:
        out["fc1_weight_spectral_norm"] = float(torch.linalg.matrix_norm(W1, ord=2).item())
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
            if "mean_output_grad_norm" in metric_names:
                totals["mean_output_grad_norm"] += float(torch.sqrt(total_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq" in metric_names:
                totals["mean_output_grad_norm_sq"] += float(total_out_grad_sq.sum().item())
            if "mean_output_grad_norm_fc1" in metric_names:
                totals["mean_output_grad_norm_fc1"] += float(torch.sqrt(fc1_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq_fc1" in metric_names:
                totals["mean_output_grad_norm_sq_fc1"] += float(fc1_out_grad_sq.sum().item())
            if "mean_output_grad_norm_fc2" in metric_names:
                totals["mean_output_grad_norm_fc2"] += float(torch.sqrt(fc2_out_grad_sq).sum().item())
            if "mean_output_grad_norm_sq_fc2" in metric_names:
                totals["mean_output_grad_norm_sq_fc2"] += float(fc2_out_grad_sq.sum().item())

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

        if "empirical_loss_grad_norm_fc1" in metric_names:
            out["empirical_loss_grad_norm_fc1"] = emp_fc1_norm
        if "empirical_loss_grad_norm_fc2" in metric_names:
            out["empirical_loss_grad_norm_fc2"] = emp_fc2_norm
        if "empirical_loss_grad_norm" in metric_names:
            out["empirical_loss_grad_norm"] = math.sqrt(emp_fc1_norm * emp_fc1_norm + emp_fc2_norm * emp_fc2_norm)

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

def _classification_metrics(model, X, y, loss_targets=None, batch_size=1024):
    """Evaluate accuracy and optional MSE loss in batches for NN metrics."""
    n = X.size(0)
    total_correct = 0
    total_loss = 0.0

    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = X[start:end]
        out = model(xb)
        total_correct += (out.argmax(dim=1) == y[start:end]).sum().item()

        if loss_targets is not None:
            target = loss_targets[start:end]
            total_loss += loss_fn(out, target).item() * len(xb)

    acc = total_correct / n
    loss = total_loss / n if loss_targets is not None else None
    return acc, loss

def _feature_stats(model, X_train, A0, A0_norm, metric_names):
    """Compute only the requested hidden-feature drift metrics."""
    stats = {}
    A_t = torch.tanh(model.fc1(X_train))

    if "feat_rel_dist" in metric_names:
        stats["feat_rel_dist"] = (A_t - A0).norm().item() / (A0_norm + 1e-12)

    if "feat_cos_dist" in metric_names:
        cos_sim = F.cosine_similarity(A_t.view(-1), A0.view(-1), dim=0).item()
        stats["feat_cos_dist"] = 1.0 - cos_sim

    if "feat_gram_lambda" in metric_names:
        A_Gram = A_t @ A_t.T
        A_Gram = 0.5 * (A_Gram + A_Gram.T)
        try:
            stats["feat_gram_lambda"] = torch.linalg.eigvalsh(A_Gram)[0].item()
        except Exception:
            print("Numerical instability occurred when computing lambda_min of feature matrix. Defaulting to nan.")
            stats["feat_gram_lambda"] = float("nan")

    return stats

@torch.no_grad()
def get_stats(model, params, params0, A0, A0_norm, data, metric_plan):
    """Compute the requested NN stats."""
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    metrics = metric_plan.compute_metrics
    stats = {}

    if "train_acc" in metrics or "train_loss" in metrics:
        # MSE classification runs use one-hot targets; regression-style runs can omit them.
        train_targets = data.get("y_train_one_hot", y_train)
        train_acc, train_loss = _classification_metrics(model, X_train, y_train, train_targets)
        if "train_acc" in metrics:
            stats["train_acc"] = train_acc
        if "train_loss" in metrics:
            stats["train_loss"] = train_loss

    if "test_acc" in metrics:
        test_acc, _ = _classification_metrics(model, X_test, y_test)
        stats["test_acc"] = test_acc

    if "param_dist" in metrics:
        stats["param_dist"] = _param_dist(params, params0)

    if metric_plan.needs_feature_activations:
        stats.update(_feature_stats(model, X_train, A0, A0_norm, metrics))

    return stats

@torch.no_grad()
def get_linear_stats(model, base_params_dict, lin_params, lin_params0, data, metric_plan):
    """Compute the requested linearized model stats."""
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    metrics = metric_plan.compute_metrics
    stats = {}

    if "lin_train_loss" in metrics or "lin_train_acc" in metrics:
        outputs_train = linearized_forward(model, base_params_dict, lin_params, X_train)
        if "lin_train_acc" in metrics:
            pred_train = outputs_train.argmax(dim=1)
            stats["lin_train_acc"] = (pred_train == y_train).float().mean().item()
        if "lin_train_loss" in metrics:
            # Same target convention as get_stats: prefer one-hot labels when present.
            stats["lin_train_loss"] = loss_fn(outputs_train, data.get("y_train_one_hot", y_train)).item()

    if "lin_test_acc" in metrics:
        outputs_test = linearized_forward(model, base_params_dict, lin_params, X_test)
        pred_test = outputs_test.argmax(dim=1)
        stats["lin_test_acc"] = (pred_test == y_test).float().mean().item()

    if "lin_param_dist" in metrics:
        stats["lin_param_dist"] = _param_dist(lin_params, lin_params0)

    return stats

@torch.no_grad()
def get_nn_lin_param_dist(params, lin_params, normalize_by=None, eps=1e-12):
    """Return L2 and cosine distance between NN and linearized parameters."""
    total_sq = 0
    dot      = 0
    norm_n   = 0
    norm_l   = 0
    for pn, pl in zip(params, lin_params):
        total_sq += float(((pn - pl)**2).sum().item())
        dot += float((pn * pl).sum().item())
        norm_n += float((pn**2).sum().item())
        norm_l += float((pl**2).sum().item())

    l2_dist = math.sqrt(total_sq)
    if normalize_by is not None:
        l2_dist /= float(normalize_by) + eps

    cos_sim = dot / ((math.sqrt(norm_n) * math.sqrt(norm_l)) + eps)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    cos_dist = 1.0 - cos_sim

    return l2_dist, cos_dist

def compute_dataset_jac_drift(model, model_at_init, X_data, batch_size=1, eps=1e-12):
    """Estimate dataset NTK drift by recomputing current/init Jacobians in batches."""
    device = next(model.parameters()).device
    total_sq = 0.0
    dot = 0.0
    norm_c_sq = 0.0
    norm_i_sq = 0.0

    n = X_data.shape[0]
    for start in range(0, n, batch_size):
        X_batch = X_data[start:start + batch_size].to(device)

        jac_curr = compute_param_jacobians(model, X_batch)
        jac_init = compute_param_jacobians(model_at_init, X_batch)

        for jc, ji in zip(jac_curr, jac_init):
            diff = jc - ji
            total_sq += float(diff.pow(2).sum().item())

            dot += float((jc * ji).sum().item())
            norm_c_sq += float((jc * jc).sum().item())
            norm_i_sq += float((ji * ji).sum().item())

        del jac_curr, jac_init  # free per-batch Jacobians  

    l2_dist  = math.sqrt(total_sq) / (math.sqrt(norm_i_sq) + eps)

    cos_sim = dot / ((math.sqrt(norm_c_sq) * math.sqrt(norm_i_sq)) + eps)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    cos_dist = 1.0 - cos_sim

    return l2_dist, cos_dist

@torch.no_grad()
def estimate_lambda_min(X, M=10000, batch_g=64, device=None):
    """Monte Carlo estimate of lambda_min(E[phi(Xg) phi(Xg)])."""
    n, d = X.shape
    A = torch.zeros((n, n), device=device, dtype=X.dtype)
    done = 0
    while done < M:
        b = min(batch_g, M - done)
        G = torch.randn(d, b, device=device, dtype=X.dtype)
        Y = torch.tanh(X @ G)
        A += (Y @ Y.T)
        done += b

    A /= M
    A = (A + A.T) * 0.5
    try:
        lam_min = torch.linalg.eigvalsh(A)[0].item()
    except Exception:
        lam_min = float("nan")
    return lam_min

def estimate_loss_floor(X_train, noisy_beta, m, device):
    """Estimate the analysis-predicted loss floor for noisy training."""
    n, d = X_train.shape
    lambda_min = estimate_lambda_min(X_train, device=device)
    loss_floor = (2/lambda_min) * ((n/noisy_beta) * (1 + d/m) + (n/noisy_beta)**2 * (1 + d*d/m))
    return loss_floor
