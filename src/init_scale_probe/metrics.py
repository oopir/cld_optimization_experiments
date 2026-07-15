import math
from typing import Dict, Optional, Sequence, Tuple

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

BINARY_TEST_METRICS = (
    "test_error",
)

BINARY_CLASSIFICATION_ERROR_METRICS = (
    "train_error",
    "test_error",
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
    "fc2_weight_euclidean_norm",
)

PROBE_NTK_EIGEN_METRICS = (
    "ntk_eig_min",
    "ntk_eig_max",
    "ntk_eig_mean",
    "ntk_eig_median",
)

PROBE_NTK_ALIGNMENT_METRICS = (
    "residual_ntk_alignment",
    "residual_ntk_alignment_over_ntk_eig_min",
    "residual_ntk_alignment_over_ntk_eig_mean",
    "residual_ntk_alignment_over_ntk_eig_max",
    "empirical_loss_times_residual_ntk_alignment",
    "empirical_loss_times_ntk_eig_min",
    "residual_ntk_alignment_residual_dynamics_term",
    "residual_ntk_alignment_ntk_dynamics_term",
)
PROBE_NTK_DRIFT_METRICS = (
    "residual_initial_ntk_alignment",
    "residual_ntk_alignment_over_initial",
    "residual_ntk_alignment_trace_normalized_over_initial",
    "task_ntk_alignment",
    "task_initial_ntk_alignment",
    "task_ntk_alignment_over_initial",
    "task_ntk_alignment_trace_normalized_over_initial",
    "ntk_cos_dist",
    "ntk_rel_fro_dist",
)
PROBE_NTK_INITIAL_MATRIX_METRICS = (
    "residual_initial_ntk_alignment",
    "residual_ntk_alignment_over_initial",
    "residual_ntk_alignment_trace_normalized_over_initial",
    "task_initial_ntk_alignment",
    "task_ntk_alignment_over_initial",
    "task_ntk_alignment_trace_normalized_over_initial",
    "ntk_cos_dist",
    "ntk_rel_fro_dist",
)
PROBE_NTK_MATRIX_METRICS = PROBE_NTK_EIGEN_METRICS + (
    "residual_ntk_alignment",
    "residual_ntk_alignment_over_ntk_eig_min",
    "residual_ntk_alignment_over_ntk_eig_mean",
    "residual_ntk_alignment_over_ntk_eig_max",
    "empirical_loss_times_residual_ntk_alignment",
    "empirical_loss_times_ntk_eig_min",
    "residual_ntk_alignment_residual_dynamics_term",
    "residual_ntk_alignment_over_initial",
    "residual_ntk_alignment_trace_normalized_over_initial",
    "task_ntk_alignment",
    "task_ntk_alignment_over_initial",
    "task_ntk_alignment_trace_normalized_over_initial",
    "ntk_cos_dist",
    "ntk_rel_fro_dist",
)
PROBE_NTK_HVP_METRICS = (
    "residual_ntk_alignment_ntk_dynamics_term",
)

PROBE_NTK_LABEL_ENERGY_PREFIX = "ntk_label_energy_top_"
PROBE_NTK_RESIDUAL_ENERGY_PREFIX = "ntk_residual_energy_top_"
PROBE_NTK_STATIC_METRICS = PROBE_NTK_EIGEN_METRICS + PROBE_NTK_ALIGNMENT_METRICS + PROBE_NTK_DRIFT_METRICS
PROBE_NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES = {
    "loss_weighted_residual_ntk_alignment": ("empirical_loss", "empirical_loss_times_residual_ntk_alignment"),
    "loss_weighted_ntk_eig_min": ("empirical_loss", "empirical_loss_times_ntk_eig_min"),
}
PROBE_NTK_LOSS_WEIGHTED_AVERAGE_METRICS = tuple(PROBE_NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES)
PROBE_NTK_LOSS_WEIGHTED_AVERAGE_BASE_METRICS = {
    "loss_weighted_residual_ntk_alignment": "residual_ntk_alignment",
    "loss_weighted_ntk_eig_min": "ntk_eig_min",
}

BINARY_DEFAULT_METRICS = BINARY_CORE_METRICS + BINARY_LAYERWISE_METRICS + BINARY_PARAMETER_METRICS
BINARY_ALL_METRICS = BINARY_DEFAULT_METRICS + BINARY_CLASSIFICATION_ERROR_METRICS + PROBE_NTK_STATIC_METRICS
BINARY_GRADIENT_METRICS = BINARY_OUTPUT_GRAD_METRICS + BINARY_EMPIRICAL_LOSS_GRAD_METRICS


def append_k_to_ntk_label_energy_metric(k: int) -> str:
    return f"{PROBE_NTK_LABEL_ENERGY_PREFIX}{int(k)}"


def append_k_to_ntk_residual_energy_metric(k: int) -> str:
    return f"{PROBE_NTK_RESIDUAL_ENERGY_PREFIX}{int(k)}"


def _parse_positive_int_suffix(name: str, prefix: str) -> Optional[int]:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    if not suffix.isdigit():
        return None
    k = int(suffix)
    return k if k > 0 else None


def parse_ntk_label_energy_metric(name: str) -> Optional[int]:
    return _parse_positive_int_suffix(name, PROBE_NTK_LABEL_ENERGY_PREFIX)


def parse_ntk_residual_energy_metric(name: str) -> Optional[int]:
    return _parse_positive_int_suffix(name, PROBE_NTK_RESIDUAL_ENERGY_PREFIX)


def parse_ntk_energy_metric(name: str) -> Optional[int]:
    label_k = parse_ntk_label_energy_metric(name)
    return label_k if label_k is not None else parse_ntk_residual_energy_metric(name)


def is_ntk_metric(name: str) -> bool:
    return name in PROBE_NTK_STATIC_METRICS or parse_ntk_energy_metric(name) is not None


def ntk_metric_needs_matrix(name: str) -> bool:
    return name in PROBE_NTK_MATRIX_METRICS or ntk_metric_needs_initial_matrix(name) or parse_ntk_energy_metric(name) is not None


def ntk_metric_needs_initial_matrix(name: str) -> bool:
    return name in PROBE_NTK_INITIAL_MATRIX_METRICS


def ntk_metric_needs_hvp(name: str) -> bool:
    return name in PROBE_NTK_HVP_METRICS


def is_ntk_loss_weighted_average_metric(name: str) -> bool:
    return name in PROBE_NTK_LOSS_WEIGHTED_AVERAGE_METRICS


def ntk_loss_weighted_average_dependencies(name: str) -> Optional[Tuple[str, ...]]:
    return PROBE_NTK_LOSS_WEIGHTED_AVERAGE_DEPENDENCIES.get(name)


def ntk_loss_weighted_average_base_metric(name: str) -> Optional[str]:
    return PROBE_NTK_LOSS_WEIGHTED_AVERAGE_BASE_METRICS.get(name)


def _output_scale(model) -> float:
    return 1.0 / float(model.alpha) if model.init_type == "alpha" and model.alpha != 0 else 1.0


def _activation_and_derivative(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tanh(z) and its derivative."""
    h = torch.tanh(z)
    return h, 1.0 - h.pow(2)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if float(denominator.detach().item()) == 0.0:
        return float("nan")
    return float((numerator / denominator).detach().item())


def _metric_or_nan(value: Optional[torch.Tensor]) -> float:
    return float("nan") if value is None else float(value.detach().item())


@torch.no_grad()
def _binary_hidden_forward(
    model,
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return scalar hidden features, outputs, activation derivatives, and optional residuals."""
    W1 = model.fc1.weight.detach()
    W2 = model.fc2.weight.detach().view(-1)
    h, activation_derivative = _activation_and_derivative(X @ W1.T)
    out = _output_scale(model) * (h @ W2).view(-1)
    residual = None if y is None else out - y.view(-1)
    return h, out, activation_derivative, residual


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

        h, _ = _activation_and_derivative(z)
        out = _output_scale(model) * model.fc2(h)
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
def _compute_binary_error(
    model,
    batch_size: int,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute sign-threshold binary classification error on one tensor dataset."""
    n = X.shape[0]
    incorrect = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = model(X[start:end]).view(-1)
        pred = torch.where(out >= 0, 1.0, -1.0).to(dtype=y.dtype, device=y.device)
        incorrect += int((pred != y[start:end].view(-1)).sum().item())
    return incorrect / n


@torch.no_grad()
def _compute_binary_classification_error_metrics(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test: Optional[torch.Tensor],
    y_test: Optional[torch.Tensor],
    batch_size: int,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """Compute sign-threshold binary classification error metrics."""
    metric_names = set(metric_names)
    metrics: Dict[str, float] = {}
    if "train_error" in metric_names:
        metrics["train_error"] = _compute_binary_error(model, batch_size=batch_size, X=X, y=y)
    if "test_error" in metric_names:
        if X_test is None or y_test is None or X_test.shape[0] == 0:
            metrics["test_error"] = float("nan")
        else:
            metrics["test_error"] = _compute_binary_error(model, batch_size=batch_size, X=X_test, y=y_test)
    return metrics


@torch.no_grad()
def _compute_binary_parameter_metrics(
    model,
    metric_names: Sequence[str],
) -> Dict[str, float]:
    """Compute scalar probe parameter metrics at the current model state."""
    metric_names = set(metric_names)
    W1 = model.fc1.weight.detach()
    W2 = model.fc2.weight.detach()
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
    if "fc2_weight_euclidean_norm" in metric_names:
        out["fc2_weight_euclidean_norm"] = float(torch.linalg.vector_norm(W2).item())
    return out

# TODO: Simplify this control flow when revisiting the binary gradient metrics.
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

    For f(x) = scale * fc2(phi(fc1(x))) with no biases:
      ||grad_fc2 f(x)||^2 = scale^2 * ||h||^2
      ||grad_fc1 f(x)||^2 = scale^2 * ||x||^2 * sum_j fc2_j^2 * phi'(z_j)^2
    """
    W1 = model.fc1.weight.detach()
    W2 = model.fc2.weight.detach().view(-1)
    scale = _output_scale(model)
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
        h, activation_derivative = _activation_and_derivative(z)
        fc1_out_grad_sq = None
        fc2_out_grad_sq = None
        total_out_grad_sq = None

        if needs_output_grad:
            x_norm_sq = xb.pow(2).sum(dim=1)
            fc2_out_grad_sq = scale_sq * h.pow(2).sum(dim=1)
            fc1_out_grad_sq = scale_sq * x_norm_sq * (W2.pow(2).view(1, -1) * activation_derivative.pow(2)).sum(dim=1)
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
            emp_fc1_batch = coeff.view(-1, 1) * W2.view(1, -1) * activation_derivative
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


@torch.no_grad()
def _compute_ntk_matrix(
    model,
    X: torch.Tensor,
    h: torch.Tensor,
    activation_derivative: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute the raw scalar two-layer activation NTK on the full dataset.

    For f(x) = s * W2 phi(W1 x), no biases:
      K_ij = s^2 * [h_i^T h_j + (x_i^T x_j) * sum_a W2_a^2 phi'(z_ia) phi'(z_ja)].
    """
    W2 = model.fc2.weight.detach().view(-1)
    scale = _output_scale(model)
    scale_sq = scale * scale
    n = X.shape[0]
    batch_size = max(1, int(batch_size))

    weighted_derivatives = activation_derivative * W2.pow(2).view(1, -1)

    K = torch.empty((n, n), device=X.device, dtype=X.dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        hidden_gram = h[start:end] @ h.T
        input_gram = X[start:end] @ X.T
        derivative_gram = weighted_derivatives[start:end] @ activation_derivative.T
        K[start:end] = scale_sq * (hidden_gram + input_gram * derivative_gram)
    return 0.5 * (K + K.T)


def _dot_param_lists(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for a, b in zip(left, right):
        term = (a * b).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Cannot take a parameter-list dot product over an empty list.")
    return total


def _sorted_median(values: torch.Tensor) -> torch.Tensor:
    n = int(values.numel())
    if n == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _compute_ntk_alignment_ntk_dynamics_term(model, X: torch.Tensor, residual: torch.Tensor) -> float:
    """
    Compute the NTK-dynamics term in d/dt [u^T K u], holding r and u fixed.
    With u     = r / ||r||,
         J^T u = grad_theta(u^T f), and
         H_r   = grad_theta^2(r^T f) = sum_i r_i grad_theta^2 f_i,
    this returns D_ntk = -(2/n) (J^T u)^T H_r (J^T u).
    This is the second term in dA/dt.
    """
    n = X.shape[0]
    denom = residual.pow(2).sum().detach()
    if float(denom.item()) <= 0.0:
        return float("nan")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return float("nan")

    residual_detached = residual.detach()
    u_detached = residual_detached / torch.sqrt(denom)
    f = model(X).view(-1)
    ut_f = (u_detached * f).sum()
    rt_f = (residual_detached * f).sum()

    # Jt_u = grad_theta(u^T f).
    Jt_u = torch.autograd.grad(ut_f, params, retain_graph=True, create_graph=False)
    # grad_rt_f = grad_theta(r^T f), kept differentiable so we can apply H_r.
    grad_rt_f = torch.autograd.grad(rt_f, params, create_graph=True)
    # Hr_Jt_u = grad_theta^2(r^T f) [J^T u].
    Hr_Jt_u = torch.autograd.grad(grad_rt_f, params, grad_outputs=[part.detach() for part in Jt_u], allow_unused=False)

    ut_J_Ht_Jt_u = _dot_param_lists([part.detach() for part in Jt_u], Hr_Jt_u)

    return float((-(2.0 / float(n)) * ut_J_Ht_Jt_u).detach().item())


def _compute_residual_ntk_alignment(K: torch.Tensor, residual: torch.Tensor, denom: torch.Tensor) -> Optional[torch.Tensor]:
    """Return A = r^T K r / ||r||^2, or None for zero residual."""
    if float(denom.detach().item()) <= 0.0:
        return None
    return (residual @ (K @ residual)) / denom


def _compute_residual_dynamics_term(K: torch.Tensor, residual: torch.Tensor, denom: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Return D_res = -(2/n) ||(I - uu^T) K u||^2 for u = r / ||r||.
    """
    if float(denom.detach().item()) <= 0.0:
        return None
    u = residual / torch.sqrt(denom)
    Ku = K @ u
    projected_Ku = Ku - u * (u @ Ku)
    return -(2.0 / float(residual.shape[0])) * projected_Ku.pow(2).sum()


def _compute_ntk_cosine_distance(K: torch.Tensor, K0: torch.Tensor) -> float:
    """Return 1 - cosine similarity between flattened current and initial NTKs."""
    K_flat = K.reshape(-1)
    K0_flat = K0.reshape(-1)
    denom = torch.linalg.vector_norm(K_flat) * torch.linalg.vector_norm(K0_flat)
    if float(denom.detach().item()) <= 0.0:
        return float("nan")
    similarity = torch.dot(K_flat, K0_flat) / denom
    similarity = torch.clamp(similarity, -1.0, 1.0)
    return float((1.0 - similarity).detach().item())


def _compute_ntk_relative_frobenius_distance(K: torch.Tensor, K0: torch.Tensor) -> float:
    """Return ||K - K0||_F / ||K0||_F."""
    denom = torch.linalg.matrix_norm(K0, ord="fro")
    if float(denom.detach().item()) <= 0.0:
        return float("nan")
    return float((torch.linalg.matrix_norm(K - K0, ord="fro") / denom).detach().item())


def _trace_normalized_alignment_ratio(
    current_alignment: Optional[torch.Tensor],
    current_trace: Optional[torch.Tensor],
    initial_alignment: Optional[torch.Tensor],
    initial_trace: torch.Tensor,
) -> float:
    if current_alignment is None or current_trace is None or initial_alignment is None:
        return float("nan")
    if float(current_trace.detach().item()) <= 0.0 or float(initial_trace.detach().item()) <= 0.0:
        return float("nan")
    return _safe_ratio(current_alignment / current_trace, initial_alignment / initial_trace)


def _check_initial_ntk_matrix(initial_ntk_matrix: torch.Tensor, K_shape: Tuple[int, int], device, dtype) -> torch.Tensor:
    if initial_ntk_matrix.shape != K_shape:
        raise ValueError(
            "initial_ntk_matrix must have shape "
            f"{tuple(K_shape)}, got {tuple(initial_ntk_matrix.shape)}."
        )
    return initial_ntk_matrix.to(device=device, dtype=dtype)


def _needs_ntk_matrix(metric_set: set, label_energy_ks: Sequence[int]) -> bool:
    return bool(label_energy_ks) or any(ntk_metric_needs_matrix(name) for name in metric_set)


def _needs_current_ntk_matrix(metric_set: set, label_energy_ks: Sequence[int]) -> bool:
    return bool(label_energy_ks) or any(name in PROBE_NTK_MATRIX_METRICS for name in metric_set)


def compute_binary_probe_ntk_matrix(
    model,
    X: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute the scalar-output binary probe NTK matrix at the model's current state."""
    h, _, activation_derivative, _ = _binary_hidden_forward(model, X)
    return _compute_ntk_matrix(model, X, h, activation_derivative, batch_size=batch_size).detach()


def _compute_ntk_metrics(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    metric_names: Sequence[str],
    initial_ntk_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute requested raw-NTK spectral/alignment metrics for the scalar binary probe."""
    metric_names = list(metric_names)
    metric_set = set(metric_names)
    label_energy_ks = sorted(
        {
            k for k in (parse_ntk_label_energy_metric(name) for name in metric_names)
            if k is not None
        }
    )
    residual_energy_ks = sorted(
        {
            k for k in (parse_ntk_residual_energy_metric(name) for name in metric_names)
            if k is not None
        }
    )
    energy_ks = sorted({*label_energy_ks, *residual_energy_ks})
    needs_matrix = _needs_current_ntk_matrix(metric_set, energy_ks)
    needs_eigenvectors = bool(energy_ks)
    needs_initial_matrix = any(ntk_metric_needs_initial_matrix(name) for name in metric_set)

    metrics: Dict[str, float] = {}
    h, _, activation_derivative, residual = _binary_hidden_forward(model, X, y)
    if residual is None:
        raise RuntimeError("Internal error: NTK metrics require residuals.")
    denom = residual.pow(2).sum()
    y_vec = y.view(-1)
    y_denom = y_vec.pow(2).sum()
    alignment = None
    task_alignment = None
    current_trace = None
    K0 = None
    if needs_initial_matrix:
        if initial_ntk_matrix is None:
            raise ValueError("initial_ntk_matrix is required for initialization-NTK metrics.")
        K0 = _check_initial_ntk_matrix(initial_ntk_matrix, (X.shape[0], X.shape[0]), X.device, X.dtype)

    if needs_matrix:
        K = _compute_ntk_matrix(
            model,
            X,
            h,
            activation_derivative,
            batch_size=batch_size,
        )
        current_trace = torch.trace(K)
        if needs_eigenvectors:
            eigenvalues, eigenvectors = torch.linalg.eigh(K)
        else:
            # print("K finite", torch.isfinite(K).all().item())
            # print("K symmetry err", (K - K.T).abs().max().item())
            # print("K max abs", K.abs().max().item())
            # print("K diag min/max", K.diag().min().item(), K.diag().max().item())
            # print("K dtype/device", K.dtype, K.device)
            eigenvalues = torch.linalg.eigvalsh(K)
            eigenvectors = None

        eig = {
            "ntk_eig_min": eigenvalues[0],
            "ntk_eig_max": eigenvalues[-1],
            "ntk_eig_mean": eigenvalues.mean(),
            "ntk_eig_median": _sorted_median(eigenvalues),
        }
        for name, value in eig.items():
            if name in metric_set:
                metrics[name] = float(value.item())

        alignment = _compute_residual_ntk_alignment(K, residual, denom)
        task_alignment = _compute_residual_ntk_alignment(K, y_vec, y_denom)

        if "residual_ntk_alignment" in metric_set:
            metrics["residual_ntk_alignment"] = _metric_or_nan(alignment)
        if "task_ntk_alignment" in metric_set:
            metrics["task_ntk_alignment"] = _metric_or_nan(task_alignment)

        for ratio_name, eig_name in (
            ("residual_ntk_alignment_over_ntk_eig_min", "ntk_eig_min"),
            ("residual_ntk_alignment_over_ntk_eig_mean", "ntk_eig_mean"),
            ("residual_ntk_alignment_over_ntk_eig_max", "ntk_eig_max"),
        ):
            if ratio_name in metric_set:
                metrics[ratio_name] = float("nan") if alignment is None else _safe_ratio(alignment, eig[eig_name])

        empirical_loss = residual.pow(2).mean()
        if "empirical_loss_times_residual_ntk_alignment" in metric_set:
            metrics["empirical_loss_times_residual_ntk_alignment"] = (
                float("nan") if alignment is None else float((empirical_loss * alignment).item())
            )
        if "empirical_loss_times_ntk_eig_min" in metric_set:
            metrics["empirical_loss_times_ntk_eig_min"] = float((empirical_loss * eig["ntk_eig_min"]).item())

        if "residual_ntk_alignment_residual_dynamics_term" in metric_set:
            metrics["residual_ntk_alignment_residual_dynamics_term"] = _metric_or_nan(
                _compute_residual_dynamics_term(K, residual, denom)
            )

        if K0 is not None:
            if "ntk_cos_dist" in metric_set:
                metrics["ntk_cos_dist"] = _compute_ntk_cosine_distance(K, K0)
            if "ntk_rel_fro_dist" in metric_set:
                metrics["ntk_rel_fro_dist"] = _compute_ntk_relative_frobenius_distance(K, K0)

        if needs_eigenvectors:
            # Label energy is sum_i<=k (u_i^T y)^2 / ||y||^2, where
            # u_i are NTK eigenvectors ordered by decreasing eigenvalue.
            top_eigenvectors = torch.flip(eigenvectors, dims=(1,))
            for k in label_energy_ks:
                name = append_k_to_ntk_label_energy_metric(k)
                k_eff = min(int(k), top_eigenvectors.shape[1])
                if float(y_denom.item()) <= 0.0:
                    metrics[name] = float("nan")
                else:
                    projection = top_eigenvectors[:, :k_eff].T @ y_vec
                    metrics[name] = float((projection.pow(2).sum() / y_denom).item())

            # Residual energy is sum_i<=k (u_i^T r)^2 / ||r||^2 using the current residual r.
            for k in residual_energy_ks:
                name = append_k_to_ntk_residual_energy_metric(k)
                k_eff = min(int(k), top_eigenvectors.shape[1])
                if float(denom.item()) <= 0.0:
                    metrics[name] = float("nan")
                else:
                    projection = top_eigenvectors[:, :k_eff].T @ residual
                    metrics[name] = float((projection.pow(2).sum() / denom).item())

    if K0 is not None:
        initial_trace = torch.trace(K0)
        initial_alignment = _compute_residual_ntk_alignment(K0, residual, denom)
        initial_task_alignment = _compute_residual_ntk_alignment(K0, y_vec, y_denom)
        if "residual_initial_ntk_alignment" in metric_set:
            metrics["residual_initial_ntk_alignment"] = _metric_or_nan(initial_alignment)
        if "residual_ntk_alignment_over_initial" in metric_set:
            metrics["residual_ntk_alignment_over_initial"] = (
                float("nan") if alignment is None or initial_alignment is None else _safe_ratio(alignment, initial_alignment)
            )
        if "residual_ntk_alignment_trace_normalized_over_initial" in metric_set:
            metrics["residual_ntk_alignment_trace_normalized_over_initial"] = _trace_normalized_alignment_ratio(
                alignment,
                current_trace,
                initial_alignment,
                initial_trace,
            )
        if "task_initial_ntk_alignment" in metric_set:
            metrics["task_initial_ntk_alignment"] = _metric_or_nan(initial_task_alignment)
        if "task_ntk_alignment_over_initial" in metric_set:
            metrics["task_ntk_alignment_over_initial"] = (
                float("nan") if task_alignment is None or initial_task_alignment is None else _safe_ratio(
                    task_alignment,
                    initial_task_alignment,
                )
            )
        if "task_ntk_alignment_trace_normalized_over_initial" in metric_set:
            metrics["task_ntk_alignment_trace_normalized_over_initial"] = _trace_normalized_alignment_ratio(
                task_alignment,
                current_trace,
                initial_task_alignment,
                initial_trace,
            )

    if "residual_ntk_alignment_ntk_dynamics_term" in metric_set:
        metrics["residual_ntk_alignment_ntk_dynamics_term"] = _compute_ntk_alignment_ntk_dynamics_term(model, X, residual)

    return metrics


def get_binary_probe_stats(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    metric_names: Sequence[str],
    batch_size: int = 1024,
    jacobian_batch_size: int = 256,
    X_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    initial_ntk_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute the requested binary init-scale probe metrics."""
    metric_names = list(metric_names)
    forward_metrics = [name for name in metric_names if name in BINARY_FORWARD_METRICS]
    classification_error_metrics = [name for name in metric_names if name in BINARY_CLASSIFICATION_ERROR_METRICS]
    gradient_metrics = [name for name in metric_names if name in BINARY_GRADIENT_METRICS]
    parameter_metrics = [name for name in metric_names if name in BINARY_PARAMETER_METRICS]
    ntk_metrics = [name for name in metric_names if is_ntk_metric(name)]

    metrics: Dict[str, float] = {}
    if parameter_metrics:
        metrics.update(_compute_binary_parameter_metrics(model, metric_names=parameter_metrics))
    if forward_metrics:
        metrics.update(_compute_binary_forward_metrics(model, X, y, batch_size=batch_size, metric_names=forward_metrics))
    if classification_error_metrics:
        metrics.update(
            _compute_binary_classification_error_metrics(
                model,
                X,
                y,
                X_test,
                y_test,
                batch_size=batch_size,
                metric_names=classification_error_metrics,
            )
        )
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
    if ntk_metrics:
        metrics.update(
            _compute_ntk_metrics(
                model,
                X,
                y,
                batch_size=jacobian_batch_size,
                metric_names=ntk_metrics,
                initial_ntk_matrix=initial_ntk_matrix,
            )
        )
    return metrics
