#!/usr/bin/env python3
"""Validate the at-init-stats spectral metrics on a tiny CPU example."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.base.model import TwoLayerNet
from src.training_stats.sweep import _rows_for_trained_initialization
from src.training_stats.metrics import (
    _hidden_forward,
    _compute_ntk_matrix,
    _sorted_median,
    append_k_to_ntk_label_energy_metric,
    append_k_to_ntk_residual_energy_metric,
    get_metrics,
)
from src.at_init_stats.core import AtInitStatsConfig


def _flatten_tensors(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _explicit_jacobian(model, params, X):
    rows = []
    for i in range(X.shape[0]):
        out = model(X[i : i + 1]).view(())
        grads = torch.autograd.grad(out, params, retain_graph=True)
        rows.append(_flatten_tensors(grads))
    return torch.stack(rows, dim=0)


def _assert_close(name, actual, expected, rtol=1e-5, atol=1e-7):
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = torch.max(torch.abs(actual - expected)).item()
        raise AssertionError(f"{name} mismatch: max abs diff={diff}")


def _dot_tensors(left, right):
    return sum((a * b).sum() for a, b in zip(left, right))


def _validate_tanh():
    torch.manual_seed(0)
    dtype = torch.float64
    d_in = 5
    width = 3
    n = 4

    model = TwoLayerNet(d_in=d_in, m=width, d_out=1, init_type="standard", alpha=1.0).to(dtype=dtype)
    params = list(model.parameters())
    X = torch.randn(n, d_in, dtype=dtype)
    y = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype).view(-1, 1)

    h, _, activation_derivative, residual = _hidden_forward(model, X, y)
    closed_form_K = _compute_ntk_matrix(model, X, h, activation_derivative, batch_size=2)
    explicit_J = _explicit_jacobian(model, params, X)
    explicit_K = explicit_J @ explicit_J.T
    _assert_close("tanh closed-form NTK", closed_form_K, explicit_K)

    metric_names = [
        "ntk_eig_min",
        "ntk_eig_max",
        "ntk_eig_mean",
        "ntk_eig_median",
        "residual_ntk_alignment",
        "residual_ntk_alignment_over_ntk_eig_min",
        "residual_ntk_alignment_over_ntk_eig_mean",
        "residual_ntk_alignment_over_ntk_eig_max",
        "empirical_loss_times_residual_ntk_alignment",
        "empirical_loss_times_ntk_eig_min",
        "residual_ntk_alignment_residual_dynamics_term",
        "residual_ntk_alignment_ntk_dynamics_term",
        "residual_initial_ntk_alignment",
        "residual_ntk_alignment_over_initial",
        "residual_ntk_alignment_trace_normalized_over_initial",
        "task_ntk_alignment",
        "task_initial_ntk_alignment",
        "task_ntk_alignment_over_initial",
        "task_ntk_alignment_trace_normalized_over_initial",
        "ntk_cos_dist",
        "ntk_rel_fro_dist",
        append_k_to_ntk_label_energy_metric(1),
        append_k_to_ntk_label_energy_metric(3),
        append_k_to_ntk_residual_energy_metric(1),
        append_k_to_ntk_residual_energy_metric(3),
    ]
    metrics = get_metrics(
        model,
        X,
        y,
        metric_names,
        jacobian_batch_size=2,
        initial_ntk_matrix=closed_form_K,
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(closed_form_K)
    alignment = residual @ (closed_form_K @ residual) / residual.pow(2).sum()
    y_vec = y.view(-1)
    task_alignment = y_vec @ (closed_form_K @ y_vec) / y_vec.pow(2).sum()
    empirical_loss = residual.pow(2).mean()
    u = residual / torch.linalg.vector_norm(residual)
    projected_Ku = closed_form_K @ u - u * (u @ (closed_form_K @ u))
    residual_dynamics_term = -(2.0 / float(n)) * projected_Ku.pow(2).sum()

    outputs = model(X).view(-1)
    v = torch.autograd.grad((u.detach() * outputs).sum(), params, retain_graph=True)
    ntk_dynamics_sum = torch.zeros((), dtype=dtype)
    for i in range(n):
        grad_i = torch.autograd.grad(outputs[i], params, retain_graph=True, create_graph=True)
        hvp_i = torch.autograd.grad(grad_i, params, grad_outputs=[vi.detach() for vi in v], retain_graph=True)
        ntk_dynamics_sum = ntk_dynamics_sum + residual[i].detach() * _dot_tensors([vi.detach() for vi in v], hvp_i)
    ntk_dynamics_term = -(2.0 / float(n)) * ntk_dynamics_sum

    expected_scalars = {
        "ntk_eig_min": eigenvalues[0],
        "ntk_eig_max": eigenvalues[-1],
        "ntk_eig_mean": eigenvalues.mean(),
        "ntk_eig_median": _sorted_median(eigenvalues),
        "residual_ntk_alignment": alignment,
        "residual_ntk_alignment_over_ntk_eig_min": alignment / eigenvalues[0],
        "residual_ntk_alignment_over_ntk_eig_mean": alignment / eigenvalues.mean(),
        "residual_ntk_alignment_over_ntk_eig_max": alignment / eigenvalues[-1],
        "empirical_loss_times_residual_ntk_alignment": empirical_loss * alignment,
        "empirical_loss_times_ntk_eig_min": empirical_loss * eigenvalues[0],
        "residual_ntk_alignment_residual_dynamics_term": residual_dynamics_term,
        "residual_ntk_alignment_ntk_dynamics_term": ntk_dynamics_term,
        "residual_initial_ntk_alignment": alignment,
        "residual_ntk_alignment_over_initial": torch.ones((), dtype=dtype),
        "residual_ntk_alignment_trace_normalized_over_initial": torch.ones((), dtype=dtype),
        "task_ntk_alignment": task_alignment,
        "task_initial_ntk_alignment": task_alignment,
        "task_ntk_alignment_over_initial": torch.ones((), dtype=dtype),
        "task_ntk_alignment_trace_normalized_over_initial": torch.ones((), dtype=dtype),
        "ntk_cos_dist": torch.zeros((), dtype=dtype),
        "ntk_rel_fro_dist": torch.zeros((), dtype=dtype),
    }
    for name, expected in expected_scalars.items():
        actual = torch.tensor(metrics[name], dtype=dtype)
        _assert_close(name, actual, expected)

    top_eigenvectors = torch.flip(eigenvectors, dims=(1,))
    for k in (1, 3):
        name = append_k_to_ntk_label_energy_metric(k)
        projection = top_eigenvectors[:, :k].T @ y_vec
        expected = projection.pow(2).sum() / y_vec.pow(2).sum()
        actual = torch.tensor(metrics[name], dtype=dtype)
        _assert_close(name, actual, expected)

    for k in (1, 3):
        name = append_k_to_ntk_residual_energy_metric(k)
        projection = top_eigenvectors[:, :k].T @ residual
        expected = projection.pow(2).sum() / residual.pow(2).sum()
        actual = torch.tensor(metrics[name], dtype=dtype)
        _assert_close(name, actual, expected)


def _validate_training_step_initial_ntk_cache():
    dtype = torch.float32
    d_in = 5
    n = 4
    metric_names = [
        "residual_initial_ntk_alignment",
        "residual_ntk_alignment_over_initial",
        "residual_ntk_alignment_trace_normalized_over_initial",
        "task_ntk_alignment",
        "task_initial_ntk_alignment",
        "task_ntk_alignment_over_initial",
        "task_ntk_alignment_trace_normalized_over_initial",
        "ntk_cos_dist",
        "ntk_rel_fro_dist",
    ]
    config = AtInitStatsConfig(
        dataset="synthetic_isotropic",
        n_values=[n],
        m_values=[3],
        alpha_values=[1.0],
        beta_values=[float("inf")],
        training_step_values=[1],
        eta=0.01,
        init_type="standard",
        data_seeds=[0],
        init_seeds=[123],
        tracked_metrics=metric_names,
        plot_metrics=metric_names,
    )
    data = {
        "X_train": torch.randn(n, d_in, dtype=dtype),
        "y_train_binary": torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype).view(-1, 1),
        "d_in": d_in,
        "n_effective": n,
    }

    rows = _rows_for_trained_initialization(
        config,
        data,
        n=n,
        m=3,
        alpha=1.0,
        beta=float("inf"),
        data_seed=0,
        init_seed=123,
        device="cpu",
    )
    if len(rows) != 1 or int(rows[0]["training_steps"]) != 1:
        raise AssertionError("hidden step-0 NTK cache should not emit an extra CSV row")
    for name in metric_names:
        value = torch.tensor(rows[0][name])
        if not torch.isfinite(value):
            raise AssertionError(f"{name} should be finite after cached K0 measurement")


def _validate_config_metric_dependencies():
    config = AtInitStatsConfig(
        tracked_metrics=[
            "residual_ntk_alignment_trace_normalized_over_initial",
            "task_ntk_alignment_trace_normalized_over_initial",
        ],
        plot_metrics=[
            "residual_ntk_alignment_trace_normalized_over_initial",
            "task_ntk_alignment_trace_normalized_over_initial",
        ],
    )
    expected = {
        "residual_ntk_alignment",
        "residual_initial_ntk_alignment",
        "residual_ntk_alignment_trace_normalized_over_initial",
        "task_ntk_alignment",
        "task_initial_ntk_alignment",
        "task_ntk_alignment_trace_normalized_over_initial",
    }
    missing = sorted(expected - set(config.tracked_metrics or []))
    if missing:
        raise AssertionError(f"Config dependency resolution missed metrics: {missing}")


def main():
    _validate_tanh()
    _validate_training_step_initial_ntk_cache()
    _validate_config_metric_dependencies()
    print("at_init_stats spectral metric validation passed")


if __name__ == "__main__":
    main()
