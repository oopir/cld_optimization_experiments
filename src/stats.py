import math
import torch
import torch.nn.functional as F

from .model import loss_fn
from .linearized import linearized_forward, compute_param_jacobians

BASE_METRIC_NAMES = [
    "train_loss",
    "train_acc",
    "test_acc",
    "param_dist",
    "feat_rel_dist",
    "feat_cos_dist",
    "feat_gram_lambda",
]
LIN_METRIC_NAMES = [
    "lin_train_loss",
    "lin_train_acc",
    "lin_test_acc",
    "lin_param_dist",
]

def _param_dist(params, params0):
    """Compute parameter displacement from the matching initialization tensors."""
    return torch.sqrt(sum((p - p0).pow(2).sum() for p, p0 in zip(params, params0))).item()

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

def _feature_stats(model, X_train, A0, A0_norm):
    """Compute hidden-feature drift metrics used by the experiment plots and bounds."""
    A_t = torch.tanh(model.fc1(X_train))
    feat_rel_dist = (A_t - A0).norm().item() / (A0_norm + 1e-12)

    cos_sim = F.cosine_similarity(A_t.view(-1), A0.view(-1), dim=0).item()
    feat_cos_dist = 1.0 - cos_sim

    A_Gram = A_t @ A_t.T
    A_Gram = 0.5 * (A_Gram + A_Gram.T)
    try:
        feat_gram_lambda = torch.linalg.eigvalsh(A_Gram)[0].item()
    except Exception:
        print("Numerical instability occurred when computing lambda_min of feature matrix. Defaulting to nan.")
        feat_gram_lambda = float("nan")

    return feat_rel_dist, feat_cos_dist, feat_gram_lambda

@torch.no_grad()
def get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, A0, A0_norm, data, collect_feature_stats):
    """Compute base neural-network metrics while preserving existing metric keys."""
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # MSE classification runs use one-hot targets; regression-style runs can omit them.
    train_targets = data.get("y_train_one_hot", y_train)
    train_acc, train_loss = _classification_metrics(model, X_train, y_train, train_targets)
    test_acc, _ = _classification_metrics(model, X_test, y_test)

    param_dist = _param_dist(params, params0)
    
    if collect_feature_stats:
        feat_rel_dist, feat_cos_dist, feat_gram_lambda = _feature_stats(model, X_train, A0, A0_norm)
    else:
        feat_rel_dist = float("nan")
        feat_cos_dist = float("nan")
        feat_gram_lambda = float("nan")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "param_dist": param_dist,
        "feat_rel_dist": feat_rel_dist,
        "feat_cos_dist": feat_cos_dist,
        "feat_gram_lambda": feat_gram_lambda,
    }

@torch.no_grad()
def get_linear_stats(model, base_params_dict, lin_params, lin_params0, param_norm0, fc1_norm0, fc2_norm0, data):
    """Compute metrics for the linearized model with the same public key names."""
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    outputs_train = linearized_forward(model, base_params_dict, lin_params, X_train)
    pred_train = outputs_train.argmax(dim=1)
    train_acc = (pred_train == y_train).float().mean().item()

    outputs_test = linearized_forward(model, base_params_dict, lin_params, X_test)
    pred_test = outputs_test.argmax(dim=1)
    test_acc = (pred_test == y_test).float().mean().item()

    # Same target convention as get_stats: prefer one-hot labels when present.
    train_loss = loss_fn(outputs_train, data.get("y_train_one_hot", y_train)).item()

    param_dist = _param_dist(lin_params, lin_params0)
    
    return {
        "lin_train_loss": train_loss,
        "lin_train_acc": train_acc,
        "lin_test_acc": test_acc,
        "lin_param_dist": param_dist,
    }

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
