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
    "param_norm",
    "param_norm_fc1",
    "param_norm_fc2",
]
LIN_METRIC_NAMES = [
    "lin_train_loss",
    "lin_train_acc",
    "lin_test_acc",
    "lin_param_dist",
    "lin_param_norm",
    "lin_param_norm_fc1",
    "lin_param_norm_fc2",
]

@torch.no_grad()
def get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, data):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    y_train_one_hot = data["y_train_one_hot"]

    train_outputs = model(X_train)
    pred_train = train_outputs.argmax(dim=1)
    train_acc = (pred_train == y_train).float().mean().item()

    test_outputs = model(X_test)
    pred_test = test_outputs.argmax(dim=1)
    test_acc = (pred_test == y_test).float().mean().item()

    train_loss = loss_fn(train_outputs, y_train_one_hot).item()

    param_dist = torch.sqrt(sum((p-p0).pow(2).sum() for p, p0 in zip(params, params0))).item()
    param_norm = torch.sqrt(sum(p.pow(2).sum() for p in params)).item()
    fc1_norm = torch.sqrt(params[0].pow(2).sum()).item()
    fc2_norm = torch.sqrt(params[1].pow(2).sum()).item()

    sigma_max_v = torch.linalg.svdvals(model.fc2.weight).max().item()

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "param_dist": param_dist,
        "param_norm": param_norm / (param_norm0 + 1e-12),
        "param_norm_fc1": fc1_norm / (fc1_norm0 + 1e-12),
        "param_norm_fc2": fc2_norm / (fc2_norm0 + 1e-12),
        "sigma_max_v": sigma_max_v
    }

@torch.no_grad()
def get_linear_stats(model, base_params_dict, lin_params, lin_params0, param_norm0, fc1_norm0, fc2_norm0, data):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    y_train_one_hot = data["y_train_one_hot"]

    outputs_train = linearized_forward(model, base_params_dict, lin_params, X_train)
    pred_train = outputs_train.argmax(dim=1)
    train_acc = (pred_train == y_train).float().mean().item()

    outputs_test = linearized_forward(model, base_params_dict, lin_params, X_test)
    pred_test = outputs_test.argmax(dim=1)
    test_acc = (pred_test == y_test).float().mean().item()

    train_loss = loss_fn(outputs_train, y_train_one_hot).item()

    param_dist = torch.sqrt(sum((p-p0).pow(2).sum() for p, p0 in zip(lin_params, lin_params0))).item()
    param_norm = torch.sqrt(sum((p).pow(2).sum() for p in lin_params)).item()
    fc1_norm = torch.sqrt(lin_params[0].pow(2).sum()).item()
    fc2_norm = torch.sqrt(lin_params[1].pow(2).sum()).item()

    return {
        "lin_train_loss": train_loss,
        "lin_train_acc": train_acc,
        "lin_test_acc": test_acc,
        "lin_param_dist": param_dist,
        "lin_param_norm": param_norm / (param_norm0 + 1e-12),
        "lin_param_norm_fc1": fc1_norm / (fc1_norm0 + 1e-12),
        "lin_param_norm_fc2": fc2_norm / (fc2_norm0 + 1e-12),
    }

# this part should *not* be inside "no_grad" blocks/functions
def compute_jacobian_dist(model, X_probe, jac_init, jac_init_norm_sq=None, eps=1e-12):
    jac_curr = compute_param_jacobians(model, X_probe)
    total_sq = 0.
    dot = 0.0
    norm_c_sq = 0.0
    norm_i_sq = 0.0 if jac_init_norm_sq is None else float(jac_init_norm_sq)

    for jc, ji in zip(jac_curr, jac_init):
        diff = jc - ji
        total_sq += float(diff.pow(2).sum().item())

        dot += float((jc * ji).sum().item())
        norm_c_sq += float((jc * jc).sum().item())
        if jac_init_norm_sq is None:
            norm_i_sq += float((ji * ji).sum().item())
    del jac_curr

    l2_dist  = math.sqrt(total_sq)
    cos_dist = 1.0 - dot / ((math.sqrt(norm_c_sq) * math.sqrt(norm_i_sq)) + eps)

    return l2_dist, cos_dist

# this part should *not* be inside "no_grad" blocks/functions
def compute_dataset_ntk_drift(model, model_init, X_data, batch_size=1, eps=1e-12):
    device = next(model.parameters()).device
    total_sq = 0.0
    dot = 0.0
    norm_c_sq = 0.0
    norm_i_sq = 0.0

    n = X_data.shape[0]
    for start in range(0, n, batch_size):
        X_batch = X_data[start:start + batch_size].to(device)

        jac_curr = compute_param_jacobians(model, X_batch)
        jac_init = compute_param_jacobians(model_init, X_batch)

        for jc, ji in zip(jac_curr, jac_init):
            diff = jc - ji
            total_sq += float(diff.pow(2).sum().item())

            dot += float((jc * ji).sum().item())
            norm_c_sq += float((jc * jc).sum().item())
            norm_i_sq += float((ji * ji).sum().item())

        del jac_curr, jac_init  # free per-batch Jacobians

    l2_dist  = math.sqrt(total_sq)
    cos_dist = 1.0 - dot / ((math.sqrt(norm_c_sq) * math.sqrt(norm_i_sq)) + eps)

    return l2_dist, cos_dist

@torch.no_grad()
def compute_dist_bound_under_GF(X_train, W0, sup_sigma_max_v):
    # compute Song's theoretical upper bound on the distance from the init
    sigma_max_X = torch.linalg.svdvals(X_train).max().item()
    H0 = F.tanh(X_train @ W0.T)
    sigma_min_phi_W0X = torch.linalg.svdvals(H0).min().item()
    param_dist_upper_bound = sigma_min_phi_W0X / (2 * math.sqrt(2) * sigma_max_X * sup_sigma_max_v)
    return param_dist_upper_bound

@torch.no_grad()
def estimate_lambda_min(X, M=10000, batch_g=64, device=None):
    # estimate λ_min(E[φ(𝐗𝐠) φ(𝐗𝐠)]) by sampling 𝐠 for M times and then averaging
    n, d = X.shape
    A = torch.zeros((n, n), device=device, dtype=X.dtype)
    done = 0
    while done < M:
        b = min(batch_g, M - done)
        G = torch.randn(d, b, device=device, dtype=X.dtype) # (d,b)
        Y = torch.tanh(X @ G)                               # (n,b)
        A += (Y @ Y.T)                                      # sum_k y_k y_k^T over batch
        done += b

    A /= M
    A = (A + A.T) * 0.5                                     # symmetrize for numerical safety
    lam_min = torch.linalg.eigvalsh(A)[0].item()
    return lam_min

def estimate_loss_floor(X_train, noisy_beta, m, device):
    # compute L_∞ from Matan's analysis
    n,d        = X_train.shape
    lambda_min = estimate_lambda_min(X_train, device=device)
    loss_floor = (2/lambda_min) * ((n/noisy_beta) * (1 + d/m) + (n/noisy_beta)**2 * (1 + d*d/m))
    return loss_floor

# for name, metrics_per_seed in results.items():
#     print(f"loss floor estimation for run {name}: {metrics_per_seed[0]["loss_floor"]}")