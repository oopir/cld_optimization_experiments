import torch
import math
from .model import loss_fn

def init_linearization(model, params0, lam_tensors):
    base_params_dict = {name: p0.detach().clone() for (name, _), p0 in zip(model.named_parameters(), params0)}
    lin_params = [torch.nn.Parameter(p0.detach().clone()) for p0 in params0]
    lin_lam_tensors = [lam.detach().clone() for lam in lam_tensors]
    return base_params_dict, lin_params, lin_lam_tensors


def linearized_forward(model, base_params_dict, lin_params, X):
    lin_params_dict = {name: p for (name, _), p in zip(model.named_parameters(), lin_params)}
    delta_params = {key: lin_params_dict[key] - base_params_dict[key] for key in base_params_dict}

    def f(params, x):
        return functional_call(model, params, (x,))

    f0, jvp_out = jvp(f, (base_params_dict, X), (delta_params, torch.zeros_like(X)),)
    return f0 + jvp_out

@torch.no_grad()
def get_linear_stats(model, base_params_dict, lin_params, lin_params0, data):
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

    return train_loss, train_acc, test_acc, param_dist, param_norm, fc1_norm, fc2_norm

from torch.autograd.functional import jacobian as _jacobian
from torch.func import functional_call, jvp

def compute_param_jacobians(model, X):
    """
    Returns list of tensors [J_i] where J_i has shape (out_flat_dim, param_numel)
    for each model parameter. Uses functional_call and returns detached tensors.
    """
    # create a function that runs the model with a specific set of parameters
    param_names = [n for n, _ in model.named_parameters()]
    def flat_out(*params):
        # reconstruct parameter dict from input, runs the model with these
        # parameters, flattens the output and returns it
        param_map = {name: t for name, t in zip(param_names, params)}
        out = functional_call(model, param_map, (X,))
        return out.view(-1)

    # create a tuple of differentiable clones of the model's parameters
    # (requires_grad_(True) ensures the gradients are computed w.r.t. the clones)
    params_tuple = tuple(p.detach().clone().requires_grad_(True) for _, p in model.named_parameters())

    jac = _jacobian(flat_out, params_tuple, vectorize=False)
    jac_flat = [j.detach().reshape(j.shape[0], -1) for j in jac] # tuple of (out_flat_dim, #weights)
    return jac_flat

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

def compute_dataset_ntk_drift(model, model_init, X_data, batch_size=1):
    device = next(model.parameters()).device
    total_sq = 0.0
    n = X_data.shape[0]

    for start in range(0, n, batch_size):
        X_batch = X_data[start:start + batch_size].to(device)

        # Jacobian at current parameters θ
        jac_curr = compute_param_jacobians(model, X_batch)
        # Jacobian at initialization θ₀
        jac_init = compute_param_jacobians(model_init, X_batch)

        for jc, ji in zip(jac_curr, jac_init):
            diff = jc - ji
            total_sq += float(diff.pow(2).sum().item())

        del jac_curr, jac_init  # free per-batch Jacobians

    return math.sqrt(total_sq)