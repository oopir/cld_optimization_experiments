import torch
from torch.autograd.functional import jacobian as _jacobian
from torch.func import functional_call, jvp

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