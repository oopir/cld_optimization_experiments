import math
import numpy as np
import random
import torch
import torch.nn.functional as F

from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import langevin_step
from .linearized import (
    init_linearization,
    linearized_forward,
    compute_param_jacobians
)
from .stats import (
    BASE_METRIC_NAMES,
    LIN_METRIC_NAMES,
    get_stats,
    get_linear_stats,
    compute_jacobian_dist,
)

def _init_base_model_vars(d, hidden_width, device, lam_fc1, lam_fc2):

    model = TwoLayerNet(d_in=d, hidden=hidden_width, d_out=10).to(device)
    params, lam_tensors = make_lambda_like_params(model, lam_fc1=lam_fc1, lam_fc2=lam_fc2)

    params0 = [p.detach().clone() for p in params]
    with torch.no_grad():
        param_norm0 = torch.sqrt(sum(p.pow(2).sum() for p in params0)).item()
        fc1_norm0 = torch.sqrt(params0[0].pow(2).sum()).item()
        fc2_norm0 = torch.sqrt(params0[1].pow(2).sum()).item()

    W0 = model.fc1.weight.detach().clone()

    return model, params, lam_tensors, params0, param_norm0, fc1_norm0, fc2_norm0, W0

def _init_linearization_vars(model, params0, lam_tensors):

    base_params_dict, lin_params, lin_lam_tensors = init_linearization(model, params0, lam_tensors)
    lin_params0 = [p.detach().clone() for p in lin_params]
    with torch.no_grad():
        lin_param_norm0 = torch.sqrt(sum(p.pow(2).sum() for p in lin_params0)).item()
        lin_fc1_norm0 = torch.sqrt(lin_params0[0].pow(2).sum()).item()
        lin_fc2_norm0 = torch.sqrt(lin_params0[1].pow(2).sum()).item()

    return (base_params_dict, lin_params, lin_lam_tensors, lin_params0, lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0)

def _init_jacobian_track_vars(d, hidden_width, device, model, X_train, probe_bs):
    # model_at_init is made in case we will want to track Jacobian
    # drift w.r.t. full Jacobian and not just a partial probe
    model_at_init = TwoLayerNet(d_in=d, hidden=hidden_width, d_out=10).to(device)
    model_at_init.load_state_dict(model.state_dict())

    X_probe = X_train[:probe_bs].to(device)
    jac_init = compute_param_jacobians(model, X_probe)
    jac_init_norm_sq = sum(float(ji.pow(2).sum().item()) for ji in jac_init)

    return model_at_init, X_probe, jac_init, jac_init_norm_sq

def _init_metrics(track_jacobian, use_linearized):
    metrics = {f"{name}_hist": [] for name in BASE_METRIC_NAMES}
    if track_jacobian:
        metrics["jacobian_dist_hist"] = []
    # if use_linearized:
    for name in LIN_METRIC_NAMES:
        metrics[f"{name}_hist"] = []
    return metrics

def train(
    data,
    eta,
    epochs,
    beta,
    lam_fc1,
    lam_fc2,
    hidden_width,
    regularization_scale,
    use_linearized,
    track_jacobian,
    jac_probe_size,
    device,
    track_every,
    print_every,
):

    # --------- init environment & compute values at init for stats -------- #
    X_train = data["X_train"]
    d = X_train.shape[1]

    model, params, lam_tensors, params0, param_norm0, fc1_norm0, fc2_norm0, W0 = \
        _init_base_model_vars(d, hidden_width, device, lam_fc1, lam_fc2)

    if use_linearized:
        (
            base_params_dict, lin_params, lin_lam_tensors, lin_params0, 
            lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0
        ) = _init_linearization_vars(model, params0, lam_tensors)

    if track_jacobian:
        _, X_probe, jac_init, jac_init_norm_sq = \
            _init_jacobian_track_vars(d, hidden_width, device, model, X_train, jac_probe_size)

    metrics = _init_metrics(track_jacobian, use_linearized)

    print("training starts...")
    stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, data)
    sup_sigma_max_v = stats["sigma_max_v"]
    print(f"epoch {0:4d} | loss {stats['train_loss']:.4f} | train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}")

    for epoch in range(1, epochs + 1):
        # ------------------ compute grads & perform steps ------------------ #
        model.train()
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(X_train)
        train_loss = loss_fn(outputs, data["y_train_one_hot"])
        train_loss.backward()
        langevin_step(params, lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale)

        if use_linearized:
            for p in lin_params:
                if p.grad is not None:
                    p.grad.zero_()
            lin_outputs = linearized_forward(model, base_params_dict, lin_params, X_train)
            lin_train_loss = loss_fn(lin_outputs, data["y_train_one_hot"])
            lin_train_loss.backward()
            langevin_step(lin_params, lin_lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale)

        # -------------------- compute metrics and stats -------------------- #
        model.eval()
        if epoch % track_every == 0:
            stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, data)
            for name in BASE_METRIC_NAMES:
                metrics[f"{name}_hist"].append(stats[name])
            sup_sigma_max_v = max(sup_sigma_max_v, stats["sigma_max_v"])

            # this part should *not* be inside "no_grad" blocks/functions
            if track_jacobian:
                jacobian_dist = compute_jacobian_dist(model, X_probe, jac_init, jac_init_norm_sq)
                metrics["jacobian_dist_hist"].append(jacobian_dist)

            if use_linearized:
                lin_stats = get_linear_stats(model, base_params_dict, lin_params, lin_params0, lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0, data)
                for name in LIN_METRIC_NAMES:
                    metrics[f"{name}_hist"].append(lin_stats[name])

            if epoch % print_every == 0:
                print(f"epoch {epoch:4d} | loss {stats['train_loss']:.4f} | train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}")

    # -------------------- compute remaining stats --------------------- #
    # compute Song's theoretical upper bound on the distance from the init
    with torch.no_grad():
        sigma_max_X = torch.linalg.svdvals(X_train).max().item()
        H0 = F.tanh(X_train @ W0.T)
        sigma_min_phi_W0X = torch.linalg.svdvals(H0).min().item()
        param_dist_upper_bound = sigma_min_phi_W0X / (2 * math.sqrt(2) * sigma_max_X * sup_sigma_max_v)
    metrics["param_dist_upper_bound"] = param_dist_upper_bound

    return metrics

def train_multiseed(
    seeds,
    data,
    eta,
    epochs,
    beta,
    lam_fc1,
    lam_fc2,
    hidden_width,
    regularization_scale,
    use_linearized,
    track_jacobian,
    jac_probe_size,
    device,
    track_every,
    print_every,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    results = {}

    for run_seed in seeds:
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        metrics = train(
            data=data,
            eta=eta,
            epochs=epochs,
            beta=beta,
            lam_fc1=lam_fc1,
            lam_fc2=lam_fc2,
            hidden_width=hidden_width,
            regularization_scale=regularization_scale,
            use_linearized=use_linearized,
            track_jacobian=track_jacobian,
            jac_probe_sizejac_probe_size,
            device=device,
            track_every=track_every,
            print_every=print_every,
        )
        results[run_seed] = metrics

    return results
