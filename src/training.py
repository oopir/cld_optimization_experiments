import numpy as np
import random
import torch

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
    compute_dist_bound_under_GF,
    estimate_loss_floor
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
    metrics["NN_to_lin_hist"] = []
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
    print(f"epoch {0:5d} | loss {stats['train_loss']:.4f} | train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}")

    for epoch in range(1, epochs + 1):
        # -------------------- compute metrics and stats -------------------- #
        model.eval()
        if epoch % track_every == 1:
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
                NN_to_lin_dist = torch.sqrt(sum((p-q).pow(2).sum() for p, q in zip(params, lin_params))).item()
                metrics["NN_to_lin_hist"].append(NN_to_lin_dist)

            if epoch % print_every == 0:
                print(
                    f"epoch {epoch:5d} | "
                    f"loss {stats['train_loss']:.4f} (lin: {lin_stats['lin_train_loss']:.4f}) | "
                    f"train acc {stats['train_acc']:.3f} (lin: {lin_stats['lin_train_acc']:.4f}) | "
                    f"test acc {stats['test_acc']:.3f} (lin: {lin_stats['lin_test_acc']:.4f})"
                )

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

        

    # -------------------- compute remaining stats --------------------- #
    metrics["param_dist_upper_bound"] = compute_dist_bound_under_GF(X_train, W0, sup_sigma_max_v)
    metrics["loss_floor"] = estimate_loss_floor(X_train, beta, m=hidden_width, device=device)

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
            jac_probe_size=jac_probe_size,
            device=device,
            track_every=track_every,
            print_every=print_every,
        )
        results[run_seed] = metrics

    return results
