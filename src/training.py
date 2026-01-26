import numpy as np
import random
import torch

from .data import load_digits_data, load_1d_regression_data
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
    get_nn_lin_param_dist,
    compute_jacobian_dist,
    compute_dataset_ntk_drift,
    compute_dist_bound_under_GF,
    estimate_loss_floor
)

def _init_base_model_vars(d_in, d_out, m, init_type, device, lam_fc1, lam_fc2):

    model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=init_type).to(device)
    params, lam_tensors = make_lambda_like_params(model, init_type, lam_fc1=lam_fc1, lam_fc2=lam_fc2)

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

def _init_jacobian_track_vars(d, d_out, m, init_type, device, model, X_train, probe_bs):
    # model_at_init is made in case we will want to track Jacobian
    # drift w.r.t. full Jacobian and not just a partial probe
    model_at_init = TwoLayerNet(d_in=d, m=m, d_out=d_out, init_type=init_type).to(device)
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
    metrics["nn_to_lin_hist"] = []
    metrics["nn_lin_param_dist_hist"] = []
    return metrics

def train(
    data,
    eta,
    epochs,
    beta,
    m,
    init_type="standard",
    lam_fc1=None,
    lam_fc2=None,
    regularization_scale=1.0,
    use_linearized=True,
    track_jacobian=True,
    jac_probe_size=1,
    device="cpu",
    track_every=1,
    print_every=100,
):

    # --------- init environment & compute values at init for stats -------- #
    X_train = data["X_train"]
    d = X_train.shape[1]

    model, params, lam_tensors, params0, param_norm0, fc1_norm0, fc2_norm0, W0 = \
        _init_base_model_vars(data["d_in"], data["d_out"], m, init_type, device, lam_fc1, lam_fc2)

    if use_linearized:
        (
            base_params_dict, lin_params, lin_lam_tensors, lin_params0, 
            lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0
        ) = _init_linearization_vars(model, params0, lam_tensors)

    if track_jacobian:
        model_at_init, X_probe, jac_init, jac_init_norm_sq = \
            _init_jacobian_track_vars(data["d_in"], data["d_out"], m, init_type, device, model, X_train, jac_probe_size)

    with torch.no_grad():
        if model.act == 'relu':
            A0 = torch.relu(X_train @ model.fc1.weight.T)
        elif model.act == 'tanh':
            A0 = torch.tanh(X_train @ model.fc1.weight.T)
        else:
            raise ValueError(f"Tracking of feature distance does not support activation '{model.act}'.")
        A0_norm = A0.norm().item()

    metrics = _init_metrics(track_jacobian, use_linearized)

    print("training starts...")
    stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, A0, A0_norm, data)
    sup_sigma_max_v = stats["sigma_max_v"]
    # print(f"epoch {0:8d} | loss {stats['train_loss']:.4f} | train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}")

    for epoch in range(1, epochs + 1):
        # -------------------- compute metrics and stats -------------------- #
        model.eval()
        if epoch % track_every == 1:
            stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, A0, A0_norm, data)
            for name in BASE_METRIC_NAMES:
                metrics[f"{name}_hist"].append(stats[name])
            sup_sigma_max_v = max(sup_sigma_max_v, stats["sigma_max_v"])

            # this part should *not* be inside "no_grad" blocks/functions
            if track_jacobian:
                # jacobian_dist = compute_jacobian_dist(model, X_probe, jac_init, jac_init_norm_sq)
                jacobian_dist = compute_dataset_ntk_drift(model, model_at_init, X_train, batch_size=jac_probe_size)
                metrics["jacobian_dist_hist"].append(jacobian_dist)

            if use_linearized:
                lin_stats = get_linear_stats(model, base_params_dict, lin_params, lin_params0, lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0, data)
                
                for name in LIN_METRIC_NAMES:
                    metrics[f"{name}_hist"].append(lin_stats[name])
                nn_to_lin_dist = torch.sqrt(sum((p-q).pow(2).sum() for p, q in zip(params, lin_params))).item()
                metrics["nn_to_lin_hist"].append(nn_to_lin_dist)
                
                nn_lin_param_dist = get_nn_lin_param_dist(params, lin_params)
                metrics["nn_lin_param_dist_hist"].append(nn_lin_param_dist)

            if epoch % print_every == 1:
                print(
                    f"epoch {epoch:8d} | "
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
        if "y_train_one_hot" in data:
            train_loss = loss_fn(outputs, data["y_train_one_hot"])
        else:
            train_loss = loss_fn(outputs, data["y_train"])
        train_loss.backward()
        langevin_step(params, lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale)

        if use_linearized:
            for p in lin_params:
                if p.grad is not None:
                    p.grad.zero_()
            lin_outputs = linearized_forward(model, base_params_dict, lin_params, X_train)
            if "y_train_one_hot" in data:
                lin_train_loss = loss_fn(lin_outputs, data["y_train_one_hot"])
            else:
                lin_train_loss = loss_fn(lin_outputs, data["y_train"])
            lin_train_loss.backward()
            langevin_step(lin_params, lin_lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale)

        

    # -------------------- compute remaining stats --------------------- #
    metrics["param_dist_upper_bound"] = compute_dist_bound_under_GF(X_train, W0, sup_sigma_max_v)
    metrics["loss_floor"] = estimate_loss_floor(X_train, beta, m=m, device=device)

    return metrics

def train_multiseed(
    seeds,
    n,
    random_labels,
    eta,
    epochs,
    beta,
    m,
    init_type="standard",
    lam_fc1=None,
    lam_fc2=None,
    regularization_scale=1.0,
    use_linearized=True,
    track_jacobian=True,
    jac_probe_size=1,
    device="cpu",
    track_every=1,
    print_every=100,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    results = {}

    for run_seed in seeds:
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        data = load_digits_data(n=n, random_labels=random_labels, device=device, seed=run_seed)

        metrics = train(
            data=data,
            eta=eta,
            epochs=epochs,
            beta=beta,
            m=m,
            init_type=init_type,
            lam_fc1=lam_fc1,
            lam_fc2=lam_fc2,
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


def train_and_return_model(
    seed,
    data,
    eta,
    epochs,
    beta,
    m,
    init_type="standard",
    alpha=0.1,
    lam_fc1=None,
    lam_fc2=None,
    regularization_scale=1.0,
    device="cpu",
    print_every=100,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_train = data["X_train"]
    y_train = data["y_train"].unsqueeze(1)  # (N, 1)
    d_in = data["d_in"]
    d_out = data["d_out"]

    model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, with_bias=True, init_type=init_type, alpha=alpha, act="relu").to(device)
    params, lam_tensors = make_lambda_like_params(model, init_type, lam_fc1, lam_fc2)

    for epoch in range(epochs):
        model.train()
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        langevin_step(params,lam_tensors,beta=beta,eta=eta,regularization_scale=regularization_scale)
        if epoch % print_every == 0:
            print(f"    epoch = {epoch:5} | loss = {loss:.2}")

    return model

def get_1d_regression_curves_for_betas(
    x_plot,
    seeds, 
    eta, 
    epochs, 
    betas, 
    init_type="standard",
    regularization_scale=1.0, 
    device="cpu", 
    print_every=100
):
    curves = {}
    m_values = [int(min(1e05, beta * np.log(beta))) for beta in betas]
    m_max = max(m_values)
    for beta in betas:
        print(f"beta={beta:.0e}, m={m_max:.2e}")
        fs = []
        for seed in seeds:
            print(f"  seed={seed}")
            data = load_1d_regression_data(device=device)

            model = train_and_return_model(
                seed=seed, 
                data=data, 
                eta=eta, 
                epochs=epochs, 
                beta=beta, 
                m=m_max,
                init_type=init_type,
                regularization_scale=regularization_scale, 
                device=device, 
                print_every=print_every
            )
            with torch.no_grad():
                f = model(x_plot).cpu().numpy().ravel()
            fs.append(f)
        curves[beta] = np.stack(fs, axis=0)  # (n_seeds, n_grid)
    return curves

def get_1d_regression_curves_for_alphas(
    x_plot,
    seeds,
    eta,
    epochs,
    alphas,
    beta,
    init_type="alpha",
    regularization_scale=0.0,
    device="cpu",
    print_every=100,
):
    curves = {}
    m = int(min(1e05, beta * np.log(beta)))  # same width rule as before
    for alpha in alphas:
        print(f"alpha={alpha:.1e}, m={m:.2e}")
        eta_alpha = min(1e-2, eta / alpha)
        print(f"eta for {alpha} is {eta_alpha:.2e}")
        fs = []
        for seed in seeds:
            print(f"  seed={seed}")
            data = load_1d_regression_data(device=device)

            # same seed + init_type="alpha" ⇒ same base w0; alpha rescales it
            model = train_and_return_model(
                seed=seed,
                data=data,
                eta=eta_alpha,
                epochs=epochs,
                beta=beta,
                m=m,
                init_type=init_type,
                regularization_scale=regularization_scale,
                device=device,
                print_every=print_every,
                alpha=alpha,
            )
            with torch.no_grad():
                f = model(x_plot).cpu().numpy().ravel()
            fs.append(f)
        curves[alpha] = np.stack(fs, axis=0)
    return curves
