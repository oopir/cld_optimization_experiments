import math
import torch
import torch.nn.functional as F

from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import langevin_step
from .linearized import (
    init_linearization,
    linearized_forward,
    get_linear_stats,
    compute_param_jacobians,
    compute_jacobian_dist,
)

def get_stats(model, params, params0, data):
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
    param_norm0 = torch.sqrt(sum(p0.pow(2).sum() for p0 in params0)).item()
    rel_param_norm = param_norm / param_norm0

    sigma_max_v = torch.linalg.svdvals(model.fc2.weight).max().item()

    return train_loss, train_acc, test_acc, param_dist, rel_param_norm, sigma_max_v

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
    device,
    seed,
    print_every,
):
    print("training starts...")

    X_train = data["X_train"]
    d = X_train.shape[1]

    model = TwoLayerNet(d_in=d, hidden=hidden_width, d_out=10).to(device)
    params, lam_tensors = make_lambda_like_params(model, lam_fc1=lam_fc1, lam_fc2=lam_fc2)

    params0 = [p.detach().clone() for p in params]
    W0 = model.fc1.weight.detach().clone()
    model_init = TwoLayerNet(d_in=d, hidden=hidden_width, d_out=10).to(device)
    model_init.load_state_dict(model.state_dict())
    if use_linearized:
        base_params_dict, lin_params, lin_lam_tensors = init_linearization(model, params0, lam_tensors)
        lin_params0 = [p.detach().clone() for p in lin_params]
    if track_jacobian:
        probe_bs = min(1, X_train.shape[0])
        X_probe = X_train[:probe_bs].to(device)
        jac_init = compute_param_jacobians(model, X_probe)
        jac_init_norm_sq = sum(float(ji.pow(2).sum().item()) for ji in jac_init)

    metrics = {
        "train_loss_hist": [],
        "train_acc_hist": [],
        "test_acc_hist": [],
        "param_dist_hist": [],
        "param_norm_hist": [],
        "jacobian_dist_hist": [],
        "lin_train_loss_hist": [],
        "lin_train_acc_hist": [],
        "lin_test_acc_hist": [],
        "lin_param_dist_hist": [],
        "lin_param_norm_hist": [],
    }

    langevin_gen = torch.Generator(device=device)
    langevin_gen.manual_seed(seed)

    train_loss, train_acc, test_acc, _, _, sup_sigma_max_v = get_stats(model, params, params0, data)
    if use_linearized:
        lin_train_loss, lin_train_acc, lin_test_acc, _, _ = get_linear_stats(model, base_params_dict, lin_params, lin_params0, data)
    print(f"epoch {0:4d} | loss {train_loss:.4f} | train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    for epoch in range(1, epochs + 1):
        # ------------------- compute grads & perform step ------------------ #
        model.train()
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(X_train)
        train_loss = loss_fn(outputs, data["y_train_one_hot"])
        train_loss.backward()
        langevin_step(params, lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale, gen=langevin_gen)

        if use_linearized:
            for p in lin_params:
                if p.grad is not None:
                    p.grad.zero_()
            lin_outputs = linearized_forward(model, base_params_dict, lin_params, X_train)
            lin_train_loss = loss_fn(lin_outputs, data["y_train_one_hot"])
            lin_train_loss.backward()
            langevin_step(lin_params, lin_lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale, gen=langevin_gen)

        # -------------------- compute metrics and stats -------------------- #
        model.eval()
        train_loss, train_acc, test_acc, param_dist, param_norm, sigma_max_v = get_stats(model, params, params0, data)
        metrics["train_loss_hist"].append(train_loss)
        metrics["train_acc_hist"].append(train_acc)
        metrics["test_acc_hist"].append(test_acc)
        metrics["param_dist_hist"].append(param_dist)
        metrics["param_norm_hist"].append(param_norm)
        sup_sigma_max_v = max(sup_sigma_max_v, sigma_max_v)

        # this part should *not* be inside "no_grad" blocks/functions
        if track_jacobian:
            jacobian_dist = compute_jacobian_dist(model, X_probe, jac_init, jac_init_norm_sq)
            metrics["jacobian_dist_hist"].append(jacobian_dist)
            # jacobian_dist_full = compute_dataset_ntk_drift(model, model_init, X_train[:10], batch_size=1)
            # metrics["jacobian_dist_hist"].append(jacobian_dist_full)

        if use_linearized:
            lin_train_loss, lin_train_acc, lin_test_acc, lin_param_dist, lin_param_norm = get_linear_stats(model, base_params_dict, lin_params, lin_params0, data)
            metrics["lin_train_loss_hist"].append(lin_train_loss)
            metrics["lin_train_acc_hist"].append(lin_train_acc)
            metrics["lin_test_acc_hist"].append(lin_test_acc)
            metrics["lin_param_dist_hist"].append(lin_param_dist)
            metrics["lin_param_norm_hist"].append(lin_param_norm)

        if epoch % print_every == 0:
            print(f"epoch {epoch:4d} | loss {train_loss:.4f} | train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    # -------------------- compute remaining stats --------------------- #
    with torch.no_grad():
        sigma_max_X = torch.linalg.svdvals(X_train).max().item()
        H0 = F.tanh(X_train @ W0.T)
        sigma_min_phi_W0X = torch.linalg.svdvals(H0).min().item()

        param_dist_upper_bound = sigma_min_phi_W0X / (2 * math.sqrt(2) * sigma_max_X * sup_sigma_max_v)

    metrics["param_dist_upper_bound"] = param_dist_upper_bound
    return metrics
