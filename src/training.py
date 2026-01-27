import numpy as np
import random
import torch

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .data import load_digits_data, load_mnist_data, load_1d_regression_data
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

def _init_base_model_vars(d_in, d_out, m, init_type, device, lam_fc1, lam_fc2, init_model_state_dict=None):

    model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=init_type).to(device)
    if init_model_state_dict is not None:
        model.load_state_dict(init_model_state_dict)
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
    init_model_state_dict=None, 
    start_model_state_dict=None,
    start_lin_params=None,
    add_noise=True,
):

    # --------- init environment & compute values at init for stats -------- #
    X_train = data["X_train"]
    d = X_train.shape[1]

    model, params, lam_tensors, params0, param_norm0, fc1_norm0, fc2_norm0, W0 = \
        _init_base_model_vars(data["d_in"], data["d_out"], m, init_type, device, lam_fc1, lam_fc2, init_model_state_dict)

    if init_model_state_dict is not None:
        init_state_for_metrics = {k: v.detach().cpu() for k, v in init_model_state_dict.items()}
    else:
        init_state_for_metrics = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if use_linearized:
        (
            base_params_dict, lin_params, lin_lam_tensors, lin_params0, 
            lin_param_norm0, lin_fc1_norm0, lin_fc2_norm0
        ) = _init_linearization_vars(model, params0, lam_tensors)
        if start_lin_params is not None:
            for p, p_prev in zip(lin_params, start_lin_params):
                p.data.copy_(p_prev.to(device=p.device, dtype=p.dtype))

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

    if start_model_state_dict is not None:
        model.load_state_dict(start_model_state_dict)

    print(f"training starts for {device}...")
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
                    f"device {device} | "
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
        langevin_step(params, lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale, add_noise=add_noise)

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
            langevin_step(lin_params, lin_lam_tensors, beta=beta, eta=eta, regularization_scale=regularization_scale, add_noise=add_noise)

        

    # -------------------- compute remaining stats --------------------- #
    metrics["param_dist_upper_bound"] = compute_dist_bound_under_GF(X_train, W0, sup_sigma_max_v)
    metrics["loss_floor"] = estimate_loss_floor(X_train, beta, m=m, device=device)

    metrics["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metrics["init_model_state_dict"] = init_state_for_metrics
    if use_linearized:
        metrics["lin_params_state"] = [p.detach().cpu() for p in lin_params]


    return metrics

def _train_multiseed_worker(
    dataset,
    run_seed,
    device,
    n,
    random_labels,
    eta,
    epochs,
    beta,
    m,
    init_type,
    lam_fc1,
    lam_fc2,
    regularization_scale,
    use_linearized,
    track_jacobian,
    jac_probe_size,
    track_every,
    print_every,
    init_model_state_dicts=None, 
    start_model_state_dicts=None,
    start_lin_params_dicts=None,
    add_noise=True,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    if dataset == "digits":
        data = load_digits_data(n=n, random_labels=random_labels, device=device, seed=run_seed)
    else:
        data = load_1d_regression_data(device=device)
    if dataset == "digits":
        data = load_digits_data(n=n, random_labels=random_labels, device=device, seed=run_seed)
    elif dataset == "mnist":
        data = load_mnist_data(n=n, random_labels=random_labels, device=device, seed=run_seed)
    else:
        data = load_1d_regression_data(n=n, device=device, seed=run_seed)

    init_state = None
    if init_model_state_dicts is not None:
        init_state = init_model_state_dicts.get(run_seed)

    start_state = None
    if start_model_state_dicts is not None:
        start_state = start_model_state_dicts.get(run_seed)
    
    start_lin_params = None
    if start_lin_params_dicts is not None:
        start_lin_params = start_lin_params_dicts.get(run_seed)

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
        init_model_state_dict=init_state,
        start_model_state_dict=start_state,
        start_lin_params=start_lin_params,
        add_noise=add_noise,
    )

    return run_seed, metrics

def train_multiseed(
    dataset,
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
    gpu_ids=None,  
    init_model_state_dicts=None,
    start_model_state_dicts=None,
    start_lin_params_dicts=None, 
    add_noise=True,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    results = {}
    if not seeds:
        return results

    args_except_seeds = (
        n,
        random_labels,
        eta,
        epochs,
        beta,
        m,
        init_type,
        lam_fc1,
        lam_fc2,
        regularization_scale,
        use_linearized,
        track_jacobian,
        jac_probe_size,
        track_every,
        print_every,
        init_model_state_dicts,
        start_model_state_dicts,
        start_lin_params_dicts,  
        add_noise,
    )

    # create a list of gpu ids & set gpus to spawn
    base_device = device
    if gpu_ids is None:
        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                # if user asks for an explicit device, e.g. "cuda:1"
                idx = int(device.split(":", 1)[1])
                gpu_ids = [idx]
            else:
                num_gpus = torch.cuda.device_count()
                gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [0]

            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass # already set, fine
        else:
            gpu_ids = [None]
    else:
        # user provided the GPU indices once for the whole experiment
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

    if len(seeds) == 1:
        # Sequential fast path (keeps old behavior for single seed)
        dev_str = (base_device if gpu_ids[0] is None else f"cuda:{gpu_ids[0]}")
        run_seed, metrics = _train_multiseed_worker(dataset, seeds[0], dev_str, *args_except_seeds)
        results[run_seed] = metrics
        return results

    # determine max workers
    if gpu_ids[0] is None:
        max_workers = len(seeds)
    else:
        max_workers = min(len(seeds), len(gpu_ids))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, run_seed in enumerate(seeds):
            if gpu_ids[0] is None:
                dev_str = base_device
            else:
                dev_str = f"cuda:{gpu_ids[i % len(gpu_ids)]}"  # round-robin over GPUs
            futures.append(pool.submit(_train_multiseed_worker, dataset, run_seed, dev_str, *args_except_seeds))

        for fut in futures:
            run_seed, metrics = fut.result()
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
    target_loss=None,
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
        # if epoch % print_every == 0:
        #     print(f"    epoch = {epoch:5} | loss = {loss:.2}")
        loss_val = loss.item()
        if epoch % print_every == 0:
            print(f"    epoch = {epoch:5} | loss = {loss_val:.2}")
        if (target_loss is not None) and (loss_val <= target_loss):
            if print_every:
                print(f"    early stopping at epoch = {epoch:5} | loss = {loss_val:.3g}")
            break

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
    print_every=100,
    target_loss=None,
):
    curves = {}
    m_values = [int(min(1e05, beta * np.log(beta))) for beta in betas]
    m_max = max(m_values)
    m_max = 10000  # hardcoded until replication works
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
                print_every=print_every,
                target_loss=target_loss,
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
    target_loss=None,
):
    curves = {}
    m = 10000  # hardcoded until replication works
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
                target_loss=target_loss,
            )
            with torch.no_grad():
                f = model(x_plot).cpu().numpy().ravel()
            fs.append(f)
        curves[alpha] = np.stack(fs, axis=0)
    return curves
