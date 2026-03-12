from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import random
import numpy as np
import torch

from .data import load_digits_data, load_mnist_data
from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import langevin_step, joint_langevin_step
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
    compute_dataset_ntk_drift,
    compute_dist_bound_under_GF,
    estimate_loss_floor
)

# -------------------------------------------------------------------------- #
# ---------------- save/load random state for checkpointing ---------------- #
# -------------------------------------------------------------------------- #

def _save_rng_state(device: str):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            idx = int(device.split(":", 1)[1])
        else:
            idx = torch.cuda.current_device()
        state["torch_cuda"] = torch.cuda.get_rng_state(device=idx)
    return state

def _load_rng_state(device: str, state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda", None)
    if cuda_state is not None and isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            idx = int(device.split(":", 1)[1])
        else:
            idx = torch.cuda.current_device()
        torch.cuda.set_rng_state(cuda_state, device=idx)

# -------------------------------------------------------------------------- #
# --------------- init variables for training & stat tracking -------------- #
# -------------------------------------------------------------------------- #

def _init_base_model_vars(d_in, d_out, m, init_type, alpha, device, lam_fc1, lam_fc2, init_model_state_dict=None):

    model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=init_type, alpha=alpha).to(device)
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

def _init_jacobian_track_vars(d, d_out, m, init_type, alpha, device, model, X_train, probe_bs):
    # model_at_init is made in case we will want to track Jacobian
    # drift w.r.t. full Jacobian and not just a partial probe
    model_at_init = TwoLayerNet(d_in=d, m=m, d_out=d_out, init_type=init_type, alpha=alpha).to(device)
    model_at_init.load_state_dict(model.state_dict())

    X_probe = X_train[:probe_bs].to(device)
    jac_init = compute_param_jacobians(model, X_probe)
    jac_init_norm_sq = sum(float(ji.pow(2).sum().item()) for ji in jac_init)

    return model_at_init, X_probe, jac_init, jac_init_norm_sq

def _init_metrics(track_jacobian):
    metrics = {f"{name}_hist": [] for name in BASE_METRIC_NAMES}
    if track_jacobian:
        metrics["jacobian_dist_hist"] = []
    for name in LIN_METRIC_NAMES:
        metrics[f"{name}_hist"] = []
    metrics["nn_to_lin_hist"] = []
    metrics["nn_lin_param_dist_hist"] = []
    metrics["epoch_hist"] = []
    return metrics

# -------------------------------------------------------------------------- #
# -------------------- train (& handle parallelization) -------------------- #
# -------------------------------------------------------------------------- #

def _forward_backward(model, data, batch_size=1024):
    X_train = data["X_train"]
    N = X_train.size(0)
    for start in range(0, N, batch_size):
        end = start + batch_size

        xb = X_train[start:end]
        yb = data["y_train_one_hot"][start:end] if "y_train_one_hot" in data else data["y_train"][start:end]

        outputs = model(xb)
        loss = loss_fn(outputs, yb) * (len(xb) / N)
        loss.backward()

def train(
    data,
    eta,
    epochs,
    beta,
    m,
    init_type="standard",
    alpha=1.0,
    lam_fc1=None,
    lam_fc2=None,
    noise_free_after_epoch=None,
    regularization_scale=1.0,
    use_linearized=True,
    same_noise=False,
    track_jacobian=True,
    jac_probe_size=1,
    device="cpu",
    track_every=1,
    print_every=100,
    init_model_state_dict=None,
    start_model_state_dict=None,
    start_lin_params=None,
    resume_rng_state=None,
    epoch_offset=0,
    collect_feature_stats=True,
    early_stop_metric=None,
    early_stop_goal="min",
    early_stop_value=None,
):

    # --------- init environment & compute values at init for stats -------- #
    X_train = data["X_train"]

    model, params, lam_tensors, params0, param_norm0, fc1_norm0, fc2_norm0, W0 = \
        _init_base_model_vars(data["d_in"], data["d_out"], m, init_type, alpha, device, lam_fc1, lam_fc2, init_model_state_dict)

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
        # model_at_init, X_probe, jac_init, jac_init_norm_sq = \
        #     _init_jacobian_track_vars(data["d_in"], data["d_out"], m, init_type, device, model, X_train, jac_probe_size)
        model_at_init, _, _, _ = \
            _init_jacobian_track_vars(data["d_in"], data["d_out"], m, init_type, alpha, device, model, X_train, jac_probe_size)


    with torch.no_grad():
        if model.act == 'relu':
            A0 = torch.relu(X_train @ model.fc1.weight.T)
        elif model.act == 'tanh':
            A0 = torch.tanh(X_train @ model.fc1.weight.T)
        else:
            raise ValueError(f"Tracking of feature distance does not support activation '{model.act}'.")
        A0_norm = A0.norm().item()

    metrics = _init_metrics(track_jacobian)
    metrics["stopped_early"] = False

    if start_model_state_dict is not None:
        model.load_state_dict(start_model_state_dict)

    if resume_rng_state is not None:
        _load_rng_state(device, resume_rng_state)

    if epoch_offset < epochs:
        print(f"device {device}: training starts for alpha={alpha}, beta={beta}, eta={eta} from epoch={epoch_offset+1}...", flush=True)
    else:
        print(f"device {device}: no need to train for alpha={alpha}, beta={beta} (early stopping triggered)", flush=True)
    stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, A0, A0_norm, data, collect_feature_stats)
    sup_sigma_max_v = stats["sigma_max_v"]
    # print(f"epoch {0:8d} | loss {stats['train_loss']:.4f} | train acc {stats['train_acc']:.3f} | test acc {stats['test_acc']:.3f}")

    last_epoch = epoch_offset
    for epoch in range(epoch_offset + 1, epochs + 1):
        last_epoch = epoch
        # -------------------- compute metrics and stats -------------------- #
        model.eval()
        if track_every == 1 or epoch % track_every == 1:
            metrics["epoch_hist"].append(epoch)

            stats = get_stats(model, params, params0, param_norm0, fc1_norm0, fc2_norm0, A0, A0_norm, data, collect_feature_stats)
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

            if print_every == 1 or epoch % print_every == 1:
                print(
                    f"device {device} | "
                    f"epoch {epoch:8d} | "
                    f"loss {stats['train_loss']:.4f} | "
                    f"train acc {stats['train_acc']:.4f} | "
                    f"test acc {stats['test_acc']:.4f}",
                    flush=True
                )

            # stop early (if needed)
            if early_stop_metric is not None and early_stop_value is not None:
                # get the value that should dictate stopping
                if early_stop_metric in stats:
                    cur = stats[early_stop_metric]
                elif use_linearized and 'lin_' in early_stop_metric and early_stop_metric in lin_stats:
                    cur = lin_stats[early_stop_metric]
                else:
                    cur = None
                # decide whether to stop or continue
                if cur is not None:
                    goal = early_stop_goal
                    if (goal == "min" and cur <= early_stop_value) or (goal == "max" and cur >= early_stop_value):
                        print(f"device {device}: early stopping, epoch={epoch}, {early_stop_metric}={cur:.4f}", flush=True)
                        metrics["stopped_early"] = True
                        break


        # ------------------ compute grads & perform steps ------------------ #
        # NN forward + backward
        model.train()
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        _forward_backward(model, data, batch_size=1024)
        # lin forward + backward
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

        # select effective β (optionally disable noise after a threshold)
        deterministic = noise_free_after_epoch is not None and epoch > noise_free_after_epoch
        current_beta = float("inf") if deterministic else beta

        # training step(s)
        if not use_linearized:
            langevin_step(params, lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
        else:
            if same_noise:
                joint_langevin_step(params, lam_tensors, lin_params, lin_lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
            else:
                langevin_step(params, lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
                langevin_step(lin_params, lin_lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)

    metrics["last_epoch"] = last_epoch
    metrics["rng_state"] = _save_rng_state(device)

    # -------------------- compute remaining stats --------------------- #
    if collect_feature_stats:
        metrics["param_dist_upper_bound"] = compute_dist_bound_under_GF(X_train, W0, sup_sigma_max_v)
        metrics["loss_floor"] = estimate_loss_floor(X_train, beta, m=m, device=device)
    else:
        metrics["param_dist_upper_bound"] = float("nan")
        metrics["loss_floor"] = float("nan")
        
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
    reserve_last,
    eta,
    epochs,
    beta,
    m,
    init_type,
    alpha,
    lam_fc1,
    lam_fc2,
    regularization_scale,
    use_linearized,
    same_noise,
    track_jacobian,
    jac_probe_size,
    track_every,
    print_every,
    resume_paths=None,
    epoch_offset=0,
    noise_free_after_epoch=None,
    collect_feature_stats=True, 
    early_stop_metric=None,
    early_stop_goal="min",
    early_stop_value=None,
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
    elif dataset == "mnist":
        data = load_mnist_data(n=n, random_labels=random_labels, device=device, seed=run_seed, reserve_last=reserve_last)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    init_state = None
    start_state = None
    start_lin_params = None
    rng_state = None
    last_epoch = None
    stopped_early = False
    if resume_paths is not None:
        p = resume_paths.get(run_seed)
        if p is not None:
            resume = torch.load(p, map_location="cpu", weights_only=False)
            init_state = resume.get("init_model_state_dict", None)
            start_state = resume.get("start_model_state_dict", None)
            start_lin_params = resume.get("start_lin_params", None)
            rng_state = resume.get("rng_state", None)
            last_epoch = resume.get("last_epoch", None)
            stopped_early = resume.get("stopped_early", False)

    # decide how many epochs (and from what offset) this seed should actually run
    call_epoch_offset = epoch_offset
    call_epochs = epochs
    if stopped_early:
        if last_epoch is not None:
            call_epoch_offset = int(last_epoch)
            call_epochs = int(last_epoch)
        else: # fallback
            call_epoch_offset = epoch_offset
            call_epochs = epoch_offset

    metrics = train(
        data=data,
        eta=eta,
        epochs=call_epochs,
        beta=beta,
        m=m,
        init_type=init_type,
        alpha=alpha,
        lam_fc1=lam_fc1,
        lam_fc2=lam_fc2,
        noise_free_after_epoch=noise_free_after_epoch,
        regularization_scale=regularization_scale,
        use_linearized=use_linearized,
        same_noise=same_noise,
        track_jacobian=track_jacobian,
        jac_probe_size=jac_probe_size,
        device=device,
        track_every=track_every,
        print_every=print_every,
        init_model_state_dict=init_state,
        start_model_state_dict=start_state,
        start_lin_params=start_lin_params,
        resume_rng_state=rng_state,
        epoch_offset=call_epoch_offset,
        collect_feature_stats=collect_feature_stats, 
        early_stop_metric=early_stop_metric,
        early_stop_goal=early_stop_goal,
        early_stop_value=early_stop_value,
    )

    return run_seed, metrics

# note to self - do not forget that the argument order matters because of how we call this function
def train_multiseed(
    dataset,
    seeds,
    n,
    random_labels,
    reserve_last,
    eta,
    epochs,
    beta,
    m,
    init_type="standard",
    alpha=1.0,
    lam_fc1=None,
    lam_fc2=None,
    regularization_scale=1.0,
    use_linearized=True,
    same_noise=False,
    track_jacobian=True,
    jac_probe_size=1,
    device="cpu",
    track_every=1,
    print_every=100,
    epoch_offset=0,
    noise_free_after_epoch=None,
    gpu_ids=None,
    resume_paths=None,
    collect_feature_stats=True,  
    early_stop_metric=None,
    early_stop_goal="min",
    early_stop_value=None,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {}
    if not seeds:
        return results

    args_except_seeds = (
        n,
        random_labels,
        reserve_last,
        eta,
        epochs,
        beta,
        m,
        init_type,
        alpha,
        lam_fc1,
        lam_fc2,
        regularization_scale,
        use_linearized,
        same_noise,
        track_jacobian,
        jac_probe_size,
        track_every,
        print_every,
        resume_paths,
        epoch_offset,
        noise_free_after_epoch,
        collect_feature_stats,
        early_stop_metric,
        early_stop_goal,
        early_stop_value,
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

    for worker_batch_start in range(0, len(seeds), max_workers):
        batch_seeds = seeds[worker_batch_start : worker_batch_start + max_workers]
        with ProcessPoolExecutor(max_workers=len(batch_seeds)) as pool:
            futures = []
            for i, run_seed in enumerate(batch_seeds):
                if gpu_ids[0] is None:
                    dev_str = base_device
                else:
                    dev_str = f"cuda:{gpu_ids[i % len(gpu_ids)]}"  # round-robin over GPUs
                futures.append(pool.submit(_train_multiseed_worker, dataset, run_seed, dev_str, *args_except_seeds))

            for fut in futures:
                run_seed, metrics = fut.result()
                results[run_seed] = metrics

    return results

