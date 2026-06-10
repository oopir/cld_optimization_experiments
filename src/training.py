from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import multiprocessing as mp
from typing import Optional

import random
import numpy as np
import torch

from .data import load_digits_data, load_mnist_data
from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import langevin_step, joint_langevin_step
from .linearized import (
    init_linearization,
    linearized_forward,
)
from .stats import (
    BASE_METRIC_NAMES,
    LIN_METRIC_NAMES,
    get_stats,
    get_linear_stats,
    get_nn_lin_param_dist,
    compute_dataset_jac_drift,
    estimate_loss_floor
)

# -------------------------------------------------------------------------- #
# ------------------------------- dataclasses ------------------------------ #
# -------------------------------------------------------------------------- #

@dataclass
class BaseModelVars:
    """Bundle NN state that is reused across training, metrics, and checkpointing."""
    model: TwoLayerNet
    params: list
    lam_tensors: list
    params0: list
    param_norm0: float
    W0: torch.Tensor

@dataclass
class LinearizationVars:
    """Bundle linearized-model state so train() does not pass seven parallel values."""
    base_params_dict: dict
    params: list
    lam_tensors: list
    params0: list

@dataclass
class TrainArgs:
    """Bundle options for one train() call."""
    eta: float
    epochs: int
    beta: float
    m: int
    init_type: str
    alpha: float
    lam_fc1: Optional[float]
    lam_fc2: Optional[float]
    regularization_scale: float
    use_linearized: bool
    same_noise: bool
    track_jacobian: bool
    jac_probe_size: int
    device: str
    track_every: int
    print_every: int
    epoch_offset: int
    collect_feature_stats: bool
    noise_free_after_epoch: Optional[int]
    early_stop_metric: Optional[str]
    early_stop_goal: str
    early_stop_value: Optional[float]

@dataclass
class MultiSeedWorkerArgs:
    """Bundle per-(alpha,beta) options sent to each seed worker."""
    n: int
    random_labels: bool
    reserve_last: int
    train: TrainArgs
    resume_paths: Optional[dict]

@dataclass
class ResumeState:
    """State loaded from a seed checkpoint before continuing training."""
    init_model_state_dict: Optional[dict] = None
    start_model_state_dict: Optional[dict] = None
    start_lin_params: Optional[list] = None
    rng_state: Optional[dict] = None
    last_epoch: Optional[int] = None
    stopped_early: bool = False

# -------------------------------------------------------------------------- #
# ---------------- save/load random state for checkpointing ---------------- #
# -------------------------------------------------------------------------- #

def _save_rng_state(device: str):
    """Capture Python, NumPy, Torch CPU, and optional CUDA RNG state for resume."""
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
    """Restore RNG state saved in checkpoints before continuing a run."""
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
    """Initialize the NN and fixed-at-init quantities used by training stats."""

    model = TwoLayerNet(d_in=d_in, m=m, d_out=d_out, init_type=init_type, alpha=alpha).to(device)
    if init_model_state_dict is not None:
        model.load_state_dict(init_model_state_dict)
    params, lam_tensors = make_lambda_like_params(model, init_type, lam_fc1=lam_fc1, lam_fc2=lam_fc2)

    params0 = [p.detach().clone() for p in params]
    with torch.no_grad():
        param_norm0 = torch.sqrt(sum(p.pow(2).sum() for p in params0)).item()

    W0 = model.fc1.weight.detach().clone()

    return BaseModelVars(model, params, lam_tensors, params0, param_norm0, W0)

def _init_linearization_vars(model, params0, lam_tensors):
    """Initialize the linearized model around the NN initialization."""

    base_params_dict, lin_params, lin_lam_tensors = init_linearization(model, params0, lam_tensors)
    lin_params0 = [p.detach().clone() for p in lin_params]

    return LinearizationVars(
        base_params_dict,
        lin_params,
        lin_lam_tensors,
        lin_params0,
    )

def _init_jacobian_track_vars(d, d_out, m, init_type, alpha, device, model):
    """Prepare an initialization copy for full-dataset NTK/Jacobian drift tracking."""
    model_at_init = TwoLayerNet(d_in=d, m=m, d_out=d_out, init_type=init_type, alpha=alpha).to(device)
    model_at_init.load_state_dict(model.state_dict())
    return model_at_init

def _init_metrics(track_jacobian):
    """Create metric history lists using the public checkpoint key names."""
    metrics = {f"{name}_hist": [] for name in BASE_METRIC_NAMES}
    if track_jacobian:
        metrics["jacobian_dist_hist"] = []
    for name in LIN_METRIC_NAMES:
        metrics[f"{name}_hist"] = []
    metrics["nn_lin_param_dist_hist"] = []
    metrics["epoch_hist"] = []
    return metrics

# -------------------------------------------------------------------------- #
# -------------------- train (& handle parallelization) -------------------- #
# -------------------------------------------------------------------------- #

def _copy_state_dict_to_cpu(state_dict):
    """Detach checkpoint-bound state dict tensors and move them to CPU."""
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def _print_training_start(device, alpha, beta, eta, epoch_offset, epochs):
    """Print the compact per-worker training/resume status line."""
    if epoch_offset < epochs:
        print(
            f"device {device}: training starts for alpha={alpha}, beta={beta}, eta={eta} "
            f"from epoch={epoch_offset+1}...",
            flush=True,
        )
    else:
        print(f"device {device}: no need to train for alpha={alpha}, beta={beta} (early stopping triggered)")

def _print_epoch_progress(device, epoch, stats):
    """Print the compact progress line at print_every checkpoints."""
    print(
        f"device {device} | "
        f"epoch {epoch:8d} | "
        f"loss {stats['train_loss']:.4f} | "
        f"train acc {stats['train_acc']:.3f} | "
        f"test acc {stats['test_acc']:.3f}",
        flush=True,
    )

def _zero_grads(params):
    """Clear manually managed parameter gradients before backward passes."""
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def _forward_backward(model, data, batch_size=1024):
    """Accumulate NN gradients over the full training set in batches."""
    X_train = data["X_train"]
    N = X_train.size(0)
    for start in range(0, N, batch_size):
        end = start + batch_size

        xb = X_train[start:end]
        yb = data["y_train_one_hot"][start:end] if "y_train_one_hot" in data else data["y_train"][start:end]

        outputs = model(xb)
        loss = loss_fn(outputs, yb) * (len(xb) / N)
        loss.backward()

def _should_stop_early(metric_name, goal, threshold, stats, lin_stats):
    """Evaluate early-stop criteria against NN or linearized metrics."""
    if metric_name is None or threshold is None:
        return False, None
    if metric_name in stats:
        current = stats[metric_name]
    elif metric_name.startswith("lin_") and metric_name in lin_stats:
        current = lin_stats[metric_name]
    else:
        return False, None

    should_stop = (goal == "min" and current <= threshold) or (goal == "max" and current >= threshold)
    return should_stop, current

def _apply_training_step(base, lin, beta, eta, regularization_scale, same_noise, noise_free_after_epoch, epoch):
    """Apply Langevin updates, including shared-noise and noise-free tail modes."""
    deterministic = noise_free_after_epoch is not None and epoch > noise_free_after_epoch
    current_beta = float("inf") if deterministic else beta

    if lin is None:
        langevin_step(base.params, base.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
    elif same_noise:
        joint_langevin_step(
            base.params,
            base.lam_tensors,
            lin.params,
            lin.lam_tensors,
            beta=current_beta,
            eta=eta,
            regularization_scale=regularization_scale,
        )
    else:
        langevin_step(base.params, base.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
        langevin_step(lin.params, lin.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)

def _record_linear_metrics(metrics, base, lin, data):
    """Append linearized metrics and NN-vs-linearized distances."""
    lin_stats = get_linear_stats(
        base.model,
        lin.base_params_dict,
        lin.params,
        lin.params0,
        data,
    )
    for name in LIN_METRIC_NAMES:
        metrics[f"{name}_hist"].append(lin_stats[name])

    metrics["nn_lin_param_dist_hist"].append(get_nn_lin_param_dist(base.params, lin.params, normalize_by=base.param_norm0))
    return lin_stats

def _record_epoch_metrics(
    metrics,
    base,
    lin,
    data,
    A0,
    A0_norm,
    collect_feature_stats,
    track_jacobian,
    model_at_init,
    jac_probe_size,
    epoch,
):
    """Append all metrics tracked at a scheduled epoch."""
    metrics["epoch_hist"].append(epoch)

    stats = get_stats(
        base.model,
        base.params,
        base.params0,
        A0,
        A0_norm,
        data,
        collect_feature_stats,
    )
    for name in BASE_METRIC_NAMES:
        metrics[f"{name}_hist"].append(stats[name])

    if track_jacobian:
        jacobian_dist = \
            compute_dataset_jac_drift(base.model, model_at_init, data["X_train"], jac_probe_size)
        metrics["jacobian_dist_hist"].append(jacobian_dist)

    lin_stats = _record_linear_metrics(metrics, base, lin, data) if lin is not None else {}
    return stats, lin_stats

def _finalize_metrics(metrics, base, lin, data, beta, m, device, collect_feature_stats, init_state_for_metrics):
    """Attach final bounds, RNG state, and checkpoint payload tensors."""
    metrics["rng_state"] = _save_rng_state(device)

    if collect_feature_stats:
        metrics["loss_floor"] = estimate_loss_floor(data["X_train"], beta, m=m, device=device)
    else:
        metrics["param_dist_upper_bound"] = float("nan")
        metrics["loss_floor"] = float("nan")

    metrics["model_state_dict"] = _copy_state_dict_to_cpu(base.model.state_dict())
    metrics["init_model_state_dict"] = init_state_for_metrics
    if lin is not None:
        metrics["lin_params_state"] = [p.detach().cpu() for p in lin.params]

    return metrics

def train(data, args, resume_state=None):
    """Run the training loop for one initialized dataset."""
    if resume_state is None:
        resume_state = ResumeState()

    X_train = data["X_train"]

    base = _init_base_model_vars(
        data["d_in"],
        data["d_out"],
        args.m,
        args.init_type,
        args.alpha,
        args.device,
        args.lam_fc1,
        args.lam_fc2,
        resume_state.init_model_state_dict,
    )

    if resume_state.init_model_state_dict is not None:
        init_state_for_metrics = _copy_state_dict_to_cpu(resume_state.init_model_state_dict)
    else:
        init_state_for_metrics = _copy_state_dict_to_cpu(base.model.state_dict())

    lin = None
    if args.use_linearized:
        lin = _init_linearization_vars(base.model, base.params0, base.lam_tensors)
        if resume_state.start_lin_params is not None:
            for p, p_prev in zip(lin.params, resume_state.start_lin_params):
                p.data.copy_(p_prev.to(device=p.device, dtype=p.dtype))

    model_at_init = None
    if args.track_jacobian:
        model_at_init = _init_jacobian_track_vars(
            data["d_in"], 
            data["d_out"], 
            args.m, 
            args.init_type, 
            args.alpha, 
            args.device, 
            base.model, 
            X_train, 
            args.jac_probe_size
        )

    with torch.no_grad():
        A0 = torch.tanh(X_train @ base.model.fc1.weight.T)
        A0_norm = A0.norm().item()

    metrics = _init_metrics(args.track_jacobian)
    metrics["stopped_early"] = False

    if resume_state.start_model_state_dict is not None:
        base.model.load_state_dict(resume_state.start_model_state_dict)

    if resume_state.rng_state is not None:
        _load_rng_state(args.device, resume_state.rng_state)

    _print_training_start(args.device, args.alpha, args.beta, args.eta, args.epoch_offset, args.epochs)
    stats = get_stats(
        base.model,
        base.params,
        base.params0,
        A0,
        A0_norm,
        data,
        args.collect_feature_stats,
    )

    last_epoch = args.epoch_offset
    for epoch in range(args.epoch_offset + 1, args.epochs + 1):
        last_epoch = epoch
        base.model.eval()
        if args.track_every == 1 or epoch % args.track_every == 1:
            stats, lin_stats = _record_epoch_metrics(
                metrics,
                base,
                lin,
                data,
                A0,
                A0_norm,
                args.collect_feature_stats,
                args.track_jacobian,
                model_at_init,
                args.jac_probe_size,
                epoch,
            )

            if args.print_every == 1 or epoch % args.print_every == 1:
                _print_epoch_progress(args.device, epoch, stats)

            should_stop, cur = _should_stop_early(
                args.early_stop_metric,
                args.early_stop_goal,
                args.early_stop_value,
                stats,
                lin_stats,
            )
            if should_stop:
                print(f"device {args.device}: early stopping, epoch={epoch}, {args.early_stop_metric}={cur:.3f}")
                metrics["stopped_early"] = True
                break

        base.model.train()
        _zero_grads(base.params)
        _forward_backward(base.model, data, batch_size=1024)

        if lin is not None:
            _zero_grads(lin.params)
            lin_outputs = linearized_forward(base.model, lin.base_params_dict, lin.params, X_train)
            lin_targets = data.get("y_train_one_hot", data["y_train"])
            loss_fn(lin_outputs, lin_targets).backward()

        _apply_training_step(
            base,
            lin,
            args.beta,
            args.eta,
            args.regularization_scale,
            args.same_noise,
            args.noise_free_after_epoch,
            epoch,
        )

    metrics["last_epoch"] = last_epoch
    return _finalize_metrics(
        metrics,
        base,
        lin,
        data,
        args.beta,
        args.m,
        args.device,
        args.collect_feature_stats,
        init_state_for_metrics,
    )

# -------------------------------------------------------------------------- #

def _load_data_for_seed(dataset, args, run_seed, device):
    """Load the configured dataset for one seed on that worker's device."""
    if dataset == "digits":
        return load_digits_data(n=args.n, random_labels=args.random_labels, device=device, seed=run_seed)
    if dataset == "mnist":
        return load_mnist_data(
            n=args.n,
            random_labels=args.random_labels,
            device=device,
            seed=run_seed,
            reserve_last=args.reserve_last,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")

def _load_resume_state(resume_paths, run_seed):
    """Load optional per-seed resume payload prepared by exp.py."""
    if resume_paths is None:
        return ResumeState()

    path = resume_paths.get(run_seed)
    if path is None:
        return ResumeState()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ResumeState(
        init_model_state_dict=payload.get("init_model_state_dict", None),
        start_model_state_dict=payload.get("start_model_state_dict", None),
        start_lin_params=payload.get("start_lin_params", None),
        rng_state=payload.get("rng_state", None),
        last_epoch=payload.get("last_epoch", None),
        stopped_early=payload.get("stopped_early", False),
    )

def _effective_epoch_window(args, resume_state):
    """Return the epoch bounds for normal runs or already-stopped resumed seeds."""
    if not resume_state.stopped_early:
        return args.train.epoch_offset, args.train.epochs

    if resume_state.last_epoch is not None:
        last_epoch = int(resume_state.last_epoch)
        return last_epoch, last_epoch

    return args.train.epoch_offset, args.train.epoch_offset

def _train_multiseed_worker(
    dataset,
    run_seed,
    device,
    args,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    data = _load_data_for_seed(dataset, args, run_seed, device)
    resume_state = _load_resume_state(args.resume_paths, run_seed)
    call_epoch_offset, call_epochs = _effective_epoch_window(args, resume_state)
    train_args = replace(args.train, device=device, epoch_offset=call_epoch_offset, epochs=call_epochs)

    return run_seed, train(data, train_args, resume_state)

# -------------------------------------------------------------------------- #

def train_multiseed(
    dataset,
    seeds,
    args,
    gpu_ids=None,
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {}
    if not seeds:
        return results

    # create a list of gpu ids & set gpus to spawn
    base_device = args.train.device
    if gpu_ids is None:
        if base_device.startswith("cuda") and torch.cuda.is_available():
            if ":" in base_device:
                # if user asks for an explicit device, e.g. "cuda:1"
                idx = int(base_device.split(":", 1)[1])
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
        if base_device.startswith("cuda") and torch.cuda.is_available():
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

    if len(seeds) == 1:
        # Sequential fast path (keeps old behavior for single seed)
        dev_str = (base_device if gpu_ids[0] is None else f"cuda:{gpu_ids[0]}")
        run_seed, metrics = _train_multiseed_worker(dataset, seeds[0], dev_str, args)
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
                futures.append(pool.submit(_train_multiseed_worker, dataset, run_seed, dev_str, args))

            for fut in futures:
                run_seed, metrics = fut.result()
                results[run_seed] = metrics

    return results
