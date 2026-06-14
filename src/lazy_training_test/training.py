from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional
import copy

import torch

from ..base.parallel import resolve_worker_gpu_ids, worker_device
from ..base.training import (
    _apply_training_step,
    _configure_deterministic_backend,
    _copy_state_dict_to_cpu,
    _forward_backward,
    _init_base_model_vars,
    _load_rng_state,
    _save_rng_state,
    _zero_grads,
    seed_training_run,
)
from ..base.data import load_digits_data, load_mnist_data
from ..base.model import loss_fn
from ..base.linearized import (
    init_linearization,
    linearized_forward,
)
from ..stats import (
    get_stats,
    get_linear_stats,
    get_nn_lin_param_dist,
    compute_dataset_jac_drift,
    estimate_loss_floor
)
from ..metric_config import BASE_METRIC_NAMES, LIN_METRIC_NAMES, MetricPlan

# -------------------------------------------------------------------------- #
# ------------------------------- dataclasses ------------------------------ #
# -------------------------------------------------------------------------- #

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
    jac_probe_size: int
    device: str
    track_every: int
    print_every: int
    epoch_offset: int
    noise_free_after_epoch: Optional[int]
    early_stop_metric: Optional[str]
    early_stop_goal: str
    early_stop_value: Optional[float]
    metric_plan: MetricPlan

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
# --------------- init variables for training & stat tracking -------------- #
# -------------------------------------------------------------------------- #

def _init_linearization_vars(model, params0, lam_tensors):
    """Initialize the linearized model around the NN initialization."""

    base_params_dict, lin_params, lin_lam_tensors = init_linearization(model, params0, lam_tensors)
    lin_params0 = [p.detach().clone() for p in lin_params]

    return LinearizationVars(base_params_dict, lin_params, lin_lam_tensors, lin_params0)

def _init_metrics(metric_plan):
    """Create metric history lists using the public checkpoint key names."""
    metrics = {
        f"{name}_hist": []
        for name in metric_plan.tracked_metrics
        if name in metric_plan.history_metrics
    }
    metrics["epoch_hist"] = []
    metrics["tracked_metrics"] = list(metric_plan.tracked_metrics)
    return metrics

# -------------------------------------------------------------------------- #
# -------------------- train (& handle parallelization) -------------------- #
# -------------------------------------------------------------------------- #

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

def _should_stop_early(metric_name, goal, threshold, stats, lin_stats):
    """Evaluate early-stop criteria against NN or linearized metrics."""
    if metric_name is None or threshold is None:
        return False, None
    if metric_name in stats:
        current = stats[metric_name]
    elif metric_name in lin_stats:
        current = lin_stats[metric_name]
    else:
        return False, None

    should_stop = (goal == "min" and current <= threshold) or (goal == "max" and current >= threshold)
    return should_stop, current

# -------------------------------------------------------------------------- #

def _record_epoch_metrics(
    metrics,
    base,
    lin,
    data,
    A0,
    A0_norm,
    metric_plan,
    model_at_init,
    jac_probe_size,
    epoch,
):
    """Append all metrics tracked at a scheduled epoch."""
    metrics["epoch_hist"].append(epoch)

    stats = get_stats(base.model, base.params, base.params0, A0, A0_norm, data, metric_plan)
    for name in BASE_METRIC_NAMES:
        if name in metric_plan.history_metrics:
            metrics[f"{name}_hist"].append(stats[name])

    if "jacobian_dist" in metric_plan.compute_metrics:
        jacobian_dist = \
            compute_dataset_jac_drift(base.model, model_at_init, data["X_train"], jac_probe_size)
        stats["jacobian_dist"] = jacobian_dist
        if "jacobian_dist" in metric_plan.history_metrics:
            metrics["jacobian_dist_hist"].append(jacobian_dist)

    lin_stats = {}
    if lin is not None:
        lin_stats = get_linear_stats(base.model, lin.base_params_dict, lin.params, lin.params0, data, metric_plan)
        for name in LIN_METRIC_NAMES:
            if name in metric_plan.history_metrics:
                metrics[f"{name}_hist"].append(lin_stats[name])

        if "nn_lin_param_dist" in metric_plan.compute_metrics:
            lin_stats["nn_lin_param_dist"] = get_nn_lin_param_dist(base.params, lin.params, normalize_by=base.param_norm0)
            if "nn_lin_param_dist" in metric_plan.history_metrics:
                metrics["nn_lin_param_dist_hist"].append(lin_stats["nn_lin_param_dist"])

    return stats, lin_stats

def _finalize_metrics(metrics, base, lin, data, beta, m, device, metric_plan, init_state_for_metrics):
    """Attach final bounds, RNG state, and checkpoint payload tensors."""
    metrics["rng_state"] = _save_rng_state(device)

    if "loss_floor" in metric_plan.final_metrics:
        metrics["loss_floor"] = estimate_loss_floor(data["X_train"], beta, m=m, device=device)

    metrics["model_state_dict"] = _copy_state_dict_to_cpu(base.model.state_dict())
    metrics["init_model_state_dict"] = init_state_for_metrics
    if lin is not None:
        metrics["lin_params_state"] = [p.detach().cpu() for p in lin.params]

    return metrics

# -------------------------------------------------------------------------- #

def train(data, args, resume_state=None):
    """Run the training loop for one initialized dataset."""

    # --------- initialization state and fixed-at-init references ---------- #

    resume_state = ResumeState() if (resume_state is None) else resume_state
    base = _init_base_model_vars(data["d_in"], data["d_out"], args.m, args.init_type, 
                                 args.alpha, args.device, args.lam_fc1, args.lam_fc2, 
                                 resume_state.init_model_state_dict)
    init_state_for_metrics = \
        _copy_state_dict_to_cpu(resume_state.init_model_state_dict) \
        if resume_state.init_model_state_dict is not None \
        else _copy_state_dict_to_cpu(base.model.state_dict())

    lin = None
    if args.use_linearized:
        lin = _init_linearization_vars(base.model, base.params0, base.lam_tensors)
    model_at_init = None
    if args.metric_plan.needs_jacobian_reference:
        model_at_init = copy.deepcopy(base.model)

    X_train = data["X_train"]
    A0 = None
    A0_norm = None
    if args.metric_plan.needs_initial_features:
        with torch.no_grad():
            A0 = torch.tanh(X_train @ base.model.fc1.weight.T)
            A0_norm = A0.norm().item()

    metrics = _init_metrics(args.metric_plan)

    last_epoch = args.epoch_offset # used if training loop ends by being empty

    # -------------- update objects according to resume_state -------------- #

    if resume_state.start_model_state_dict is not None:
        base.model.load_state_dict(resume_state.start_model_state_dict)
    if resume_state.start_lin_params is not None:
        if lin is not None:
            for p, p_prev in zip(lin.params, resume_state.start_lin_params):
                p.data.copy_(p_prev.to(device=p.device, dtype=p.dtype))
    if resume_state.rng_state is not None:
        _load_rng_state(args.device, resume_state.rng_state)
    if bool(resume_state.stopped_early):
        metrics["stopped_early"] = True
    
    # --------------------------- training loop ---------------------------- #

    _print_training_start(args.device, args.alpha, args.beta, args.eta, args.epoch_offset, args.epochs)
    
    for epoch in range(args.epoch_offset + 1, args.epochs + 1):
        last_epoch = epoch

        # record metrics, print training progess and check stopping criteria
        base.model.eval()
        if args.track_every == 1 or epoch % args.track_every == 1:
            stats, lin_stats = \
                _record_epoch_metrics(metrics, base, lin, data, A0, A0_norm, 
                                      args.metric_plan, model_at_init, args.jac_probe_size, epoch)

            if args.print_every == 1 or epoch % args.print_every == 1:
                _print_epoch_progress(args.device, epoch, stats)

            should_stop, cur = _should_stop_early(args.early_stop_metric, args.early_stop_goal, 
                                                  args.early_stop_value, stats, lin_stats)
            if should_stop:
                print(f"device {args.device}: early stopping, epoch={epoch}, {args.early_stop_metric}={cur:.3f}")
                metrics["stopped_early"] = True
                break
        
        # now do the actual training
        base.model.train()
        _zero_grads(base.params)
        _forward_backward(base.model, data, batch_size=1024)
        if lin is not None:
            _zero_grads(lin.params)
            lin_outputs = linearized_forward(base.model, lin.base_params_dict, lin.params, X_train)
            lin_targets = data.get("y_train_one_hot", data["y_train"])
            loss_fn(lin_outputs, lin_targets).backward()
        
        _apply_training_step(base, lin, args.beta, args.eta, args.regularization_scale, 
                             args.same_noise, args.noise_free_after_epoch, epoch)

    # ------------------------- finalize metrics --------------------------- #

    metrics["last_epoch"] = last_epoch
    return _finalize_metrics(metrics, base, lin, data, args.beta, args.m, 
                             args.device, args.metric_plan, init_state_for_metrics)

# -------------------------------------------------------------------------- #

def _load_data_for_seed(dataset, args, run_seed, device):
    """Load the configured dataset for one seed on that worker's device."""
    if dataset == "digits":
        return load_digits_data(n=args.n, random_labels=args.random_labels, device=device, seed=run_seed)
    if dataset == "mnist":
        return load_mnist_data(n=args.n, random_labels=args.random_labels, device=device, seed=run_seed, 
                               reserve_last=args.reserve_last)
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
    seed_training_run(run_seed, device)

    data = _load_data_for_seed(dataset, args, run_seed, device)
    resume_state = _load_resume_state(args.resume_paths, run_seed)
    call_epoch_offset, call_epochs = _effective_epoch_window(args, resume_state)
    train_args = replace(args.train, device=device, epoch_offset=call_epoch_offset, epochs=call_epochs)

    return run_seed, train(data, train_args, resume_state)

def train_multiseed(
    dataset,
    seeds,
    args,
    gpu_ids=None,
):
    _configure_deterministic_backend()

    results = {}
    if not seeds:
        return results

    base_device = args.train.device
    gpu_ids = resolve_worker_gpu_ids(base_device, gpu_ids)

    if len(seeds) == 1:
        # Sequential fast path (keeps old behavior for single seed)
        dev_str = worker_device(base_device, gpu_ids, 0)
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
                dev_str = worker_device(base_device, gpu_ids, i)
                futures.append(pool.submit(_train_multiseed_worker, dataset, run_seed, dev_str, args))

            for fut in futures:
                run_seed, metrics = fut.result()
                results[run_seed] = metrics

    return results
