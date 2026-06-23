from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import copy
import multiprocessing as mp
from typing import Optional

import random
import numpy as np
import torch

from .data import load_digits_data, load_mnist_data
from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import (
    init_momenta,
    langevin_step,
    momentum_baoab_final_gradient_step,
    momentum_baoab_position_step,
    momentum_euler_step,
)
from .linearized import (
    init_linearization,
    linearized_forward,
)
from .stats import (
    get_stats,
    get_linear_stats,
    get_nn_lin_param_dist,
    compute_dataset_jac_drift,
    estimate_loss_floor
)
from .metric_config import BASE_METRIC_NAMES, LIN_METRIC_NAMES, MetricPlan

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
class MomentumVars:
    """State for the momentum NN and its optional linearized counterpart."""
    base: BaseModelVars
    lin: Optional[LinearizationVars]
    buffers: list
    lin_buffers: Optional[list]

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
    compare_momentum: bool
    momentum_discretization: str
    momentum_h: Optional[float]
    momentum_gamma: Optional[float]
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
    start_momentum_model_state_dict: Optional[dict] = None
    start_momentum_lin_params: Optional[list] = None
    start_momentum_buffers: Optional[list] = None
    start_momentum_lin_buffers: Optional[list] = None
    rng_state: Optional[dict] = None
    last_epoch: Optional[int] = None
    stopped_early: bool = False

# -------------------------------------------------------------------------- #
# ----------------------------- small helpers ------------------------------ #
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

def _randn_like_params(params):
        return [torch.randn_like(p) for p in params]

def _copy_state_dict_to_cpu(state_dict):
    """Detach checkpoint-bound state dict tensors and move them to CPU."""
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def _print_training_start(device, alpha, beta, eta, epoch_offset, epochs, momentum_h=None, momentum_gamma=None):
    """Print the compact per-worker training/resume status line."""
    if epoch_offset < epochs:
        print(
            f"device {device}: training starts for alpha={alpha}, beta={beta}, eta={eta} "
            + (f"h={momentum_h}, gamma={momentum_gamma} " if momentum_h is not None else "")
            + f"from epoch={epoch_offset+1}...",
            flush=True,
        )
    else:
        print(f"device {device}: no need to train for alpha={alpha}, beta={beta} (early stopping triggered)")

def _print_epoch_progress(device, epoch, stats, momentum_stats=None):
    """Print the compact progress line at print_every checkpoints."""
    print(
        f"device {device} | "
        f"epoch {epoch:8d} | "
        f"loss {stats['train_loss']:.4f} | "
        f"train acc {stats['train_acc']:.3f} | "
        f"test acc {stats['test_acc']:.3f}"
        + (f" | momentum loss {momentum_stats['train_loss']:.4f}" if momentum_stats is not None else ""),
        flush=True,
    )

def _zero_grads(params):
    """Clear manually managed parameter gradients before backward passes."""
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

# -------------------------------------------------------------------------- #
# --------------- prep variables for training & stat tracking -------------- #
# -------------------------------------------------------------------------- #

def _clone_base_model_vars(base):
    """Clone an initialized NN without consuming random numbers."""
    model = copy.deepcopy(base.model)
    params = list(model.parameters())
    return BaseModelVars(
        model=model,
        params=params,
        lam_tensors=[lam.detach().clone() for lam in base.lam_tensors],
        params0=[p0.detach().clone() for p0 in base.params0],
        param_norm0=base.param_norm0,
        W0=base.W0.detach().clone(),
    )

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
    return LinearizationVars(base_params_dict, lin_params, lin_lam_tensors, lin_params0)

def _init_momentum_vars(base, args):
    """Initialize momentum trajectories at the common parameter initialization."""
    if not args.compare_momentum:
        return None

    momentum_base = _clone_base_model_vars(base)
    momentum_lin = None
    momentum_buffers = init_momenta(momentum_base.params)
    momentum_lin_buffers = None
    if args.use_linearized:
        momentum_lin = _init_linearization_vars(
            momentum_base.model, momentum_base.params0, momentum_base.lam_tensors
        )
        momentum_lin_buffers = init_momenta(momentum_lin.params)

    return MomentumVars(momentum_base, momentum_lin, momentum_buffers, momentum_lin_buffers)

def _init_jacobian_track_vars(d, d_out, m, init_type, alpha, device, model):
    """Prepare an initialization copy for full-dataset NTK/Jacobian drift tracking."""
    model_at_init = TwoLayerNet(d_in=d, m=m, d_out=d_out, init_type=init_type, alpha=alpha).to(device)
    model_at_init.load_state_dict(model.state_dict())
    return model_at_init

def _init_metrics(metric_plan, compare_momentum=False):
    """Create metric history lists using the public checkpoint key names."""
    metrics = {
        f"{name}_hist": []
        for name in metric_plan.tracked_metrics
        if name in metric_plan.history_metrics
    }
    if compare_momentum:
        metrics.update({
            f"momentum_{name}_hist": []
            for name in metric_plan.tracked_metrics
            if name in metric_plan.history_metrics
        })
    metrics["epoch_hist"] = []
    metrics["tracked_metrics"] = list(metric_plan.tracked_metrics)
    return metrics

def _restore_training_state(base, lin, momentum, args, resume_state):
    """Move all active trajectories from initialization to their checkpoint state."""
    # NN
    if resume_state.start_model_state_dict is not None:
        base.model.load_state_dict(resume_state.start_model_state_dict)
    # linearized model
    if lin is not None and resume_state.start_lin_params is not None:
        for p, previous in zip(lin.params, resume_state.start_lin_params):
            p.data.copy_(previous.to(device=p.device, dtype=p.dtype))

    if momentum is not None:
        # validate that resume_state contains everything needed for momentum resume
        if resume_state.start_model_state_dict is not None:
            required = {
                "momentum model": resume_state.start_momentum_model_state_dict,
                "momentum buffers": resume_state.start_momentum_buffers,
            }
            if args.use_linearized:
                required.update({
                    "momentum linearized parameters": resume_state.start_momentum_lin_params,
                    "momentum linearized buffers": resume_state.start_momentum_lin_buffers,
                })
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("Cannot resume momentum run; checkpoint is missing " + ", ".join(missing))
        # NN
        if resume_state.start_momentum_model_state_dict is not None:
            momentum.base.model.load_state_dict(resume_state.start_momentum_model_state_dict)
        if resume_state.start_momentum_buffers is not None:
            for p, previous in zip(momentum.buffers, resume_state.start_momentum_buffers):
                p.copy_(previous.to(device=p.device, dtype=p.dtype))
        # linearized model
        if momentum.lin is not None and resume_state.start_momentum_lin_params is not None:
            for p, previous in zip(momentum.lin.params, resume_state.start_momentum_lin_params):
                p.data.copy_(previous.to(device=p.device, dtype=p.dtype))
        if momentum.lin_buffers is not None and resume_state.start_momentum_lin_buffers is not None:
            for p, previous in zip(momentum.lin_buffers, resume_state.start_momentum_lin_buffers):
                p.copy_(previous.to(device=p.device, dtype=p.dtype))

    # RNG
    if resume_state.rng_state is not None:
        _load_rng_state(args.device, resume_state.rng_state)

# -------------------------------------------------------------------------- #
# ----------------------------- record metrics ----------------------------- #
# -------------------------------------------------------------------------- #

def _record_linear_metrics(metrics, base, lin, data, metric_plan, prefix=""):
    """Append linearized metrics and NN-vs-linearized distances."""
    lin_stats = get_linear_stats(
        base.model,
        lin.base_params_dict,
        lin.params,
        lin.params0,
        data,
        metric_plan,
    )
    for name in LIN_METRIC_NAMES:
        if name in metric_plan.history_metrics:
            metrics[f"{prefix}{name}_hist"].append(lin_stats[name])

    if "nn_lin_param_dist" in metric_plan.compute_metrics:
        lin_stats["nn_lin_param_dist"] = get_nn_lin_param_dist(base.params, lin.params, normalize_by=base.param_norm0)
        if "nn_lin_param_dist" in metric_plan.history_metrics:
            metrics[f"{prefix}nn_lin_param_dist_hist"].append(lin_stats["nn_lin_param_dist"])
    return lin_stats

def _record_trajectory_metrics(metrics, base, lin, data, A0, A0_norm, metric_plan,
                               model_at_init, jac_probe_size, prefix=""):
    """Compute and append metrics for one NN/linearized dynamics pair."""
    stats = get_stats(
        base.model,
        base.params,
        base.params0,
        A0,
        A0_norm,
        data,
        metric_plan,
    )
    for name in BASE_METRIC_NAMES:
        if name in metric_plan.history_metrics:
            metrics[f"{prefix}{name}_hist"].append(stats[name])

    if "jacobian_dist" in metric_plan.compute_metrics:
        jacobian_dist = \
            compute_dataset_jac_drift(base.model, model_at_init, data["X_train"], jac_probe_size)
        stats["jacobian_dist"] = jacobian_dist
        if "jacobian_dist" in metric_plan.history_metrics:
            metrics[f"{prefix}jacobian_dist_hist"].append(jacobian_dist)

    lin_stats = _record_linear_metrics(metrics, base, lin, data, metric_plan, prefix) if lin is not None else {}
    return stats, lin_stats

def _record_epoch_metrics(metrics, base, lin, momentum, data, A0, A0_norm,
                          args, model_at_init, epoch):
    """Record all active trajectories at one scheduled experiment epoch."""
    metrics["epoch_hist"].append(epoch)

    stats, lin_stats = \
        _record_trajectory_metrics(metrics, base, lin, data, A0, A0_norm,
                                   args.metric_plan, model_at_init, args.jac_probe_size)

    momentum_stats = None
    if momentum is not None:
        momentum_stats, _ = \
            _record_trajectory_metrics(metrics, momentum.base, momentum.lin, data, A0, A0_norm,
                                       args.metric_plan, model_at_init, args.jac_probe_size,
                                       prefix="momentum_")

    return stats, lin_stats, momentum_stats

def _finalize_metrics(metrics, base, lin, data, beta, m, device, metric_plan, init_state_for_metrics, momentum=None):
    """Attach final bounds, RNG state, and checkpoint payload tensors."""
    metrics["rng_state"] = _save_rng_state(device)

    if "loss_floor" in metric_plan.final_metrics:
        metrics["loss_floor"] = estimate_loss_floor(data["X_train"], beta, m=m, device=device)

    metrics["model_state_dict"] = _copy_state_dict_to_cpu(base.model.state_dict())
    metrics["init_model_state_dict"] = init_state_for_metrics
    if lin is not None:
        metrics["lin_params_state"] = [p.detach().cpu() for p in lin.params]
    if momentum is not None:
        metrics["momentum_model_state_dict"] = _copy_state_dict_to_cpu(momentum.base.model.state_dict())
        metrics["momentum_buffers_state"] = [p.detach().cpu() for p in momentum.buffers]
        if momentum.lin is not None:
            metrics["momentum_lin_params_state"] = [p.detach().cpu() for p in momentum.lin.params]
            metrics["momentum_lin_buffers_state"] = [p.detach().cpu() for p in momentum.lin_buffers]

    return metrics

# -------------------------------------------------------------------------- #
# -------------------- train (& handle parallelization) -------------------- #
# -------------------------------------------------------------------------- #

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

def _compute_trajectory_gradients(base, lin, data):
    _zero_grads(base.params)
    _forward_backward(base.model, data, batch_size=1024)
    if lin is not None:
        _zero_grads(lin.params)
        outputs = linearized_forward(base.model, lin.base_params_dict, lin.params, data["X_train"])
        loss_fn(outputs, data.get("y_train_one_hot", data["y_train"])).backward()

def _trajectory_noises(base, lin, momentum_base, momentum_lin, same_noise):
    """Return noise for overdamped NN/linearized and momentum NN/linearized, in that order."""
    overdamped_base = _randn_like_params(base.params)
    if same_noise:
        return (
            overdamped_base,
            overdamped_base if lin is not None else None,
            overdamped_base if momentum_base is not None else None,
            overdamped_base if momentum_lin is not None else None,
        )
    return (
        overdamped_base,
        _randn_like_params(lin.params) if lin is not None else None,
        _randn_like_params(momentum_base.params) if momentum_base is not None else None,
        _randn_like_params(momentum_lin.params) if momentum_lin is not None else None,
    )

def _apply_overdamped_training_step(base, lin, current_beta, eta, regularization_scale, base_noise, lin_noise):
    """Advance the overdamped NN and its optional linearized counterpart."""
    langevin_step(base.params, base.lam_tensors, beta=current_beta, eta=eta,
                  regularization_scale=regularization_scale, noises=base_noise)
    if lin is not None:
        langevin_step(lin.params, lin.lam_tensors, beta=current_beta, eta=eta,
                      regularization_scale=regularization_scale, noises=lin_noise)

def _apply_momentum_training_step(momentum, data, args, current_beta, base_noise, lin_noise):
    """Advance momentum trajectories and complete BAOAB when needed."""
    step = momentum_euler_step
    if args.momentum_discretization == "baoab":
        step = momentum_baoab_position_step

    step(momentum.base.params, momentum.base.lam_tensors, momentum.buffers, current_beta,
         args.momentum_h, args.momentum_gamma, args.regularization_scale, noises=base_noise)
    if momentum.lin is not None:
        step(momentum.lin.params, momentum.lin.lam_tensors, momentum.lin_buffers, current_beta,
             args.momentum_h, args.momentum_gamma, args.regularization_scale, noises=lin_noise)

    if args.momentum_discretization != "baoab":
        return False
    else:
        _compute_trajectory_gradients(momentum.base, momentum.lin, data)
        momentum_baoab_final_gradient_step(momentum.base.params, momentum.base.lam_tensors, momentum.buffers,
                                  current_beta, args.momentum_h, args.regularization_scale)
        if momentum.lin is not None:
            momentum_baoab_final_gradient_step(momentum.lin.params, momentum.lin.lam_tensors, momentum.lin_buffers,
                                      current_beta, args.momentum_h, args.regularization_scale)
        return True

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

def train(data, args, resume_state=None):
    """Run the training loop for one initialized dataset."""

    # --------- initialization state and fixed-at-init references ---------- #

    if resume_state is None:
        resume_state = ResumeState()

    X_train = data["X_train"]

    # Initialize the baseline NN, its optional linearization, and the coupled
    # momentum variants from one common parameter initialization.
    base = _init_base_model_vars(data["d_in"], data["d_out"], args.m, args.init_type,
                                 args.alpha, args.device, args.lam_fc1, args.lam_fc2,
                                 resume_state.init_model_state_dict)

    if resume_state.init_model_state_dict is not None:
        init_state_for_metrics = _copy_state_dict_to_cpu(resume_state.init_model_state_dict)
    else:
        init_state_for_metrics = _copy_state_dict_to_cpu(base.model.state_dict())

    lin = None
    if args.use_linearized:
        lin = _init_linearization_vars(base.model, base.params0, base.lam_tensors)

    momentum = _init_momentum_vars(base, args)

    # Keep initialization-time references only when requested metrics need them.
    model_at_init = None
    if args.metric_plan.needs_jacobian_reference:
        model_at_init = _init_jacobian_track_vars(data["d_in"], data["d_out"], args.m,
                                                  args.init_type, args.alpha, args.device, base.model)

    A0 = None
    A0_norm = None
    if args.metric_plan.needs_initial_features:
        with torch.no_grad():
            A0 = torch.tanh(X_train @ base.model.fc1.weight.T)
            A0_norm = A0.norm().item()

    metrics = _init_metrics(args.metric_plan, args.compare_momentum)
    metrics["stopped_early"] = False

    # -------- restore current trajectory positions and random state ------- #

    _restore_training_state(base, lin, momentum, args, resume_state)

    # --------------------------- training loop ---------------------------- #

    _print_training_start(args.device, args.alpha, args.beta, args.eta, args.epoch_offset, args.epochs,
                          args.momentum_h if args.compare_momentum else None,
                          args.momentum_gamma if args.compare_momentum else None)
    stats = get_stats(base.model, base.params, base.params0, A0, A0_norm, data, args.metric_plan)

    momentum_grads_ready = False
    last_epoch = args.epoch_offset
    for epoch in range(args.epoch_offset + 1, args.epochs + 1):
        last_epoch = epoch

        # Record metrics, print progress, and check the baseline stopping criterion.
        base.model.eval()
        if momentum is not None:
            momentum.base.model.eval()
        if args.track_every == 1 or epoch % args.track_every == 1:
            stats, lin_stats, momentum_stats = \
                _record_epoch_metrics(metrics, base, lin, momentum, data, A0, A0_norm,
                                      args, model_at_init, epoch)

            if args.print_every == 1 or epoch % args.print_every == 1:
                _print_epoch_progress(args.device, epoch, stats, momentum_stats)

            should_stop, cur = _should_stop_early(args.early_stop_metric, args.early_stop_goal,
                                                  args.early_stop_value, stats, lin_stats)
            if should_stop:
                print(f"device {args.device}: early stopping, epoch={epoch}, {args.early_stop_metric}={cur:.3f}")
                metrics["stopped_early"] = True
                break

        # Compute gradients and advance the active trajectories by one step.
        base.model.train()
        _compute_trajectory_gradients(base, lin, data)
        if momentum is not None:
            momentum.base.model.train()
            if args.momentum_discretization != "baoab" or not momentum_grads_ready:
                _compute_trajectory_gradients(momentum.base, momentum.lin, data)

        # Draw either one common Gaussian sample or one independent sample
        # per trajectory, then apply each dynamics' own noise scaling.
        overdamped_base_noise, overdamped_lin_noise, momentum_base_noise, momentum_lin_noise = \
            _trajectory_noises(
                base,
                lin,
                momentum.base if momentum is not None else None,
                momentum.lin  if momentum is not None else None,
                args.same_noise
            )
        deterministic = args.noise_free_after_epoch is not None and epoch > args.noise_free_after_epoch
        current_beta = float("inf") if deterministic else args.beta

        # Advance the overdamped pair, followed by the momentum pair.
        _apply_overdamped_training_step(base, lin, current_beta, args.eta, args.regularization_scale,
                                        overdamped_base_noise, overdamped_lin_noise)
        if momentum is not None:
            momentum_grads_ready = \
                _apply_momentum_training_step(momentum, data, args, current_beta,
                                          momentum_base_noise, momentum_lin_noise)

    # ------------------------- finalize metrics --------------------------- #

    metrics["last_epoch"] = last_epoch
    return _finalize_metrics(metrics, base, lin, data, args.beta, args.m, args.device,
                             args.metric_plan, init_state_for_metrics, momentum)

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
        start_momentum_model_state_dict=payload.get("start_momentum_model_state_dict", None),
        start_momentum_lin_params=payload.get("start_momentum_lin_params", None),
        start_momentum_buffers=payload.get("start_momentum_buffers", None),
        start_momentum_lin_buffers=payload.get("start_momentum_lin_buffers", None),
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
