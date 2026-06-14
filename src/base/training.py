from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import random
import numpy as np
import torch

from .model import TwoLayerNet, loss_fn, make_lambda_like_params
from .langevin import langevin_step, joint_langevin_step

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

def _configure_deterministic_backend():
    """Set deterministic backend flags shared by training entrypoints."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_training_run(run_seed: int, device: str = "cpu"):
    """Configure reproducible backend behavior and seed all RNGs for one trajectory."""
    _configure_deterministic_backend()
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

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

# -------------------------------------------------------------------------- #
# ----------------------------- train helpers ------------------------------ #
# -------------------------------------------------------------------------- #

def _copy_state_dict_to_cpu(state_dict):
    """Detach checkpoint-bound state dict tensors and move them to CPU."""
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def _zero_grads(params):
    """Clear manually managed parameter gradients before backward passes."""
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def _forward_backward_tensors(model, X_train, targets, batch_size=1024):
    """Accumulate model gradients over explicit tensors in full-batch form."""
    N = X_train.size(0)
    for start in range(0, N, batch_size):
        end = start + batch_size

        xb = X_train[start:end]
        yb = targets[start:end]

        outputs = model(xb)
        loss = loss_fn(outputs, yb) * (len(xb) / N)
        loss.backward()

def _forward_backward(model, data, batch_size=1024):
    """Accumulate NN gradients over the full training set in batches."""
    targets = data["y_train_one_hot"] if "y_train_one_hot" in data else data["y_train"]
    _forward_backward_tensors(model, data["X_train"], targets, batch_size=batch_size)

def _apply_training_step(base, lin, beta, eta, regularization_scale, same_noise, noise_free_after_epoch, epoch):
    """Apply Langevin updates, including shared-noise and noise-free tail modes."""
    deterministic = noise_free_after_epoch is not None and epoch > noise_free_after_epoch
    current_beta = float("inf") if deterministic else beta

    if lin is None:
        langevin_step(base.params, base.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
    elif same_noise:
        joint_langevin_step(base.params, base.lam_tensors, lin.params, lin.lam_tensors, 
                            beta=current_beta, eta=eta, regularization_scale=regularization_scale)
    else:
        langevin_step(base.params, base.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)
        langevin_step(lin.params, lin.lam_tensors, beta=current_beta, eta=eta, regularization_scale=regularization_scale)

def run_full_batch_training_checkpoints(
    d_in: int,
    d_out: int,
    m: int,
    init_type: str,
    alpha: float,
    device: str,
    X_train: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
    eta: float,
    checkpoint_steps: Sequence[int],
    measure_fn: Callable[[int, BaseModelVars], None],
    batch_size: int = 1024,
    lam_fc1: Optional[float] = None,
    lam_fc2: Optional[float] = None,
    regularization_scale: float = 1.0,
    init_model_state_dict: Optional[dict] = None,
) -> BaseModelVars:
    """
    Train one base model trajectory and measure after selected completed updates.

    This is the common-denominator loop shared by scalar probe runs and any
    future full-batch callers that need explicit targets instead of checkpoint
    histories. The richer experiment train() loop keeps its own metric timing,
    linearized model handling, and early-stopping policy.

    measure_fn is called as `measure_fn(completed_updates, base)` at requested
    checkpoints. Its return value is ignored; callers should use it for side
    effects such as appending rows. It should treat `base` as read-only unless
    the caller intentionally wants measurement to alter the continuing trajectory.
    """
    requested_steps = sorted({int(step) for step in checkpoint_steps})
    if not requested_steps:
        raise ValueError("checkpoint_steps must be non-empty.")
    if requested_steps[0] < 0:
        raise ValueError("checkpoint_steps must be non-negative.")

    base = _init_base_model_vars(d_in, d_out, m, init_type, alpha, device, lam_fc1, lam_fc2,
                                 init_model_state_dict=init_model_state_dict)

    next_checkpoint_idx = 0
    if requested_steps[0] == 0:
        base.model.eval()
        measure_fn(0, base)
        next_checkpoint_idx = 1

    max_step = requested_steps[-1]
    for step in range(1, max_step + 1):
        base.model.train()
        _zero_grads(base.params)
        _forward_backward_tensors(base.model, X_train, targets, batch_size=batch_size)
        _apply_training_step(base, lin=None, beta=beta, eta=eta, regularization_scale=regularization_scale, 
                             same_noise=False, noise_free_after_epoch=None, epoch=step,)

        if next_checkpoint_idx < len(requested_steps) and step == requested_steps[next_checkpoint_idx]:
            base.model.eval()
            measure_fn(step, base)
            next_checkpoint_idx += 1

    return base
