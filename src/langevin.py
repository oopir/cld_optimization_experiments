import math
import torch


MOMENTUM_DISCRETIZATIONS = {"baoab", "euler_maruyama"}


def init_momenta(params):
    """Return zero initial momenta matching a parameter sequence."""
    return [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]


def _noise_like(param, gen=None):
    if gen is not None:
        return torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=gen)
    return torch.randn(param.shape, device=param.device, dtype=param.dtype)


def _regularized_gradient(param, lam, beta, regularization_scale):
    gradient = param.grad.detach().clone()
    if not math.isinf(beta) and regularization_scale != 0.0:
        gradient.add_(lam * param, alpha=regularization_scale / beta)
    return gradient

# ----------------------------------------------------------------------------------------------------
# Langevin + diagonal drift step (Euler–Maruyama)
# theta <- theta - eta * grad - eta*(beta^{-1})*(lambda ⊙ theta) + sqrt(2 eta / beta) * N(0,I)
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def langevin_step(params, lam_tensors, beta, eta, regularization_scale=1.0, gen=None, noises=None):
    noise_scale = math.sqrt(2.0 * eta / beta)
    for i, (param, lam) in enumerate(zip(params, lam_tensors)):
        if param.grad is None:
            continue

        gradient = _regularized_gradient(param, lam, beta, regularization_scale)
        noise = noises[i] if noises is not None else _noise_like(param, gen)

        param.add_(gradient, alpha=-eta)
        param.add_(noise, alpha=noise_scale)


@torch.no_grad()
def momentum_euler_step(
    params,
    lam_tensors,
    momenta,
    beta,
    h,
    gamma,
    regularization_scale=1.0,
    gen=None,
    noises=None,
):
    """Literal explicit Euler--Maruyama step for underdamped Langevin."""
    damping = 1.0 - gamma * h
    noise_scale = math.sqrt(2.0 * gamma * h / beta)
    for i, (param, lam, momentum) in enumerate(zip(params, lam_tensors, momenta)):
        if param.grad is None:
            continue

        gradient = _regularized_gradient(param, lam, beta, regularization_scale)
        noise = noises[i] if noises is not None else _noise_like(param, gen)

        param.add_(momentum, alpha=h)

        momentum.mul_(damping)
        momentum.add_(gradient, alpha=-h)
        momentum.add_(noise, alpha=noise_scale)


@torch.no_grad()
def momentum_baoab_position_step(
    params,
    lam_tensors,
    momenta,
    beta,
    h,
    gamma,
    regularization_scale=1.0,
    gen=None,
    noises=None,
):
    """Apply BAOAB's B-A-O-A stages, leaving the endpoint B kick pending."""
    decay = math.exp(-gamma * h)
    noise_scale = math.sqrt(max(0.0, 1.0 - decay * decay) / beta)
    for i, (param, lam, momentum) in enumerate(zip(params, lam_tensors, momenta)):
        if param.grad is None:
            continue

        gradient = _regularized_gradient(param, lam, beta, regularization_scale)
        noise = noises[i] if noises is not None else _noise_like(param, gen)

        # B: half gradient step
        momentum.add_(gradient, alpha=-0.5 * h)

        # A: half parameter step
        param.add_(momentum, alpha=0.5 * h)

        # O: exact damping and noise step
        momentum.mul_(decay)
        momentum.add_(noise, alpha=noise_scale)

        # A: half parameter step
        param.add_(momentum, alpha=0.5 * h)


@torch.no_grad()
def momentum_baoab_final_gradient_step(params, lam_tensors, momenta, beta, h, regularization_scale=1.0):
    """Complete BAOAB using gradients evaluated at the new position."""
    for param, lam, momentum in zip(params, lam_tensors, momenta):
        if param.grad is None:
            continue

        gradient = _regularized_gradient(param, lam, beta, regularization_scale)

        # B: final half gradient step at the updated parameters
        momentum.add_(gradient, alpha=-0.5 * h)
