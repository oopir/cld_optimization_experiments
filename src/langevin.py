import torch
import math

# ----------------------------------------------------------------------------------------------------
# Langevin + diagonal drift step (Euler–Maruyama)
# theta <- theta - eta * grad - eta*(beta^{-1})*(lambda ⊙ theta) + sqrt(2 eta / beta) * N(0,I)
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def langevin_step(params, lam_tensors, beta, eta, regularization_scale=1.0, gen=None):
    noise_scale = math.sqrt(2.0 * eta / beta)
    for p, lam in zip(params, lam_tensors):
        if p.grad is None:
            continue
        # gradient term
        p.add_(p.grad, alpha=-eta)
        # diagonal shrink term: -(eta/beta) * (lam ⊙ theta)
        p.add_(regularization_scale * lam * p, alpha=-(eta / beta))
        # isotropic noise
        if gen:
            p.add_(torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen) * noise_scale)
        else:
            p.add_(torch.randn(p.shape, device=p.device, dtype=p.dtype) * noise_scale)

@torch.no_grad()
def joint_langevin_step(params1, lam_tensors1, params2, lam_tensors2, beta, eta, regularization_scale=1.0, gen=None):
    noise_scale = math.sqrt(2.0 * eta / beta)
    for p1, lam1, p2, lam2 in zip(params1, lam_tensors1, params2, lam_tensors2):
        if p1.grad is None or p2.grad is None:
            continue
        
        if gen:
            noise = torch.randn(p1.shape, device=p1.device, dtype=p1.dtype, generator=gen) * noise_scale
        else:
            noise = torch.randn(p1.shape, device=p1.device, dtype=p1.dtype) * noise_scale
        
        for p, lam in ((p1, lam1), (p2, lam2)):
            # gradient term
            p.add_(p.grad, alpha=-eta)
            # diagonal shrink term: -(eta/beta) * (lam ⊙ theta)
            p.add_(regularization_scale * lam * p, alpha=-(eta / beta))
            # isotropic noise
            p.add_(noise)