import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, hidden, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_out, bias=False)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="tanh")
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

def loss_fn(outputs, targets):
    return torch.nn.functional.mse_loss(outputs, targets)

# ---------------------------------------------------------------------------
# Diagonal lambda (per-parameter) construction
# You can set different lambdas for different parameter tensors.
# This matches diag(lambda) theta as elementwise shrink.
# ---------------------------------------------------------------------------
def make_lambda_like_params(model, lam_fc1, lam_fc2):
    lam_tensors = []
    params = []
    for name, p in model.named_parameters():
        params.append(p)
        if "fc1.weight" in name:
            lam = torch.full_like(p, lam_fc1)
        elif "fc2.weight" in name:
            lam = torch.full_like(p, lam_fc2)
        else:
            raise ValueError(f"Unknown parameter name: {name}")
        lam_tensors.append(lam)
    return params, lam_tensors
