import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, m, d_out=10, with_bias=False, init_type="standard"):
        super().__init__()
        self.d_in = d_in
        self.m = m
        self.d_out = d_out
        self.init_type = init_type
        self.fc1 = nn.Linear(d_in, m, bias=with_bias)
        self.fc2 = nn.Linear(m, d_out, bias=with_bias)
        if init_type == "standard":
            torch.nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="tanh")
            torch.nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        elif init_type == "mean-field":
            with torch.no_grad():
                self.fc1.weight.normal_(0.0, nn.init.calculate_gain("tanh") / d_in)
                self.fc1.weight.normal_(0.0, nn.init.calculate_gain("linear") / m)
        elif init_type == "alpha":
            model.alpha = 0.1  # hardcoded scale factor

            torch.nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="tanh")
            torch.nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")

            with torch.no_grad():
                self.fc1.weight.mul_(alpha)
                self.fc2.weight.mul_(alpha)

                if self.fc1.bias is not None:
                    self.fc1.bias.mul_(alpha)
                if self.fc2.bias is not None:
                    self.fc2.bias.mul_(alpha)
        else:
            raise ValueError(f"Unknown init='{init_type}'. Use 'standard' or 'mean-field' or 'alpha'.")

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
def make_lambda_like_params(model, init_type, lam_fc1, lam_fc2):
    if lam_fc1 is None or lam_fc2 is None:
        if init_type == "standard":
            lam_fc1 = nn.init.calculate_gain("tanh")**2 / model.d_in
            lam_fc2 = nn.init.calculate_gain("linear")**2 / model.m
        elif init_type == "mean-field":
            lam_fc1 = nn.init.calculate_gain("tanh")**2 / (model.d_in**2)
            lam_fc2 = nn.init.calculate_gain("linear")**2 / (model.m**2)
        elif init_type == "alpha":
            lam_fc1 = nn.init.calculate_gain("tanh")**2 / model.d_in / model.alpha
            lam_fc2 = nn.init.calculate_gain("linear")**2 / model.m / model.alpha
        else:
            raise ValueError(f"Unknown init='{init_type}'. Use 'standard' or 'mean-field' or 'alpha'.")

    lam_tensors = []
    params = []
    for name, p in model.named_parameters():
        params.append(p)
        if "fc1" in name:
            lam = torch.full_like(p, lam_fc1)
        elif "fc2" in name:
            lam = torch.full_like(p, lam_fc2)
        else:
            raise ValueError(f"Unknown parameter name: {name}")
        lam_tensors.append(lam)
    return params, lam_tensors
