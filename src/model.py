import math
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    # don't rely on alpha's default value, as train_multiseed also has a 
    # default alpha value and it will always override the value in here...
    def __init__(self, d_in, m, d_out=10, with_bias=False, init_type="standard", alpha=1, act="tanh"):
        super().__init__()
        self.d_in = d_in
        self.m = m
        self.d_out = d_out
        self.init_type = init_type
        self.alpha = alpha
        self.fc1 = nn.Linear(d_in, m, bias=with_bias)
        self.fc2 = nn.Linear(m, d_out, bias=with_bias)

        if act not in ["tanh", "relu"]:
            raise ValueError(f"Unknown act='{act}' Use 'tanh' or 'relu'.")
        self.act = act

        # init weights
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity=self.act)
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        if init_type == "standard":
            pass
        elif init_type == "mean-field":
            with torch.no_grad():
                self.fc1.weight.mul_(1 / math.sqrt(d_in))
                self.fc2.weight.mul_(1 / math.sqrt(m))
        elif init_type == "alpha":
            with torch.no_grad():
                # kernel-vs-rich, figure 6 caption: "The weights of the network are set using 
                # the Uniform He initialization, and then multiplied by α" (this means std is multiplied by α)
                self.fc1.weight.mul_(self.alpha)
                self.fc2.weight.mul_(self.alpha)
        else:
            raise ValueError(f"Unknown init='{init_type}'. Use 'standard' or 'mean-field' or 'alpha'.")

        # init biases
        # "biases are same as the weights (just have 1 as input)"
        if with_bias:
            with torch.no_grad():
                self.fc1.bias.normal_(mean=0.0, std=nn.init.calculate_gain(self.act))
                self.fc2.bias.normal_(mean=0.0, std=nn.init.calculate_gain("linear"))
                if init_type == "alpha":
                    self.fc1.bias.mul_(self.alpha)
                    self.fc2.bias.mul_(self.alpha)


    def forward(self, x):
        x = self.fc1(x)
        if self.act == "tanh":
            x = torch.tanh(x)
        elif self.act == "relu":
            x = torch.relu(x)
        else:
            raise ValueError(f"Model's 'forward' does not support activation '{self.act}'.")
        x = self.fc2(x)
        if self.init_type == "alpha" and self.alpha != 0:
            x = x / self.alpha
        return x

def loss_fn(outputs, targets):
    return torch.nn.functional.mse_loss(outputs, targets)

# ---------------------------------------------------------------------------
# Diagonal lambda (per-parameter) construction
# ---------------------------------------------------------------------------
def make_lambda_like_params(model, init_type, lam_fc1, lam_fc2, lam_bi1=None, lam_bi2=None):
    tanh_gain_sq = nn.init.calculate_gain(model.act)**2
    lin_gain_sq  = nn.init.calculate_gain("linear")**2

    if lam_fc1 is None or lam_fc2 is None:
        if init_type == "standard":
            lam_fc1 = model.d_in / tanh_gain_sq
            lam_fc2 = model.m / lin_gain_sq
        elif init_type == "mean-field":
            lam_fc1 = (model.d_in**2) / tanh_gain_sq
            lam_fc2 = (model.m**2) / lin_gain_sq
        elif init_type == "alpha":
            lam_fc1 = (model.d_in / tanh_gain_sq) / model.alpha**2
            lam_fc2 = (model.m / lin_gain_sq) / model.alpha**2
        else:
            raise ValueError(f"Unknown init='{init_type}'. Use 'standard' or 'mean-field' or 'alpha'.")

    # note for future: I think this isn't updated to deal with mean-field initialization
    if lam_bi1 is None or lam_bi2 is None:
        lam_bi1 = 1 / tanh_gain_sq
        lam_bi2 = 1 / lin_gain_sq
        if init_type == "alpha":
            lam_bi1 /= model.alpha**2
            lam_bi2 /= model.alpha**2


    lam_tensors = []
    params = []
    for name, p in model.named_parameters():
        params.append(p)
        if "fc1.weight" in name:
            lam = torch.full_like(p, lam_fc1)
        elif "fc2.weight" in name:
            lam = torch.full_like(p, lam_fc2)
        elif "fc1.bias" in name:
            lam = torch.full_like(p, lam_bi1)
        elif "fc2.bias" in name:
            lam = torch.full_like(p, lam_bi2)
        else:
            raise ValueError(f"Unknown parameter name: {name}")
        lam_tensors.append(lam)
    return params, lam_tensors
