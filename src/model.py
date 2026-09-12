from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn

from .config import ExpConfig


@dataclass(frozen=True)
class LayerSpec:
    name: str
    fan_in: int
    nonlinearity: str
    lam: float


def _gain(nonlinearity: str) -> float:
    if nonlinearity == "tanh":
        return nn.init.calculate_gain("tanh")
    if nonlinearity == "linear":
        return nn.init.calculate_gain("linear")
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


def default_lambda(fan_in: int, nonlinearity: str, init_type: str, alpha: float) -> float:
    gain_sq = _gain(nonlinearity) ** 2
    if init_type == "standard":
        return fan_in / gain_sq
    if init_type == "mean-field":
        return (fan_in**2) / gain_sq
    if init_type == "alpha":
        return (fan_in / gain_sq) / (alpha**2)
    raise ValueError(f"Unknown init_type={init_type!r}")


def _scale_initialized_weight(
    weight: torch.Tensor,
    fan_in: int,
    nonlinearity: str,
    init_type: str,
    alpha: float,
) -> torch.Tensor:
    std = _gain(nonlinearity) / math.sqrt(fan_in)
    weight.mul_(std)
    if init_type == "mean-field":
        weight.mul_(1.0 / math.sqrt(fan_in))
    elif init_type == "alpha":
        weight.mul_(alpha)
    elif init_type != "standard":
        raise ValueError(f"Unknown init_type={init_type!r}")
    return weight


def _chunk_seed(layer_seed: int, chunk_idx: int) -> int:
    return int(layer_seed) + int(chunk_idx) * 1_000_003


def _initialized_weight(
    rows: int,
    cols: int,
    nonlinearity: str,
    init_type: str,
    alpha: float,
    device: torch.device,
    seed: int,
    max_chunk_elements: int,
) -> torch.Tensor:
    """Initialize a dense tensor from deterministic global row chunks."""
    # TODO: In the single-device engine, replace temporary chunk generation with
    # in-place initialization of `out` if that preserves the desired seed behavior.
    out = torch.empty((rows, cols), device=device, dtype=torch.float32)
    rows_per_chunk = max(1, max_chunk_elements // max(1, cols))
    for chunk_idx, start in enumerate(range(0, rows, rows_per_chunk)):
        end = min(rows, start + rows_per_chunk)
        generator = torch.Generator(device=device)
        generator.manual_seed(_chunk_seed(seed, chunk_idx))
        chunk = torch.randn(
            (end - start, cols),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        _scale_initialized_weight(chunk, cols, nonlinearity, init_type, alpha)
        out[start:end].copy_(chunk)
    return out


class MLP(nn.Module):
    """Single-device dense tanh MLP retained under the historical package name."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        config: ExpConfig,
        device: torch.device,
        alpha: float,
        seed: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.m = config.m
        self.L = config.L
        self.init_type = config.init_type
        self.alpha = alpha
        self.device = device
        self.use_linearized = config.use_linearized
        self.output_scale = (1.0 / alpha) if config.init_type == "alpha" and alpha != 0 else 1.0
        self.noise_chunk_elements = 50_000_000
        self.init_chunk_elements = 50_000_000

        self.layer_specs = self._build_layer_specs(config, alpha)
        self.hidden = nn.ParameterList()
        self.lin_hidden = nn.ParameterList()
        self.init_hidden: list[torch.Tensor] = []

        # TODO: Simplify the legacy per-layer/per-chunk seed derivation while
        # retaining a separate checkpointable generator for Langevin noise.
        base_seed = int(seed) * 1_000_003
        for idx, spec in enumerate(self.layer_specs[:-1]):
            init = _initialized_weight(
                config.m,
                spec.fan_in,
                spec.nonlinearity,
                config.init_type,
                alpha,
                device,
                base_seed + idx * 97_409,
                self.init_chunk_elements,
            )
            self.hidden.append(nn.Parameter(init.clone()))
            if self.use_linearized:
                self.lin_hidden.append(nn.Parameter(init.clone()))
            self.init_hidden.append(init.detach().clone())

        out_spec = self.layer_specs[-1]
        out_init = _initialized_weight(
            d_out,
            config.m,
            out_spec.nonlinearity,
            config.init_type,
            alpha,
            device,
            base_seed + 9_991_337,
            self.init_chunk_elements,
        )

        self.output = nn.Parameter(out_init.clone())
        if self.use_linearized:
            self.lin_output = nn.Parameter(out_init.clone())
        else:
            self.lin_output = None
        self.init_output = out_init.detach().clone()

        self.noise_gen = torch.Generator(device=device)
        self.noise_gen.manual_seed(base_seed + 123_457)

    def _build_layer_specs(self, config: ExpConfig, alpha: float) -> list[LayerSpec]:
        lam_fc1 = config.lam_fc1
        if lam_fc1 is None:
            lam_fc1 = default_lambda(self.d_in, "tanh", config.init_type, alpha)

        lam_hidden = config.lam_hidden
        if lam_hidden is None:
            lam_hidden = default_lambda(config.m, "tanh", config.init_type, alpha)

        lam_fc2 = config.lam_fc2
        if lam_fc2 is None:
            lam_fc2 = default_lambda(config.m, "linear", config.init_type, alpha)

        specs = [LayerSpec("hidden_0", self.d_in, "tanh", float(lam_fc1))]
        for idx in range(1, config.L):
            specs.append(LayerSpec(f"hidden_{idx}", config.m, "tanh", float(lam_hidden)))
        specs.append(LayerSpec("output", config.m, "linear", float(lam_fc2)))
        return specs

    def base_parameters_list(self) -> list[torch.Tensor]:
        return list(self.hidden) + [self.output]

    def lin_parameters_list(self) -> list[torch.Tensor]:
        if not self.use_linearized:
            return []
        return list(self.lin_hidden) + [self.lin_output]

    def init_tensors_list(self) -> list[torch.Tensor]:
        return list(self.init_hidden) + [self.init_output]

    def lambdas(self) -> list[float]:
        return [spec.lam for spec in self.layer_specs]

    def _forward_from(
        self,
        hidden_weights: Iterable[torch.Tensor],
        output_weight: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        a_prev_full = X
        last_local = None
        hidden_weights = list(hidden_weights)
        for idx, weight in enumerate(hidden_weights):
            z_local = a_prev_full.matmul(weight.t())
            a_local = torch.tanh(z_local)
            a_prev_full = a_local
            last_local = a_local

        assert last_local is not None
        out_local = last_local.matmul(output_weight.t())
        out = out_local
        if self.output_scale != 1.0:
            out = out * self.output_scale
        return out

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._forward_from(self.hidden, self.output, X)

    def forward_init(self, X: torch.Tensor) -> torch.Tensor:
        return self._forward_from(self.init_hidden, self.init_output, X)

    def linearized_forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.use_linearized:
            raise RuntimeError("linearized_forward called with use_linearized=False")

        h0_prev_full = X
        dh_prev_full = torch.zeros_like(X)
        last_a0_local = None
        last_da_local = None

        for idx, (w0, wlin) in enumerate(zip(self.init_hidden, self.lin_hidden)):
            dw = wlin - w0
            z0_local = h0_prev_full.matmul(w0.t())
            dz_local = dh_prev_full.matmul(w0.t()) + h0_prev_full.matmul(dw.t())
            a0_local = torch.tanh(z0_local)
            da_local = (1.0 - a0_local.pow(2)) * dz_local
            h0_prev_full = a0_local
            dh_prev_full = da_local
            last_a0_local = a0_local
            last_da_local = da_local

        assert last_a0_local is not None and last_da_local is not None
        dw_out = self.lin_output - self.init_output
        f0_local = last_a0_local.matmul(self.init_output.t())
        df_local = last_da_local.matmul(self.init_output.t()) + last_a0_local.matmul(dw_out.t())
        out = f0_local + df_local
        if self.output_scale != 1.0:
            out = out * self.output_scale
        return out

    @torch.no_grad()
    def first_hidden_local(self, X: torch.Tensor) -> torch.Tensor:
        return torch.tanh(X.matmul(self.hidden[0].t()))

    @torch.no_grad()
    def param_norm0(self) -> float:
        local = torch.zeros((), device=self.device, dtype=torch.float64)
        for tensor in self.init_tensors_list():
            local += tensor.detach().pow(2).sum(dtype=torch.float64)
        return math.sqrt(float(local.item()))

    def clear_grads(self, which: str = "all") -> None:
        if which in {"all", "base"}:
            for param in self.base_parameters_list():
                param.grad = None
        if which in {"all", "lin"}:
            for param in self.lin_parameters_list():
                param.grad = None

    def named_state_tensors(self):
        for idx, param in enumerate(self.hidden):
            yield f"hidden_{idx:03d}", param
        yield "output", self.output

        if self.use_linearized:
            for idx, param in enumerate(self.lin_hidden):
                yield f"lin_hidden_{idx:03d}", param
            yield "lin_output", self.lin_output

        for idx, tensor in enumerate(self.init_hidden):
            yield f"init_hidden_{idx:03d}", tensor
        yield "init_output", self.init_output
