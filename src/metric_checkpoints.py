from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class Exp1Config:
    # parallelization
    seeds: list = field(default_factory=lambda: [0])
    device: str = "cpu"
    # data
    dataset: str = "digits"
    n: int = 10
    random_labels: bool = False
    # model
    m: int = 1
    init_type: str = "standard"
    # training
    epochs: int = 1
    eta: float  = 1.0
    betas: list = field(default_factory=lambda: [1.0])
    regularization_scale: float = 1.0
    lam_fc1: Optional[float] = None
    lam_fc2: Optional[float] = None
    # stats
    use_linearized: bool = True
    same_noise: bool = False
    track_jacobian: bool = True
    jac_probe_size: int = 10
    track_every: int = 10
    print_every: int = 100

    def train_kwargs(self):
        return dict(
            seeds=self.seeds,
            device=self.device,
            dataset=self.dataset,
            n=self.n,
            random_labels=self.random_labels,
            m=self.m,
            init_type=self.init_type,
            lam_fc1=self.lam_fc1,
            lam_fc2=self.lam_fc2,
            epochs=self.epochs,
            eta=self.eta,
            regularization_scale=self.regularization_scale,
            use_linearized=self.use_linearized,
            same_noise=self.same_noise,
            track_jacobian=self.track_jacobian,
            jac_probe_size=self.jac_probe_size,
            track_every=self.track_every,
            print_every=self.print_every
        )


def save_exp1_checkpoint(path, results, config: Exp1Config):
    payload = {"type": "exp1", "config": config, "results": results}
    torch.save(payload, path)

def load_exp1_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)

    payload_type = payload.get("type", "exp1") # 2nd argument "tolerates" old ckpts w/o "type" field
    if payload_type != "exp1":
        raise ValueError(f"Unexpected checkpoint type: {payload_type}")

    return payload["results"], payload["config"]
