import torch
from dataclasses import dataclass

@dataclass
class Exp1Config:
    seeds: list
    n: int
    random_labels: bool
    betas: list
    epochs: int
    eta: float
    m: int
    init_type: str = "standard"
    regularization_scale: float = 1.0
    use_linearized: bool = True
    track_jacobian: bool = True
    jac_probe_size: int = 10
    device: str = "cpu"
    track_every: int = 1
    print_every: int = 100
    
    def train_kwargs(self):
        return dict(
            seeds=self.seeds,
            n=self.n,
            random_labels=self.random_labels,
            epochs=self.epochs,
            eta=self.eta,
            m=self.m,
            init_type=self.init_type,
            regularization_scale=self.regularization_scale,
            use_linearized=self.use_linearized,
            track_jacobian=self.track_jacobian,
            jac_probe_size=self.jac_probe_size,
            device=self.device,
            track_every=self.track_every,
            print_every=self.print_every
        )


def save_exp1_checkpoint(path, results, config: Exp1Config):
    payload = {"type": "exp1", "config": config, "results": results}
    torch.save(payload, path)

def load_exp1_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False))
    return payload["results"], payload["config"]



