from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import torch

from .metric_checkpoints import Exp1Config, save_exp1_checkpoint, load_exp1_checkpoint

@dataclass
class ExpConfig:
    # parallelization
    seeds: list = field(default_factory=lambda: [0])
    device: str = "cpu"
    # data
    dataset: str = "digits"
    n: int = 10
    random_labels: bool = False
    reserve_last: int = 1000
    # model
    m: int = 1
    init_type: str = "standard"
    # training
    epochs: int = 1
    eta: float  = 1.0
    eta_mode: str = "scalar"               # "scalar", "per_beta", "per_alpha_beta"
    eta_table_path: Optional[str] = None   # path to YAML table
    eta_default: Optional[float] = None    # fallback if key missing (defaults to eta if None)
    betas: list = field(default_factory=lambda: [])
    alphas: list = field(default_factory=lambda: [])
    regularization_scale: float = 1.0
    lam_fc1: Optional[float] = None
    lam_fc2: Optional[float] = None
    noise_free_after_epoch: Optional[int] = None
    # early stopping
    early_stop_metric: Optional[str] = None
    early_stop_goal: str = "min"
    early_stop_value: Optional[float] = None
    # stats
    use_linearized: bool = True
    same_noise: bool = False
    track_jacobian: bool = True
    jac_probe_size: int = 10
    track_every: int = 10
    print_every: int = 100
    collect_feature_stats: bool = True

    def train_kwargs(self):
        return dict(
            seeds=self.seeds,
            device=self.device,
            dataset=self.dataset,
            n=self.n,
            random_labels=self.random_labels,
            reserve_last=self.reserve_last,
            m=self.m,
            init_type=self.init_type,
            lam_fc1=self.lam_fc1,
            lam_fc2=self.lam_fc2,
            noise_free_after_epoch=getattr(self, "noise_free_after_epoch", None),
            epochs=self.epochs,
            eta=self.eta,
            regularization_scale=self.regularization_scale,
            early_stop_metric=self.early_stop_metric,
            early_stop_goal=self.early_stop_goal,
            early_stop_value=self.early_stop_value,
            use_linearized=self.use_linearized,
            same_noise=self.same_noise,
            track_jacobian=self.track_jacobian,
            jac_probe_size=self.jac_probe_size,
            track_every=self.track_every,
            print_every=self.print_every,
            collect_feature_stats=self.collect_feature_stats, 
        )


@dataclass
class RunOpts:
    ckpt_dir:         Path
    save_ckpt:        bool = False # for saving progress after training
    load_ckpt:        bool = False # for plotting/extending an existing ckpt  
    load_ckpt_name:   Optional[Path] = None
    resume_from_ckpt: bool = False
    new_total_epochs: Optional[int] = None
    config_overrides: Optional[List[str]] = None


def save_checkpoint(path, results, config: ExpConfig):
    payload = {"type": "exp1", "config": config, "results": results}
    torch.save(payload, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)

    payload_type = payload.get("type", "exp1") # 2nd argument "tolerates" old ckpts w/o "type" field
    if payload_type != "exp1":
        raise ValueError(f"Unexpected checkpoint type: {payload_type}")

    return payload["results"], payload["config"]
