"""
Compatibility shim for old torch checkpoints.

Some old checkpoints pickle config objects as src.metric_checkpoints.Exp1Config.
Keep that class importable so torch.load can deserialize them. New code should
use ExpConfig/save_checkpoint/load_checkpoint from config.py.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Exp1Config:
    """Old config shape kept for unpickling old checkpoints."""
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
    # stats
    use_linearized: bool = True
    same_noise: bool = False
    track_jacobian: bool = True
    jac_probe_size: int = 10
    track_every: int = 10
    print_every: int = 100
    collect_feature_stats: bool = True

    def train_kwargs(self):
        # Legacy checkpoints should only use fields this old config actually knows about.
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
            epochs=self.epochs,
            eta=self.eta,
            regularization_scale=self.regularization_scale,
            use_linearized=self.use_linearized,
            same_noise=self.same_noise,
            track_jacobian=self.track_jacobian,
            jac_probe_size=self.jac_probe_size,
            track_every=self.track_every,
            print_every=self.print_every,
            collect_feature_stats=self.collect_feature_stats, 
        )
