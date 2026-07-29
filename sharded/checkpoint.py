from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re

import torch

from .config import SHARDED_METRIC_SCHEMA_VERSION, ShardedExpConfig
from .distributed import DistContext, barrier
from .model import ShardedMLP


def timestamped_checkpoint_path(ckpt_dir: Path, dataset: str, checkpoint_state: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_state == "sharded_state":
        return ckpt_dir / f"sharded_{dataset}_{stamp}"
    return ckpt_dir / f"sharded_{dataset}_{stamp}.pt"


def safe_path_part(text: str) -> str:
    text = text.replace("α", "alpha").replace("β", "beta")
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", text)
    return text.strip("_") or "run"


def save_state_shard(root: Path, label: str, seed: int, model: ShardedMLP, ctx: DistContext) -> str:
    rel_dir = Path("states") / safe_path_part(label) / f"seed_{seed}"
    shard_dir = root / rel_dir / f"rank_{ctx.rank:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tensor_files = {}
    for name, tensor in model.named_state_tensors():
        filename = f"{name}.pt"
        torch.save(tensor.detach().cpu(), shard_dir / filename)
        tensor_files[name] = filename

    manifest = {
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "local_m": model.local_m,
        "use_linearized": model.use_linearized,
        "tensor_files": tensor_files,
    }
    torch.save(manifest, shard_dir / "manifest.pt")
    barrier()
    return str(rel_dir)


def save_metrics_checkpoint(
    path: Path,
    results: dict,
    config: ShardedExpConfig,
    tracked_metrics: list[str],
    ctx: DistContext,
) -> None:
    if not ctx.is_rank0:
        return

    payload = {
        "type": "sharded_exp1",
        "metric_schema_version": SHARDED_METRIC_SCHEMA_VERSION,
        "tracked_metrics": tracked_metrics,
        "config": config,
        "config_dict": asdict(config),
        "world_size": ctx.world_size,
        "results": results,
    }

    if config.checkpoint_state == "sharded_state":
        path.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path / "results.pt")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
