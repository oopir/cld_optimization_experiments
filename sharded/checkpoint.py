from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import random
import re

import numpy as np
import torch

from .config import SHARDED_METRIC_SCHEMA_VERSION, ShardedExpConfig, resolve_metric_plan
from .distributed import DistContext, barrier
from .model import ShardedMLP


SHARDED_CHECKPOINT_FORMAT_VERSION = 2
SHARDED_CHECKPOINT_TYPE = "sharded_exp1"
_RESUME_OVERRIDE_FIELDS = {
    "eta",
    "eta_mode",
    "eta_table_path",
    "eta_default",
    "regularization_scale",
    "same_noise",
    "noise_free_after_epoch",
    "early_stop_metric",
    "early_stop_goal",
    "early_stop_value",
    "jac_probe_size",
    "device",
    "gpu_indices",
    "print_every",
}


@dataclass(frozen=True)
class LoadedShardedCheckpoint:
    path: Path
    payload_path: Path
    results: dict[str, Any]
    config: ShardedExpConfig
    metadata: dict[str, Any]


@dataclass
class ShardedResumeState:
    state_tensors: dict[str, torch.Tensor]
    rng_state: dict[str, Any]
    last_epoch: int | None = None
    stopped_early: bool = False


def timestamped_checkpoint_path(ckpt_dir: Path, dataset: str, checkpoint_state: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_state == "sharded_state":
        candidate = ckpt_dir / f"sharded_{dataset}_{stamp}"
        suffix = ""
    else:
        candidate = ckpt_dir / f"sharded_{dataset}_{stamp}.pt"
        suffix = ".pt"

    if not candidate.exists():
        return candidate

    stem = candidate.name[:-len(suffix)] if suffix else candidate.name
    counter = 1
    while True:
        next_candidate = ckpt_dir / f"{stem}_{counter:02d}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def safe_path_part(text: str) -> str:
    text = text.replace("α", "alpha").replace("β", "beta")
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", text)
    return text.strip("_") or "run"


def _checkpoint_payload_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir() or (path / "results.pt").is_file():
        return path / "results.pt"
    return path


def resolve_checkpoint_path(ckpt_dir: Path, load_ckpt_name: Path | None) -> Path:
    if load_ckpt_name is None:
        raise ValueError("load_ckpt_name must be set when load_ckpt is True.")

    path = Path(load_ckpt_name).expanduser()
    if path.is_absolute():
        return path
    return ckpt_dir.expanduser() / path


def capture_rng_state(ctx: DistContext, model: ShardedMLP) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu(),
        "torch_cuda": None,
        "noise_gen": model.noise_gen.get_state().detach().cpu(),
    }
    if ctx.device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device=ctx.device).detach().cpu()
    return state


def _validate_rng_state_for_resume(state: Mapping[str, Any], ctx: DistContext) -> None:
    required = {"python", "numpy", "torch_cpu", "noise_gen"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            "Cannot resume checkpoint because its RNG state is incomplete; "
            f"missing: {', '.join(missing)}."
        )
    if ctx.device.type == "cuda" and state.get("torch_cuda") is None:
        raise ValueError("Cannot resume CUDA run because checkpoint has no CUDA RNG state.")


def restore_rng_state(ctx: DistContext, model: ShardedMLP, state: Mapping[str, Any]) -> None:
    _validate_rng_state_for_resume(state, ctx)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if ctx.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=ctx.device)
    model.noise_gen.set_state(state["noise_gen"].cpu())


def save_state_shard(
    root: Path,
    label: str,
    seed: int,
    model: ShardedMLP,
    ctx: DistContext,
    metrics: Mapping[str, Any],
) -> str:
    rel_dir = Path("states") / safe_path_part(label) / f"seed_{seed}"
    shard_dir = root / rel_dir / f"rank_{ctx.rank:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tensor_files = {}
    for name, tensor in model.named_state_tensors():
        filename = f"{name}.pt"
        torch.save(tensor.detach().cpu(), shard_dir / filename)
        tensor_files[name] = filename

    rng_file = "rng_state.pt"
    torch.save(capture_rng_state(ctx, model), shard_dir / rng_file)

    metrics_file = "metrics.pt"
    metrics_payload = dict(metrics)
    metrics_payload["state_shard_dir"] = str(rel_dir)
    torch.save(metrics_payload, shard_dir / metrics_file)

    manifest = {
        "checkpoint_format_version": SHARDED_CHECKPOINT_FORMAT_VERSION,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "local_m": model.local_m,
        "use_linearized": model.use_linearized,
        "tensor_files": tensor_files,
        "rng_file": rng_file,
        "metrics_file": metrics_file,
        "last_epoch": metrics.get("last_epoch"),
        "stopped_early": metrics.get("stopped_early", False),
    }
    torch.save(manifest, shard_dir / "manifest.pt")
    barrier()
    return str(rel_dir)


def load_state_shard(root: Path, metrics: Mapping[str, Any], ctx: DistContext) -> ShardedResumeState:
    rel_dir = metrics.get("state_shard_dir")
    if not rel_dir:
        raise ValueError(
            "Cannot resume checkpoint because a seed is missing state_shard_dir. "
            "Use a checkpoint saved with checkpoint_state='sharded_state'."
        )

    shard_dir = root / rel_dir / f"rank_{ctx.rank:03d}"
    manifest_path = shard_dir / "manifest.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing rank shard manifest: {manifest_path}")

    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if manifest.get("checkpoint_format_version") != SHARDED_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Cannot resume checkpoint because its shard format is unsupported. "
            "Previously generated sharded checkpoints are not resumable."
        )
    if int(manifest.get("world_size", -1)) != int(ctx.world_size):
        raise ValueError(
            "Cannot resume checkpoint with a different world_size: "
            f"checkpoint={manifest.get('world_size')}, current={ctx.world_size}."
        )
    if int(manifest.get("rank", -1)) != int(ctx.rank):
        raise ValueError(
            f"Rank shard mismatch for {shard_dir}: manifest rank={manifest.get('rank')}, "
            f"current rank={ctx.rank}."
        )

    rng_file = manifest.get("rng_file")
    if rng_file is None:
        raise ValueError("Cannot resume checkpoint because its shard has no RNG state.")
    rng_state = torch.load(shard_dir / rng_file, map_location="cpu", weights_only=False)
    _validate_rng_state_for_resume(rng_state, ctx)

    state_tensors = {}
    tensor_files = manifest.get("tensor_files") or {}
    for name, filename in tensor_files.items():
        state_tensors[name] = torch.load(
            shard_dir / filename,
            map_location=ctx.device,
            weights_only=False,
        )

    return ShardedResumeState(
        state_tensors=state_tensors,
        rng_state=rng_state,
        last_epoch=manifest.get("last_epoch"),
        stopped_early=bool(manifest.get("stopped_early", False)),
    )


@torch.no_grad()
def apply_resume_state_to_model(model: ShardedMLP, resume_state: ShardedResumeState) -> None:
    targets = {name: tensor for name, tensor in model.named_state_tensors()}
    expected = set(targets)
    actual = set(resume_state.state_tensors)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing tensors: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected tensors: {', '.join(extra)}")
        raise ValueError("Resume shard tensors do not match current model (" + "; ".join(details) + ").")

    for name, target in targets.items():
        source = resume_state.state_tensors[name]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Resume tensor {name!r} has shape {tuple(source.shape)}, "
                f"expected {tuple(target.shape)}."
            )
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    restore_rng_state(model.ctx, model, resume_state.rng_state)


def load_checkpoint_with_metadata(path: Path) -> LoadedShardedCheckpoint:
    payload_path = _checkpoint_payload_path(path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"Checkpoint payload not found: {payload_path}")

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload_type = payload.get("type")
    if payload_type != SHARDED_CHECKPOINT_TYPE:
        raise ValueError(f"Expected {SHARDED_CHECKPOINT_TYPE!r} checkpoint, got {payload_type!r}.")

    config = payload.get("config")
    if config is None:
        config = ShardedExpConfig(**payload["config_dict"])

    metadata = {
        "checkpoint_format_version": payload.get("checkpoint_format_version"),
        "metric_schema_version": payload.get("metric_schema_version"),
        "tracked_metrics": payload.get("tracked_metrics"),
        "has_metric_metadata": "tracked_metrics" in payload,
        "world_size": payload.get("world_size"),
        "checkpoint_state": getattr(config, "checkpoint_state", None),
    }
    return LoadedShardedCheckpoint(
        path=payload_path.parent,
        payload_path=payload_path,
        results=payload["results"],
        config=config,
        metadata=metadata,
    )


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
        "type": SHARDED_CHECKPOINT_TYPE,
        "checkpoint_format_version": SHARDED_CHECKPOINT_FORMAT_VERSION,
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


def apply_config_overrides(
    base: ShardedExpConfig,
    override_src: ShardedExpConfig,
    override_keys: Iterable[str] | None,
) -> ShardedExpConfig:
    if not override_keys:
        return base

    unsupported = sorted(set(override_keys) - _RESUME_OVERRIDE_FIELDS)
    if unsupported:
        raise ValueError("Unsupported resume config override(s): " + ", ".join(unsupported))

    src_dict = override_src.__dict__
    kwargs = {key: src_dict[key] for key in override_keys if key in src_dict}
    if not kwargs:
        return base
    return replace(base, **kwargs)


def infer_last_epoch_from_results(results: Mapping[str, Mapping[int, Mapping[str, Any]]], fallback_epochs: int) -> int:
    last_epochs = []
    for per_label in results.values():
        for metrics in per_label.values():
            value = metrics.get("last_epoch")
            if value is None:
                continue
            try:
                last_epochs.append(int(value))
            except (TypeError, ValueError):
                continue
    if last_epochs:
        return max(last_epochs)
    return fallback_epochs


def validate_resume_request(
    config: ShardedExpConfig,
    metadata: Mapping[str, Any],
    results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    expected_labels: Iterable[str],
    world_size: int,
    new_total_epochs: int,
) -> int:
    if config.checkpoint_state != "sharded_state":
        raise ValueError("Sharded resume requires checkpoint_state='sharded_state'.")
    if not metadata.get("has_metric_metadata", False):
        raise ValueError("Cannot resume checkpoint because it does not record tracked_metrics.")
    if metadata.get("metric_schema_version") != SHARDED_METRIC_SCHEMA_VERSION:
        raise ValueError(
            "Cannot resume checkpoint with metric schema version "
            f"{metadata.get('metric_schema_version')}; expected {SHARDED_METRIC_SCHEMA_VERSION}."
        )
    if int(metadata.get("world_size", -1)) != int(world_size):
        raise ValueError(
            "Cannot resume checkpoint with a different world_size: "
            f"checkpoint={metadata.get('world_size')}, current={world_size}."
        )

    checkpoint_metrics = tuple(metadata.get("tracked_metrics") or ())
    current_metrics = resolve_metric_plan(config).tracked_metrics
    if checkpoint_metrics != current_metrics:
        raise ValueError(
            "Cannot alter tracked metrics when resuming a checkpoint.\n"
            f"  checkpoint tracked_metrics: {list(checkpoint_metrics)}\n"
            f"  current tracked_metrics:    {list(current_metrics)}"
        )

    expected_label_set = set(expected_labels)
    if set(results.keys()) != expected_label_set:
        raise ValueError("Checkpoint alpha/betas do not match config.alphas/config.betas.")

    for label in expected_label_set:
        per_seed = results[label]
        missing_seeds = [int(seed) for seed in config.seeds if int(seed) not in per_seed]
        if missing_seeds:
            raise ValueError(f"Checkpoint label {label!r} is missing seed(s): {missing_seeds}.")
        for seed in config.seeds:
            if not per_seed[int(seed)].get("state_shard_dir"):
                raise ValueError(
                    f"Checkpoint label {label!r}, seed {int(seed)} is missing state_shard_dir."
                )

    base_effective_epochs = infer_last_epoch_from_results(results, config.epochs)
    if int(new_total_epochs) <= base_effective_epochs:
        raise ValueError(
            f"new_total_epochs ({new_total_epochs}) must be > existing epochs ({base_effective_epochs})."
        )
    return base_effective_epochs


def _has_new_history(metrics: Mapping[str, Any]) -> bool:
    for key, value in metrics.items():
        if not key.endswith("_hist") or value is None:
            continue
        if isinstance(value, list) and len(value) > 0:
            return True
        if torch.is_tensor(value) and value.numel() > 0:
            return True
        if isinstance(value, np.ndarray) and value.size > 0:
            return True
    return False


def merge_metrics(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    base_keys = set(base)
    extra_keys = set(extra)
    allowed_key_diffs = {"epoch_hist", "last_epoch", "stopped_early", "state_shard_dir", "tracked_metrics"}
    unexpected = (base_keys ^ extra_keys) - allowed_key_diffs
    if unexpected:
        raise ValueError(f"Metric keys differ between base and extra runs: {sorted(unexpected)}")

    if not _has_new_history(extra):
        merged = dict(base)
        for key in ("last_epoch", "stopped_early", "state_shard_dir"):
            if key in extra:
                merged[key] = extra[key]
        return merged

    merged: dict[str, Any] = {}
    for key in base_keys | extra_keys:
        if key not in base:
            merged[key] = extra[key]
            continue
        if key not in extra:
            merged[key] = base[key]
            continue

        base_value = base[key]
        extra_value = extra[key]
        if key.endswith("_hist"):
            if base_value is None:
                merged[key] = extra_value
            elif extra_value is None:
                merged[key] = base_value
            elif isinstance(base_value, list) and isinstance(extra_value, list):
                merged[key] = base_value + extra_value
            elif torch.is_tensor(base_value) and torch.is_tensor(extra_value):
                merged[key] = torch.cat([base_value, extra_value], dim=0)
            elif isinstance(base_value, np.ndarray) and isinstance(extra_value, np.ndarray):
                merged[key] = np.concatenate([base_value, extra_value], axis=0)
            else:
                raise ValueError(f"Metric histogram concatenation failed for {key!r}.")
        elif key == "tracked_metrics":
            if list(base_value) != list(extra_value):
                raise ValueError("tracked_metrics changed during resume.")
            merged[key] = list(base_value)
        elif key in {"last_epoch", "stopped_early", "state_shard_dir"}:
            merged[key] = extra_value
        else:
            merged[key] = extra_value

    return merged


def merge_results(
    base_results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    extra_results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Iterable[int],
    expected_labels: Iterable[str],
) -> dict[str, dict[int, dict[str, Any]]]:
    merged: dict[str, dict[int, dict[str, Any]]] = {}
    for label in expected_labels:
        merged[label] = {}
        for seed in seeds:
            seed_int = int(seed)
            merged[label][seed_int] = merge_metrics(
                base_results[label][seed_int],
                extra_results[label][seed_int],
            )
    return merged
