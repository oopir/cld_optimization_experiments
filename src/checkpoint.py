from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import random
import re

import numpy as np
import torch

from .config import METRIC_SCHEMA_VERSION, ExpConfig, resolve_metric_plan
from .model import MLP

CHECKPOINT_TYPE = "single_device_deep_exp1"
CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = {1, CHECKPOINT_FORMAT_VERSION}
LEGACY_CHECKPOINT_TYPE = "sharded_exp1"
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
class LoadedCheckpoint:
    path: Path
    payload_path: Path
    results: dict[str, Any]
    states: dict[str, Any] | None
    config: ExpConfig
    metadata: dict[str, Any]


@dataclass
class ResumeState:
    state_tensors: dict[str, torch.Tensor]
    rng_state: dict[str, Any]
    last_epoch: int | None = None
    stopped_early: bool = False


def timestamped_checkpoint_path(ckpt_dir: Path, first_seed: int, checkpoint_state: str = "metrics_only") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "" if checkpoint_state == "resumable_state" else "_metrics.pt"
    candidate = ckpt_dir / f"ckpt_{stamp}_{first_seed}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = ckpt_dir / f"ckpt_{stamp}_{first_seed}_{counter:02d}{suffix}"
        counter += 1
    return candidate


def safe_path_part(text: str) -> str:
    text = text.replace("α", "alpha").replace("β", "beta")
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", text)
    return text.strip("_") or "run"


def _checkpoint_payload_path(path: Path) -> Path:
    path = Path(path).expanduser()
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


def capture_rng_state(model: MLP, device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu(),
        "torch_cuda": None,
        "noise_gen": model.noise_gen.get_state().detach().cpu(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device=device).detach().cpu()
    return state


def _validate_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    required = {"python", "numpy", "torch_cpu", "noise_gen"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            "Checkpoint RNG state is incomplete; "
            f"missing: {', '.join(missing)}."
        )
    if device.type == "cuda" and state.get("torch_cuda") is None:
        raise ValueError("Cannot resume CUDA run because checkpoint has no CUDA RNG state.")


def restore_rng_state(model: MLP, state: Mapping[str, Any], device: torch.device) -> None:
    _validate_rng_state(state, device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=device)
    model.noise_gen.set_state(state["noise_gen"].cpu())


def capture_model_state(model: MLP, metrics: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    tensors = {}
    for name, tensor in model.named_state_tensors():
        tensors[name] = tensor.detach().cpu()
    return {
        "tensors": tensors,
        "rng_state": capture_rng_state(model, device),
        "last_epoch": metrics.get("last_epoch"),
        "stopped_early": metrics.get("stopped_early", False),
    }


def save_model_state(
    root: Path,
    label: str,
    seed: int,
    model: MLP,
    metrics: Mapping[str, Any],
    device: torch.device,
) -> str:
    rel_path = Path("states") / safe_path_part(label) / f"seed_{int(seed)}.pt"
    state_path = root / rel_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture_model_state(model, metrics, device), state_path)
    return str(rel_path)


@torch.no_grad()
def apply_resume_state_to_model(model: MLP, resume_state: ResumeState, device: torch.device) -> None:
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
        raise ValueError("Resume tensors do not match current model (" + "; ".join(details) + ").")

    for name, target in targets.items():
        source = resume_state.state_tensors[name]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Resume tensor {name!r} has shape {tuple(source.shape)}, "
                f"expected {tuple(target.shape)}."
            )
        target.copy_(source.to(device=device, dtype=target.dtype))

    restore_rng_state(model, resume_state.rng_state, device)


def load_checkpoint_with_metadata(path: Path) -> LoadedCheckpoint:
    payload_path = _checkpoint_payload_path(path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {payload_path}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload_type = payload.get("type")
    if payload_type == LEGACY_CHECKPOINT_TYPE:
        raise ValueError("Legacy sharded_exp1 checkpoints are unsupported; start a new single-device run.")
    if payload_type != CHECKPOINT_TYPE:
        raise ValueError(f"Expected {CHECKPOINT_TYPE!r}, got {payload_type!r}.")
    if payload.get("checkpoint_format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError("Unsupported single-device checkpoint format version.")
    config = payload.get("config")
    if config is None:
        config = ExpConfig(**payload["config_dict"])

    metadata = {
        "metric_schema_version": payload.get("metric_schema_version"),
        "tracked_metrics": payload.get("tracked_metrics"),
        "has_states": payload.get("states") is not None,
    }
    return LoadedCheckpoint(
        path=payload_path.parent if payload_path.name == "results.pt" else payload_path,
        payload_path=payload_path,
        results=payload["results"],
        states=payload.get("states"),
        config=config,
        metadata=metadata,
    )


def save_checkpoint(
    path: Path,
    results: dict,
    states: dict[str, Any] | None,
    config: ExpConfig,
    tracked_metrics: list[str],
) -> None:
    payload_path = path / "results.pt" if states is not None else path
    payload = {
        "type": CHECKPOINT_TYPE,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "tracked_metrics": tracked_metrics,
        "config": config,
        "config_dict": asdict(config),
        "results": results,
    }
    if states is not None:
        payload["states"] = states
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)


def apply_config_overrides(
    base: ExpConfig,
    override_src: ExpConfig,
    override_keys: Iterable[str] | None,
) -> ExpConfig:
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


def load_resume_state(loaded: LoadedCheckpoint, label: str, seed: int) -> ResumeState:
    if loaded.states is None:
        raise ValueError("This is a metrics-only checkpoint and cannot be resumed.")
    try:
        state = loaded.states[label][int(seed)]
    except KeyError as exc:
        raise ValueError(
            f"Checkpoint is missing resumable state for {label!r}, seed={seed}."
        ) from exc
    if isinstance(state, (str, Path)):
        state = torch.load(loaded.path / state, map_location="cpu", weights_only=False)
    return ResumeState(
        state["tensors"],
        state["rng_state"],
        state.get("last_epoch"),
        bool(state.get("stopped_early", False)),
    )


def validate_resume_request(
    config: ExpConfig,
    loaded: LoadedCheckpoint,
    expected_labels: Iterable[str],
    new_total_epochs: int,
) -> int:
    if loaded.metadata.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("Checkpoint metric schema is incompatible with this engine.")

    checkpoint_metrics = tuple(loaded.metadata.get("tracked_metrics") or ())
    current_metrics = resolve_metric_plan(config).tracked_metrics
    if checkpoint_metrics != current_metrics:
        raise ValueError(
            "Cannot alter tracked metrics when resuming a checkpoint.\n"
            f"  checkpoint tracked_metrics: {list(checkpoint_metrics)}\n"
            f"  current tracked_metrics:    {list(current_metrics)}"
        )
    if set(loaded.results) != set(expected_labels):
        raise ValueError("Checkpoint alpha/betas do not match the requested config.")
    for label in expected_labels:
        for seed in config.seeds:
            load_resume_state(loaded, label, int(seed))
    base_effective_epochs = infer_last_epoch_from_results(loaded.results, config.epochs)
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
    allowed_key_diffs = {"epoch_hist", "last_epoch", "stopped_early", "tracked_metrics"}
    unexpected = (base_keys ^ extra_keys) - allowed_key_diffs
    if unexpected:
        raise ValueError(f"Metric keys differ between base and extra runs: {sorted(unexpected)}")

    if not _has_new_history(extra):
        merged = dict(base)
        for key in ("last_epoch", "stopped_early"):
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
        elif key in {"last_epoch", "stopped_early"}:
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
