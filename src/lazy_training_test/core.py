from __future__ import annotations

from dataclasses import replace, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable
import itertools

import numpy as np
import torch

from ..config import ExpConfig, RunOpts, save_checkpoint, load_checkpoint_with_metadata
from ..metric_config import METRIC_SCHEMA_VERSION, resolve_metric_plan
from .eta import _resolve_eta
from .training import MultiSeedWorkerArgs, TrainArgs, train_multiseed

Metrics = Dict[str, Any]
ResultsByLabel = Dict[str, Dict[int, Metrics]]

EXP1_CHECKPOINT_PREFIX = ""
_IGNORED_LEGACY_EXP_FIELDS = {"track_param_norms"}

# -------------------------------------------------------------------------- #
# ----------------------------- config helpers ----------------------------- #
# -------------------------------------------------------------------------- #

def _parse_beta(beta) -> float:
    if isinstance(beta, str) and beta in {".inf", "inf", "+inf"}:
        return np.inf
    return float(beta)


def _prepare_dataclass_kwargs(cls, kwargs: Mapping[str, Any], ignored_fields: set[str] | None = None) -> Dict[str, Any]:
    ignored_fields = ignored_fields or set()
    field_names = {f.name for f in fields(cls)}
    unknown = set(kwargs) - field_names - ignored_fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {cls.__name__} config field(s): {names}")
    return {k: v for k, v in kwargs.items() if k in field_names}


def _expand_path_arg(kwargs: Dict[str, Any], key: str) -> None:
    if key in kwargs and kwargs[key] is not None:
        kwargs[key] = Path(kwargs[key]).expanduser()


def build_from_config_mapping(cfg: dict) -> tuple[ExpConfig, RunOpts]:
    """
    cfg can be either:
      {
        "experiment": { ...fields of ExpConfig... },
        "run":        { ...fields of RunOpts... }
      }
    or a flat mapping of ExpConfig fields only.
    """
    exp_section = cfg.get("experiment")
    run_section = cfg.get("run")

    if exp_section is None:
        exp_kwargs = dict(cfg)
        run_kwargs = {}
    else:
        exp_kwargs = dict(exp_section)
        run_kwargs = dict(run_section or {})

    if "betas" in exp_kwargs:
        exp_kwargs["betas"] = [_parse_beta(b) for b in exp_kwargs["betas"]]
    exp_kwargs = _prepare_dataclass_kwargs(ExpConfig, exp_kwargs, ignored_fields=_IGNORED_LEGACY_EXP_FIELDS)

    exp_config = ExpConfig(**exp_kwargs)

    run_kwargs = _prepare_dataclass_kwargs(RunOpts, run_kwargs)
    _expand_path_arg(run_kwargs, "ckpt_dir")
    _expand_path_arg(run_kwargs, "load_ckpt_name")
    _expand_path_arg(run_kwargs, "plot_output_dir")
    run_opts = RunOpts(**run_kwargs)

    return exp_config, run_opts

def _apply_config_overrides(base: ExpConfig, override_src: ExpConfig, override_keys: Optional[Iterable[str]]) -> ExpConfig:
    if not override_keys:
        return base
    for k in override_keys:
        if k not in [
            "eta", 
            "eta_table_path",
            "regularization_scale", 
            "same_noise", 
            "noise_free_after_epoch",
            "early_stop_metric",
            "early_stop_goal",
            "early_stop_value",
            "jac_probe_size", 
            "device", 
            "print_every",
            "collect_feature_stats"
        ]:
            raise ValueError(f"Error: overriding {k} is not supported yet.")
    
    src_dict = override_src.__dict__
    kwargs = {k: src_dict[k] for k in override_keys if (k in src_dict and k != "epochs")}

    if not kwargs:
        return base
    return replace(base, **kwargs)

def _print_exp_config(
    exp_config: ExpConfig,
    prev_config: ExpConfig | None = None,
    override_keys: Iterable[str] | None = None,
) -> None:
    print("configuration:")

    curr = asdict(exp_config)
    prev = asdict(prev_config) if prev_config is not None else {}
    override_keys = set(override_keys or ())
    tracked_metrics_set = getattr(exp_config, "tracked_metrics", None) is not None
    legacy_metric_fields = {"track_jacobian", "collect_feature_stats"}

    for k, v in curr.items():
        legacy_note = None
        if k in legacy_metric_fields:
            if tracked_metrics_set:
                legacy_note = "legacy; ignored because tracked_metrics is set"
            else:
                legacy_note = "legacy; used only because tracked_metrics is omitted"

        if prev_config is not None and k in override_keys and k in prev and prev[k] != v:
            if legacy_note is not None:
                print(f"  {k}: {v} (previously {prev[k]}; {legacy_note})")
            else:
                print(f"  {k}: {v} (previously {prev[k]})")
        else:
            suffix = f" ({legacy_note})" if legacy_note is not None else ""
            print(f"  {k}: {v}{suffix}")

    metric_plan = _resolve_metric_plan_for_config(exp_config)
    print(f"  effective_tracked_metrics: {list(metric_plan.tracked_metrics)}")


def _resolve_metric_plan_for_config(config: ExpConfig):
    return resolve_metric_plan(
        tracked_metrics=getattr(config, "tracked_metrics", None),
        use_linearized=getattr(config, "use_linearized", True),
        track_jacobian=getattr(config, "track_jacobian", True),
        collect_feature_stats=getattr(config, "collect_feature_stats", True),
        early_stop_metric=getattr(config, "early_stop_metric", None),
    )


def _validate_resume_metrics(config: ExpConfig, ckpt_metadata: Mapping[str, Any]) -> ExpConfig:
    """
    Refuse resume if the checkpoint lacks metadata or the checkpoint config drifts.

    Loading/plotting old checkpoints is still allowed; only extending them is blocked.
    Resume uses the checkpoint config as the source of truth, so tracked_metrics
    from the incoming YAML is not treated as an override.
    """
    if not ckpt_metadata.get("has_metric_metadata", False):
        raise ValueError(
            "Cannot resume this checkpoint because it does not record tracked_metrics. "
            "Load/plot it normally, or start a new run with explicit metric metadata."
        )

    schema_version = ckpt_metadata.get("metric_schema_version")
    if schema_version != METRIC_SCHEMA_VERSION:
        raise ValueError(
            f"Cannot resume checkpoint with metric schema version {schema_version}; "
            f"expected {METRIC_SCHEMA_VERSION}."
        )

    ckpt_metrics = tuple(ckpt_metadata.get("tracked_metrics") or ())
    current_metrics = _resolve_metric_plan_for_config(config).tracked_metrics
    if ckpt_metrics != current_metrics:
        raise ValueError(
            "Cannot alter tracked metrics when resuming a checkpoint.\n"
            f"  checkpoint tracked_metrics: {list(ckpt_metrics)}\n"
            f"  current tracked_metrics:    {list(current_metrics)}"
        )

    if getattr(config, "tracked_metrics", None) is None:
        config = replace(config, tracked_metrics=list(ckpt_metrics))
    return config

# -------------------------------------------------------------------------- #
# ------------------------------- labels ----------------------------------- #
# -------------------------------------------------------------------------- #

def label_from_alpha_beta(alpha=None, beta=None, n=None):
    label = ""
    if alpha is not None:
        label += f"α={alpha:.0e} "
    if beta == np.inf:
        label += "inf"
    elif n is None:
        label += f"β={int(beta)}"
    else:
        label += f"β={int(beta//n)}n"
    return label

# -------------------------------------------------------------------------- #
# --------------------------- training primitive --------------------------- #
# -------------------------------------------------------------------------- #

def _train_single_alpha_beta(
    config: ExpConfig,
    alpha: float = 1.0,
    beta: float = np.inf,
    gpu_ids: Optional[List[int]] = None,
    resume_paths: Optional[Dict[int, str]] = None,
    epoch_offset: int = 0,
) -> Dict[int, Metrics]:
    metric_plan = _resolve_metric_plan_for_config(config)
    train_args = TrainArgs(
        eta=_resolve_eta(config, alpha=alpha, beta=beta),
        epochs=config.epochs,
        beta=beta,
        m=config.m,
        init_type=config.init_type,
        alpha=alpha,
        lam_fc1=config.lam_fc1,
        lam_fc2=config.lam_fc2,
        regularization_scale=config.regularization_scale,
        use_linearized=config.use_linearized,
        same_noise=config.same_noise,
        jac_probe_size=config.jac_probe_size,
        device=config.device,
        track_every=config.track_every,
        print_every=config.print_every,
        epoch_offset=epoch_offset,
        noise_free_after_epoch=config.noise_free_after_epoch,
        early_stop_metric=config.early_stop_metric,
        early_stop_goal=config.early_stop_goal,
        early_stop_value=config.early_stop_value,
        metric_plan=metric_plan,
    )
    worker_args = MultiSeedWorkerArgs(
        n=config.n,
        random_labels=config.random_labels,
        reserve_last=config.reserve_last,
        train=train_args,
        resume_paths=resume_paths,
    )
    return train_multiseed(config.dataset, config.seeds, worker_args, gpu_ids=gpu_ids)

# -------------------------------------------------------------------------- #
# ------------------ training & checkpoint orchestration ------------------- #
# -------------------------------------------------------------------------- #

# write to disk the info needed to resume training for a specific (alpha,beta) pair
# so that each worker can load to RAM its own contents without other stuff
def _write_base_ckpt_data_for_beta_to_disk(
    label: str,
    base_seed_metrics: Mapping[int, Mapping[str, Any]],
    resume_root: Path,
) -> Dict[int, str]:
    """
    populate a tmp directory with the data needed to resume training from some saved state
    (initial model parameters, last model (NN + linearized) parameters, randomness state)
    """
    dirr = resume_root / label.replace("α=", "alpha_").replace("β=", "beta_")
    dirr.mkdir(parents=True, exist_ok=True)

    resume_paths: Dict[int, str] = {}

    for seed, metrics in base_seed_metrics.items():
        payload = {
            "init_model_state_dict": metrics["init_model_state_dict"],
            "start_model_state_dict": metrics["model_state_dict"],
            "start_lin_params": metrics.get("lin_params_state"),
            "rng_state": metrics.get("rng_state"),
            "last_epoch": metrics.get("last_epoch"),
            "stopped_early": metrics.get("stopped_early", False),
        }
        path = dirr / f"seed_{seed}.pt"
        torch.save(payload, path)
        resume_paths[seed] = str(path)

    return resume_paths


def _iter_alpha_beta_pairs(alpha_range: Optional[List[float]], beta_range: Optional[List[float]]):
    betas = beta_range or []
    alphas = alpha_range or []
    if not alphas:
        return [(None, beta) for beta in betas]
    return list(itertools.product(alphas, betas))


def _labels_for_config(config: ExpConfig) -> List[str]:
    return [
        label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
        for alpha, beta in _iter_alpha_beta_pairs(config.alphas, config.betas)
    ]


def _train_over_range(
    config: ExpConfig,
    alpha_range: Optional[List[float]] = None,
    beta_range: Optional[List[float]] = None,
    gpu_ids: Optional[List[int]] = None,
    resume_root=None,
    base_results=None,
    epoch_offset: int = 0,
) -> Dict[str, Metrics]: 
    results: ResultsByLabel = {}

    for alpha, beta in _iter_alpha_beta_pairs(alpha_range, beta_range):
        label = label_from_alpha_beta(alpha=alpha, beta=beta, n=config.n)
        resume_paths = None
        if base_results is not None:
            resume_paths = _write_base_ckpt_data_for_beta_to_disk(label, base_results[label], resume_root)

        pair_kwargs = {"beta": beta} if alpha is None else {"alpha": alpha, "beta": beta}
        results[label] = _train_single_alpha_beta(
            config,
            gpu_ids=gpu_ids,
            resume_paths=resume_paths,
            epoch_offset=epoch_offset,
            **pair_kwargs,
        )
    
    return results

def _merge_metrics(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    base_keys = set(base.keys())
    extra_keys = set(extra.keys())

    # allow new-style metrics to exist only in the resumed run
    allowed_new_keys = {"epoch_hist", "last_epoch", "stopped_early"}
    if base_keys != extra_keys:
        diff = base_keys ^ extra_keys
        unexpected = diff - allowed_new_keys
        if unexpected:
            raise ValueError(f"Metric keys differ between base and extra runs: {unexpected}")

    # if the resumed run produced no new history at all, keep base metrics
    has_new_hist = False
    for k in extra_keys:
        if not k.endswith("_hist"):
            continue
        v = extra[k]
        if v is None:
            continue
        if isinstance(v, list) and len(v) > 0:
            has_new_hist = True
            break
        if torch.is_tensor(v) and v.numel() > 0:
            has_new_hist = True
            break
        if isinstance(v, np.ndarray) and v.size > 0:
            has_new_hist = True
            break
    if not has_new_hist:
        return dict(base)

    merged: Dict[str, Any] = {}

    for k in base_keys:
        b = base[k]
        e = extra[k]

        if k.endswith("_hist"):
            if b is None:
                merged[k] = e
            elif e is None:
                merged[k] = b
            elif isinstance(b, list) and isinstance(e, list):
                merged[k] = b + e
            elif torch.is_tensor(b) and torch.is_tensor(e):
                merged[k] = torch.cat([b, e], dim=0)
            elif isinstance(b, np.ndarray) and isinstance(e, np.ndarray):
                merged[k] = np.concatenate([b, e], axis=0)
            else:
                raise ValueError("Metric histogram concatenation failed.")
        elif k == "init_model_state_dict":
            merged[k] = b
        elif k in {"model_state_dict", "lin_params_state"}:
            if b is not None and e is None:
                raise ValueError(f"New run is missing {k} while base run has it.")
            merged[k] = e if e is not None else b
        else:
            merged[k] = e

    return merged

def _infer_last_epoch_from_results(results: ResultsByLabel, fallback_epochs: int) -> int:
    last_epochs: List[int] = []
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

def resume_from_ckpt(
    base_results: ResultsByLabel,
    config: ExpConfig,
    new_epochs: int,
    gpu_ids: List[int],
    tmp_dir: Path,
) -> Tuple[ResultsByLabel, ExpConfig]:
    # some validation
    base_effective_epochs = _infer_last_epoch_from_results(base_results, config.epochs)
    if new_epochs <= base_effective_epochs:
        raise ValueError(f"new_epochs ({new_epochs}) must be > existing epochs ({base_effective_epochs})")
    expected_labels = _labels_for_config(config)
    if set(base_results.keys()) != set(expected_labels):
        raise ValueError("Checkpoint alpha/betas do not match config.alphas/config.betas.")

    # create tmp directory for passing data to child processes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_root = tmp_dir / f"exp1_resume_{timestamp}"
    resume_root.mkdir(parents=True, exist_ok=True)

    # train
    print(f"extending to a new total of {new_epochs} epochs...")
    extra_cfg = replace(config, epochs=new_epochs)
    new_results: ResultsByLabel = \
        _train_over_range(extra_cfg, config.alphas, config.betas, gpu_ids, resume_root, base_results, base_effective_epochs)

    # merge base + new
    merged_results: ResultsByLabel = {}
    for label in expected_labels:
        base_seed_metrics  = base_results[label]
        extra_seed_metrics = new_results[label]

        merged_seed_metrics: Dict[int, Metrics] = {}
        for seed in config.seeds:
            merged_seed_metrics[seed] = _merge_metrics(base_seed_metrics[seed], extra_seed_metrics[seed])
        merged_results[label] = merged_seed_metrics

    final_epochs = _infer_last_epoch_from_results(merged_results, new_epochs)
    new_config = replace(config, epochs=final_epochs)
    
    return merged_results, new_config

# -------------------------------------------------------------------------- #
# ------------------------------ "public API" ------------------------------ #
# -------------------------------------------------------------------------- #

def run_exp(config: ExpConfig, run_opts: RunOpts, gpu_ids: List[int],) -> Tuple[ResultsByLabel, ExpConfig]:
    run_opts.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if run_opts.load_ckpt:
        # load ckpt
        load_ckpt_path = run_opts.ckpt_dir / run_opts.load_ckpt_name
        base_results, base_config, ckpt_metadata = load_checkpoint_with_metadata(str(load_ckpt_path))
        print(f"Loaded checkpoint: {load_ckpt_path}")
        
        # resume run if needed
        if run_opts.resume_from_ckpt:
            if run_opts.new_total_epochs is None:
                raise ValueError("new_total_epochs must be set when resume_from_ckpt is True")
            
            # Resume starts from the checkpoint config. The incoming YAML only changes fields
            # explicitly listed in run.config_overrides, and tracked_metrics is intentionally
            # not an overrideable field.
            exp_config = _apply_config_overrides(base_config, config, run_opts.config_overrides)
            exp_config = _validate_resume_metrics(exp_config, ckpt_metadata)

            override_keys = (run_opts.config_overrides or {})
            _print_exp_config(exp_config, prev_config=base_config, override_keys=override_keys)

            results, exp_config = resume_from_ckpt(
                base_results=base_results,
                config=exp_config,
                new_epochs=run_opts.new_total_epochs,
                gpu_ids=gpu_ids,
                tmp_dir=run_opts.ckpt_dir,
            )
        # otherwise, go with existing ckpt data
        else:
            # Old checkpoints may not have metric metadata; loading/plotting them is still allowed.
            exp_config = base_config
            _print_exp_config(exp_config)
            results = base_results 
    else:
        exp_config = config
        _print_exp_config(exp_config)
        results = _train_over_range(config, config.alphas, config.betas, gpu_ids)
        final_epochs = _infer_last_epoch_from_results(results, config.epochs)
        exp_config = replace(config, epochs=final_epochs)

    if run_opts.save_ckpt:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = run_opts.ckpt_dir / f"{EXP1_CHECKPOINT_PREFIX}_{exp_config.dataset}_{timestamp}.pt"
        save_checkpoint(str(ckpt_path), results, exp_config)
        print(f"Saved checkpoint: {ckpt_path}")

    return results, exp_config

# required for plotting the ICML-deadline ckpt
def infer_effective_track_every(results: ResultsByLabel, config: ExpConfig) -> int:
    beta_key = next(iter(results))
    seed_key = next(iter(results[beta_key]))
    hist = results[beta_key][seed_key]["train_loss_hist"]
    L = len(hist)

    eff = config.track_every
    if L <= 1:
        return eff

    E = config.epochs
    low = (E - 1) // L + 1
    high = (E - 1) // (L - 1)

    if low <= high and not low <= eff <= high:
        eff = low

    return eff
