#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root:\n"
        f"  cd {REPO_ROOT}\n"
        "  python3 scripts/merge_sharded_metric_checkpoints.py --out merged.pt tmp0.log tmp1.log ..."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

SAVED_RE = re.compile(r"Saved checkpoint:\s*(.+)")


def checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix != ".log":
        return path

    matches = SAVED_RE.findall(path.read_text())
    if not matches:
        raise ValueError(f"No 'Saved checkpoint:' line found in {path}")
    return Path(matches[-1]).expanduser()


def config_without_seeds(config) -> dict:
    data = asdict(config)
    data.pop("seeds", None)
    data.pop("gpu_indices", None)
    return data


def is_inf_beta_label(label: str) -> bool:
    return label == "inf" or label.endswith(" inf") or "β=inf" in label or "beta=inf" in label


def drop_inf_beta_results(results: dict) -> dict:
    return {label: per_seed for label, per_seed in results.items() if not is_inf_beta_label(label)}


def finite_betas(config) -> list:
    return [beta for beta in getattr(config, "betas", []) if not math.isinf(float(beta))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge single-seed sharded checkpoints into one plotting-only metrics checkpoint."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Checkpoint paths or run logs.")
    parser.add_argument("--out", type=Path, required=True, help="Output .pt checkpoint path.")
    parser.add_argument("--exclude-beta-inf", action="store_true", help="Drop beta=inf results from the merged file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sharded.checkpoint import load_checkpoint_with_metadata, save_metrics_checkpoint

    paths = [checkpoint_path(path) for path in args.inputs]
    loaded = [load_checkpoint_with_metadata(path) for path in paths]

    ref = loaded[0]
    ref_config = ref.config
    ref_config_cmp = config_without_seeds(ref_config)
    ref_results = drop_inf_beta_results(ref.results) if args.exclude_beta_inf else ref.results
    ref_labels = set(ref_results)
    if not ref_labels:
        raise ValueError("No result labels remain after filtering.")
    ref_tracked = tuple(ref.metadata.get("tracked_metrics") or ())
    ref_schema = ref.metadata.get("metric_schema_version")
    ref_world_size = int(ref.metadata.get("world_size") or 1)

    merged = {label: {} for label in ref_results}
    seeds = []

    for path, item in zip(paths, loaded):
        item_results = drop_inf_beta_results(item.results) if args.exclude_beta_inf else item.results
        if config_without_seeds(item.config) != ref_config_cmp:
            raise ValueError(f"Config mismatch outside seeds: {path}")
        if set(item_results) != ref_labels:
            raise ValueError(f"Label mismatch: {path}")
        if tuple(item.metadata.get("tracked_metrics") or ()) != ref_tracked:
            raise ValueError(f"tracked_metrics mismatch: {path}")
        if item.metadata.get("metric_schema_version") != ref_schema:
            raise ValueError(f"metric schema mismatch: {path}")
        if int(item.metadata.get("world_size") or 1) != ref_world_size:
            raise ValueError(f"world_size mismatch: {path}")

        for label, per_seed in item_results.items():
            for seed, metrics in per_seed.items():
                seed = int(seed)
                if seed in merged[label]:
                    raise ValueError(f"Duplicate seed {seed} for {label!r}")
                merged[label][seed] = metrics
                seeds.append(seed)

    out_config = replace(ref_config, seeds=sorted(set(seeds)), checkpoint_state="metrics_only")
    if args.exclude_beta_inf:
        out_config = replace(out_config, betas=finite_betas(out_config))
    ctx = SimpleNamespace(is_rank0=True, world_size=ref_world_size)
    save_metrics_checkpoint(args.out.expanduser(), merged, out_config, list(ref_tracked), ctx)

    print(f"Merged {len(paths)} checkpoint(s), seeds={out_config.seeds}")
    print(f"Saved plotting checkpoint: {args.out.expanduser()}")


if __name__ == "__main__":
    main()
