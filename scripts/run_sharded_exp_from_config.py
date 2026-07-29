#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths, configs, "
        f"checkpoints, and imports resolve consistently:\n  cd {REPO_ROOT}\n"
        "  torchrun --standalone --nproc_per_node=<N> "
        "scripts/run_sharded_exp_from_config.py --config <path>"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from sharded.config import build_from_config_mapping, load_mapping
from sharded.distributed import cleanup_distributed, init_distributed, rank0_print
from sharded.exp import run_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sharded heavy dense-network experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--save-ckpt", action="store_true", help="Force checkpoint saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.config)
    exp_config, run_opts = build_from_config_mapping(mapping)
    if args.save_ckpt:
        run_opts.save_ckpt = True

    ctx = init_distributed(exp_config.device, exp_config.gpu_indices)
    try:
        rank0_print(ctx, datetime.now(), flush=True)
        run_exp(exp_config, run_opts, ctx)
        rank0_print(ctx, datetime.now(), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
