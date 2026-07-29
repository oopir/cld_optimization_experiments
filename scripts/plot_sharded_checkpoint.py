#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths and imports "
        f"resolve consistently:\n  cd {REPO_ROOT}\n"
        "  python3 scripts/plot_sharded_checkpoint.py <checkpoint>"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch

from src.plots import plot_ex1_multiseed, plot_test_error_vs_alpha


def _checkpoint_payload_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / "results.pt"
    return path


def load_sharded_checkpoint(path: Path):
    payload_path = _checkpoint_payload_path(path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload_type = payload.get("type")
    if payload_type != "sharded_exp1":
        raise ValueError(f"Expected a sharded_exp1 checkpoint, got {payload_type!r}")
    results = payload["results"]
    config = payload.get("config")
    if config is None:
        config = SimpleNamespace(**payload["config_dict"])
    return results, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from a sharded_exp1 checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Metrics .pt file or sharded checkpoint directory.")
    parser.add_argument("--outdir", type=Path, default=Path("plots"), help="Directory for generated plots.")
    parser.add_argument("--no-alpha-plot", action="store_true", help="Skip alpha test-error plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, config = load_sharded_checkpoint(args.checkpoint)
    plot_ex1_multiseed(
        results,
        config.epochs,
        config.track_every,
        config.use_linearized,
        plot_output_dir=args.outdir,
    )
    if not args.no_alpha_plot and len(getattr(config, "alphas", []) or []) > 1:
        plot_test_error_vs_alpha(results, output_path=args.outdir / "alpha_test_error.pdf")
    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()

