#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
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

from sharded.checkpoint import load_checkpoint_with_metadata
from src.plots import plot_ex1_multiseed, plot_test_error_vs_alpha


def load_sharded_checkpoint(path: Path):
    loaded = load_checkpoint_with_metadata(path)
    return loaded.results, loaded.config


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
