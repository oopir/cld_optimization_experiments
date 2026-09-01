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

def load_sharded_checkpoint(path: Path):
    from sharded.checkpoint import load_checkpoint_with_metadata

    loaded = load_checkpoint_with_metadata(path)
    return loaded.results, loaded.config


def parse_epoch(value: str) -> int:
    epoch = int(float(value))
    if epoch < 1:
        raise argparse.ArgumentTypeError("epoch must be >= 1")
    return epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from a sharded_exp1 checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Metrics .pt file or sharded checkpoint directory.")
    parser.add_argument("--outdir", type=Path, default=Path("plots"), help="Directory for generated plots.")
    parser.add_argument("--no-alpha-plot", action="store_true", help="Skip alpha test-error plot.")
    parser.add_argument("--start-epoch", type=parse_epoch, default=None, help="Start plots near this epoch.")
    parser.add_argument(
        "--final-epoch",
        "--max-epoch",
        dest="final_epoch",
        type=parse_epoch,
        default=None,
        help="End plots near this epoch.",
    )
    args = parser.parse_args()
    if args.start_epoch is not None and args.final_epoch is not None and args.start_epoch > args.final_epoch:
        parser.error("--start-epoch must be <= --final-epoch")
    return args


def main() -> None:
    args = parse_args()
    from src.plots import plot_ex1_multiseed, plot_test_error_vs_alpha

    results, config = load_sharded_checkpoint(args.checkpoint)
    plot_ex1_multiseed(
        results,
        config.epochs,
        config.track_every,
        config.use_linearized,
        plot_output_dir=args.outdir,
        min_epoch=args.start_epoch,
        max_epoch=args.final_epoch,
    )
    if not args.no_alpha_plot and len(getattr(config, "alphas", []) or []) > 1:
        plot_test_error_vs_alpha(
            results,
            output_path=args.outdir / "alpha_test_error.pdf",
            track_every=config.track_every,
            min_epoch=args.start_epoch,
            max_epoch=args.final_epoch,
        )
    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
