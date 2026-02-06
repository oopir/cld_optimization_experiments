import sys
import os
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from src.exp1 import run_exp1, build_from_config_mapping
from src.plots import plot_ex1_multiseed
from src.utils import select_idle_gpus_for_experiment


def load_mapping(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        with path.open("r") as f:
            return yaml.safe_load(f)
    else:
        with path.open("r") as f:
            return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON config file",
    )
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Override run.ckpt_path from config",
    )
    p.add_argument(
        "--new-epochs",
        type=int,
        default=None,
        help="Override run.new_epochs from config",
    )
    p.add_argument(
        "--no-use-checkpoint",
        action="store_true",
        help="Ignore config.run.use_checkpoint and train from scratch",
    )
    p.add_argument(
        "--extend-from-checkpoint",
        action="store_true",
        help="Force extending from checkpoint even if config disables it",
    )
    p.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Force saving a checkpoint at the end",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting after run",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mapping = load_mapping(args.config)

    exp_config, run_opts = build_from_config_mapping(mapping)

    # CLI overrides config
    if args.ckpt_path is not None:
        run_opts.ckpt_path = args.ckpt_path
    if args.new_epochs is not None:
        run_opts.new_epochs = args.new_epochs
    if args.no_use_checkpoint:
        run_opts.use_checkpoint = False
    if args.extend_from_checkpoint:
        run_opts.extend_from_checkpoint = True
    if args.save_checkpoint:
        run_opts.save_checkpoint = True

    gpu_ids = select_idle_gpus_for_experiment(device=exp_config.device)

    results, final_config = run_exp1(config=exp_config, run_opts=run_opts, gpu_ids=gpu_ids)

    if not args.no_plot:
        plot_ex1_multiseed(results, final_config.epochs, final_config.track_every, final_config.use_linearized)


if __name__ == "__main__":
    main()
