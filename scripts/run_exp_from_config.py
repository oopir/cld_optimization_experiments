import sys
import os
import argparse
import json
from pathlib import Path
import yaml
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit(
        "Run this script from the repository root so relative paths, configs, "
        f"checkpoints, and imports resolve consistently:\n  cd {REPO_ROOT}\n"
        "  python scripts/run_exp_from_config.py --config <path>"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from src.exp import run_exp, build_from_config_mapping, tune_eta_for_exp
from src.plots import plot_ex1_multiseed, plot_test_error_vs_alpha
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
        "--load-ckpt-name",
        type=Path,
        default=None,
        help="Override run.load_ckpt_name from config",
    )
    p.add_argument(
        "--new-total-epochs",
        type=int,
        default=None,
        help="Override run.new_total_epochs from config",
    )
    p.add_argument(
        "--no-load-ckpt",
        action="store_true",
        help="Ignore config.run.load_ckpt and train from scratch",
    )
    p.add_argument(
        "--resume-from-ckpt",
        action="store_true",
        help="Force resuming from checkpoint even if config disables it",
    )
    p.add_argument(
        "--save-ckpt",
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
    if args.load_ckpt_name is not None:
        run_opts.load_ckpt_name = args.load_ckpt_name
    if args.new_total_epochs is not None:
        run_opts.new_total_epochs = args.new_total_epochs
    if args.no_load_ckpt:
        run_opts.load_ckpt = False
    if args.resume_from_ckpt:
        run_opts.resume_from_ckpt = True
    if args.save_ckpt:
        run_opts.save_ckpt = True

    gpu_ids = select_idle_gpus_for_experiment(device=exp_config.device)

    eta_tuning_cfg = mapping.get("eta_tuning")
    if eta_tuning_cfg and eta_tuning_cfg.get("enabled", False):
        tune_eta_for_exp(exp_config, eta_tuning_cfg, gpu_ids=gpu_ids)
    else:
        results, final_config = run_exp(config=exp_config, run_opts=run_opts, gpu_ids=gpu_ids)
        if not args.no_plot:
            plot_ex1_multiseed(results, final_config.epochs, final_config.track_every, final_config.use_linearized)
            if len(final_config.alphas) > 1:
                plot_test_error_vs_alpha(results)


if __name__ == "__main__":
    print(datetime.now())
    main()
    print(datetime.now())
