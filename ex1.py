# %%
import sys
import os
from pathlib import Path

ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

# %%
import os
import glob
from datetime import datetime
from dataclasses import replace
import shutil

import numpy as np
import random
import torch

from src.training import train_multiseed
from src.utils import select_idle_gpus_for_experiment
from src.plots import plot_ex1_multiseed, plot_ex1_multiseed_short
from src.metric_checkpoints import Exp1Config, save_exp1_checkpoint, load_exp1_checkpoint

def main():
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_ids = select_idle_gpus_for_experiment(device=device, util_threshold=5)
    print(f"Using GPUs: {gpu_ids}")

    # %% [markdown]
    # <h1 style="color:red;">VERY IMPORTANT THINGS TO SET</h1>

    # %%
    SAVE_CHECKPOINT = True
    USE_CHECKPOINT = True
    EXTEND_FROM_CHECKPOINT = True
    NEW_EPOCHS = int(6e06)  # this number should be old_num_epochs + extra_num_epochs

    if "google.colab" in sys.modules:
        CKPT_DIR = "/content/drive/MyDrive/cld_checkpoints"
    else:
        CKPT_DIR = os.path.expanduser("~/cld_checkpoints/expr1")
    CKPT_PATH = os.path.join(CKPT_DIR, "exp1_digits_20260127_135649.pt")


    if not USE_CHECKPOINT:
        epochs = int(4e06)
        eta    = 1e-5
        n      = 10
        betas_to_plot = [10*n, 50*n, 100*n]
        seeds = list(range(5))

    # %%
    def latest_exp1_checkpoint():
        paths = glob.glob(os.path.join(CKPT_DIR, "exp1_digits_*.pt"))
        if not paths:
            return None
        return max(paths, key=os.path.getmtime)

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    os.makedirs(CKPT_DIR, exist_ok=True)

    if USE_CHECKPOINT:
        ckpt_path = CKPT_PATH or latest_exp1_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(
                "No exp1 checkpoint found; set USE_CHECKPOINT=False to train."
            )
        results, config = load_exp1_checkpoint(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")

        if EXTEND_FROM_CHECKPOINT:
            if NEW_EPOCHS is None:
                raise ValueError("Set NEW_EPOCHS when EXTEND_FROM_CHECKPOINT=True.")
            if NEW_EPOCHS <= config.epochs:
                raise ValueError(f"NEW_EPOCHS={NEW_EPOCHS} must be > config.epochs={config.epochs}.")

            extra_epochs = NEW_EPOCHS - config.epochs
            print(f"Extending from {config.epochs} to {NEW_EPOCHS} epochs (extra {extra_epochs}).")

            common = replace(config, epochs=extra_epochs).train_kwargs()
            common["gpu_ids"] = gpu_ids
            common["print_every"] = max(1, extra_epochs // 50)

            extended_results = {}

            # determine which metric keys are timeseries (to concatenate)
            sample_beta_key = next(iter(results.keys()))
            sample_seed_key = next(iter(results[sample_beta_key].keys()))
            sample_metrics = results[sample_beta_key][sample_seed_key]
            timeseries_keys = {k for k in sample_metrics.keys() if k.endswith("_hist")}

            resume_root = os.path.join(CKPT_DIR, "resume_states_tmp")
            os.makedirs(resume_root, exist_ok=True)

            # config.betas is aligned with the dict order you created originally
            for beta, beta_key in zip(config.betas, results.keys()):
                old_by_seed = results[beta_key]

                # # per-seed init and start states
                # init_model_state_dicts = {}
                # start_model_state_dicts = {}
                # start_lin_params_dicts = {}
                # for seed, seed_metrics in old_by_seed.items():
                #     start_model_state_dicts[seed] = seed_metrics["model_state_dict"]
                #     if "init_model_state_dict" in seed_metrics:
                #         init_model_state_dicts[seed] = seed_metrics["init_model_state_dict"]
                #     else:
                #         # fallback: if old checkpoint lacks init, use its model_state_dict
                #         init_model_state_dicts[seed] = seed_metrics["model_state_dict"]
                #     if "lin_params_state" in seed_metrics:
                #         start_lin_params_dicts[seed] = seed_metrics["lin_params_state"]
                #     else:
                #         start_lin_params_dicts[seed] = None

                # new_by_seed = train_multiseed(
                #     dataset="digits",
                #     beta=beta,
                #     init_model_state_dicts=init_model_state_dicts,
                #     start_model_state_dicts=start_model_state_dicts,
                #     start_lin_params_dicts=start_lin_params_dicts,
                #     **common,
                # )
                
                
                # write per-seed resume payloads to disk; pass only paths to workers (spawn-friendly)
                beta_resume_dir = os.path.join(resume_root, beta_key.replace("β", "b"))
                if os.path.isdir(beta_resume_dir):
                    shutil.rmtree(beta_resume_dir)
                os.makedirs(beta_resume_dir, exist_ok=True)

                resume_paths = {}
                for seed, seed_metrics in old_by_seed.items():
                    payload = {
                        "init_model_state_dict": seed_metrics.get(
                            "init_model_state_dict",
                            seed_metrics["model_state_dict"],
                        ),
                        "start_model_state_dict": seed_metrics["model_state_dict"],
                        "start_lin_params": seed_metrics.get("lin_params_state", None),
                    }
                    p = os.path.join(beta_resume_dir, f"seed_{seed}.pt")
                    torch.save(payload, p)  # CPU tensors
                    resume_paths[seed] = p

                new_by_seed = train_multiseed(
                     dataset="digits",
                     beta=beta,
                    resume_paths=resume_paths,
                    **common,
                 )

                merged_by_seed = {}
                for seed in old_by_seed.keys():
                    old_metrics = old_by_seed[seed]
                    new_metrics = new_by_seed[seed]

                    merged_seed = {}
                    for k, v_old in old_metrics.items():
                        if k in ("model_state_dict", "init_model_state_dict", "lin_params_state"):
                            # keep newest versions
                            continue

                        if k in timeseries_keys:
                            v_old_arr = _to_np(v_old)
                            v_new_arr = _to_np(new_metrics[k])
                            merged_seed[k] = np.concatenate([v_old_arr, v_new_arr], axis=0)
                        else:
                            # scalar / non-timeseries; just take from the new run
                            merged_seed[k] = new_metrics[k]

                    merged_seed["model_state_dict"] = new_metrics["model_state_dict"]
                    if "init_model_state_dict" in new_metrics:
                        merged_seed["init_model_state_dict"] = new_metrics["init_model_state_dict"]
                    if "lin_params_state" in new_metrics:
                        merged_seed["lin_params_state"] = new_metrics["lin_params_state"]
                    merged_by_seed[seed] = merged_seed

                extended_results[beta_key] = merged_by_seed

            results = extended_results
            config = replace(config, epochs=NEW_EPOCHS, print_every=max(1, NEW_EPOCHS // 10))
    else:
        m = max([n * np.log(n) * beta * np.log(beta) for beta in betas_to_plot])
        m = int(max(4096, m))
        print(f"m={m}")

        config = Exp1Config(
            seeds=seeds,
            n=n,
            random_labels=False,
            betas=betas_to_plot,
            epochs=epochs,
            eta=eta,
            m=m,
            device=device,
            track_every=max(1,epochs//100),
            print_every=epochs//5,
        )

        common = config.train_kwargs()
        common["gpu_ids"] = gpu_ids
        
        results = {
            f"β={beta // config.n}n": train_multiseed(dataset="digits", beta=beta, **common)
            for beta in config.betas
        }

    if SAVE_CHECKPOINT:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(CKPT_DIR, f"exp1_digits_{timestamp}.pt")
        save_exp1_checkpoint(ckpt_path, results, config)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
