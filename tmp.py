# inspect_checkpoints.py
import os
import numpy as np
import torch

from src.metric_checkpoints import load_exp1_checkpoint

def summarize_ckpt(path, label):
    results, config = load_exp1_checkpoint(path)
    print(f"=== {label} ===")
    print(f"path         : {path}")
    print(f"epochs       : {config.epochs}")
    print(f"track_every  : {config.track_every}")
    print(f"print_every  : {config.print_every}")
    print(f"betas        : {config.betas}")
    print()

    beta_key = next(iter(results.keys()))
    seed_key = next(iter(results[beta_key].keys()))
    metrics = results[beta_key][seed_key]

    print(f"sample beta_key: {beta_key}")
    print(f"sample seed    : {seed_key}")
    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):
            l = len(v)
            elem_type = type(v[0]) if l > 0 else None
            print(f"  {k:30s} len={l:5d} elem_type={elem_type}")
        elif hasattr(v, "shape"):
            print(f"  {k:30s} shape={tuple(v.shape)} type={type(v)}")
        else:
            print(f"  {k:30s} scalar type={type(v)}")

    print()
    return results, config


if __name__ == "__main__":
    ckpt_4e6 = "/home/ofirg/cld_checkpoints/expr1/exp1_digits_20260127_135649.pt"
    ckpt_6e6 = "/home/ofirg/cld_checkpoints/expr1/exp1_digits_20260127_220538.pt"

    summarize_ckpt(ckpt_4e6, "4e6")
    summarize_ckpt(ckpt_6e6, "6e6")
