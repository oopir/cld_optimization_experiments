# retrofit_epoch_hist.py
import os
import numpy as np
import torch

from src.metric_checkpoints import load_exp1_checkpoint, save_exp1_checkpoint


def build_epoch_hist(segments):
    """
    segments: list of (start_epoch, end_epoch, track_every), global epochs.
    Assumes metrics were logged when local_epoch % track_every == 1,
    with local_epoch running from 1..(end - start).
    """
    epochs = []
    for start, end, t in segments:
        assert end > start
        extra = end - start
        # local epochs: 1, 1+t, 1+2t, ..., <= extra
        local = 1 + np.arange(0, extra, t, dtype=np.int64)
        local = local[local <= extra]
        global_e = start + local
        epochs.append(global_e)

    return np.concatenate(epochs, axis=0)


def retrofit_checkpoint(in_path, out_path, segments):
    results, config = load_exp1_checkpoint(in_path)
    print(f"Loaded: {in_path}")
    print(f"config.epochs={config.epochs}, track_every={config.track_every}")

    # Build epoch_hist once and sanity-check its length
    sample_beta_key = next(iter(results.keys()))
    sample_seed_key = next(iter(results[sample_beta_key].keys()))
    sample_metrics = results[sample_beta_key][sample_seed_key]

    T_metrics = len(sample_metrics["train_loss_hist"])
    epoch_hist = build_epoch_hist(segments)
    T_epochs = len(epoch_hist)

    print(f"metrics length = {T_metrics}, epoch_hist length = {T_epochs}")
    if T_metrics != T_epochs:
        raise ValueError("Segment specification is inconsistent with stored history length.")

    # Attach epoch_hist to all seeds / betas
    for beta_key, by_seed in results.items():
        for seed, metrics in by_seed.items():
            metrics["epoch_hist"] = epoch_hist.copy()

    # Optionally also update config.track_every to something consistent:
    # here we just keep the old value; plotting will prefer epoch_hist anyway.

    save_exp1_checkpoint(out_path, results, config)
    print(f"Saved fixed checkpoint to: {out_path}")


if __name__ == "__main__":
    # EXAMPLE for your 6e6 checkpoint, **based on the schedule we inferred**:
    #
    #   0   → 3_000_000   with track_every = 30_000
    #   3e6 → 4_000_000   with track_every = 10_000
    #   4e6 → 6_000_000   with track_every = 30_000
    #
    # Adjust these if your run history was different.

    SEGMENTS_6E6 = [
        (0,       3_000_000, 30_000),
        (3_000_000, 4_000_000, 10_000),
        (4_000_000, 6_000_000, 30_000),
    ]

    ckpt_in  = "/home/ofirg/cld_checkpoints/expr1/exp1_digits_20260127_220538.pt"
    ckpt_out = "/home/ofirg/cld_checkpoints/expr1/exp1_digits_20260127_220538_fixed.pt"

    retrofit_checkpoint(ckpt_in, ckpt_out, SEGMENTS_6E6)

    # For the 4e6 checkpoint you’d use just the first two segments:
    # SEGMENTS_4E6 = [
    #     (0,       3_000_000, 30_000),
    #     (3_000_000, 4_000_000, 10_000),
    # ]
    # retrofit_checkpoint(ckpt_4e6_in, ckpt_4e6_out, SEGMENTS_4E6)
