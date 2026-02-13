# test_constant_track_every.py
import numpy as np
from src.config import load_checkpoint
import os 

def test_constant_track_every(path):
    results, config = load_checkpoint(path)
    E = config.epochs
    T_conf = config.track_every

    # pick one metrics dict (any beta/seed)
    beta_key = next(iter(results.keys()))
    seed_key = next(iter(results[beta_key].keys()))
    metrics = results[beta_key][seed_key]

    hist = np.asarray(metrics["train_loss_hist"])
    L = len(hist)

    print(f"Checkpoint: {path}")
    print(f"epochs       = {E}")
    print(f"config.track_every = {T_conf}")
    print(f"len(train_loss_hist) = {L}")

    if L <= 1:
        print("Not enough points to infer track_every.")
        return

    # For a *constant* track_every t, we have:
    #   L = floor((E - 1)/t) + 1
    # t must lie in [t_low, t_high].
    t_low  = (E - 1) // L + 1          # ceil((E-1)/L)
    t_high = (E - 1) // (L - 1)        # floor((E-1)/(L-1))

    print(f"feasible t range (if constant logging): [{t_low}, {t_high}]")

    if t_low != t_high:
        print("=> History length is NOT consistent with a single constant track_every.")
    else:
        t = t_low
        print(f"=> Unique constant track_every compatible with this history: t = {t}")
        if t == T_conf:
            print("   and it matches config.track_every ✅")
        else:
            print(f"   but it does NOT match config.track_every ({T_conf}) ❌")


if __name__ == "__main__":
    CKPT_DIR = os.path.expanduser("~/cld_checkpoints/expr1")
    CKPT_PATH = os.path.join(CKPT_DIR, "exp1_digits_20260127_072707.pt")
    test_constant_track_every(CKPT_PATH)
