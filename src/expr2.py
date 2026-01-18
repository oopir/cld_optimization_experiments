# ex2_linearized.py
from .data import load_digits_data
from .training import train
from .plots import plot_ex2
import torch

def run(epochs):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0

    data = load_digits_data(n=1450, random_labels=True, device=device, seed=seed)
    n = data["X_train"].shape[0]
    d = data["X_train"].shape[1]

    results = {
        "noisy": train(
            data=data,
            eta=1e-5,
            epochs=epochs,
            beta=n * 1e2,
            lam_fc1=d / (torch.nn.init.calculate_gain("tanh") ** 2),
            lam_fc2=n * d,
            hidden_width=n * d,
            regularization_scale=1.0,
            use_linearized=True,
            track_jacobian=True,
            device=device,
            seed=seed,
            print_every=epochs//10,
        )
    }
    plot_ex2(results)
