from .data import load_digits_data
from .training import train
from .plots import plot_ex1
import torch

def run(epochs, n=500):
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0

    data = load_digits_data(n=n, random_labels=False, device=device, seed=seed)
    n = data["X_train"].shape[0]
    d = data["X_train"].shape[1]

    common = dict(
        data=data,
        eta=1e-5,
        epochs=epochs,
        lam_fc1=d / (torch.nn.init.calculate_gain("tanh") ** 2),
        lam_fc2=n**2,
        hidden_width=n**2,
        regularization_scale=1.0,
        use_linearized=False,
        track_jacobian=False,
        device=device,
        seed=seed,
        print_every=epochs//10,
    )

    results = {
        "clean": train(beta=n * 1e5, **common),
        "noisy": train(beta=n * 1e2, **common),
    }
    plot_ex1(results)
