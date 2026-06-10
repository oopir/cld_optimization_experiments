from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.datasets._openml",
)


def _normalize_rows(X, pixel_scale):
    X = X.astype(np.float32) / pixel_scale
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1])
    return X.astype(np.float32)


def _randomize_labels(y_train, seed, num_classes=10):
    rng = np.random.RandomState(seed)
    return rng.randint(0, num_classes, size=y_train.shape[0])


def _as_tensor_dataset(X_train, X_test, y_train, y_test, d_in, device, num_classes=10):
    X_train = torch.tensor(X_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    return {
        "d_in": d_in,
        "d_out": num_classes,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_one_hot": torch.eye(num_classes, device=device)[y_train],
        "y_test_one_hot": torch.eye(num_classes, device=device)[y_test],
    }


def load_digits_data(n, random_labels=False, device="cpu", seed=42):
    digits = load_digits()
    X = _normalize_rows(digits.data, pixel_scale=16.0)
    y = digits.target.astype(np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=max(100, n//5), stratify=y_tmp, random_state=seed)

    if random_labels:
        y_train = _randomize_labels(y_train, seed)

    return _as_tensor_dataset(X_train, X_test, y_train, y_test, d_in=64, device=device)


def load_mnist_data(n, random_labels=False, device="cpu", seed=42, reserve_last=1000):
    """
    Load MNIST with the same preprocessing used by the experiments.

    If n is too large to leave the usual test holdout, this falls back to an
    80/20 stratified split and prints the resulting train size.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = _normalize_rows(mnist["data"], pixel_scale=255.0)
    y = mnist["target"].astype(np.int64)

    if reserve_last > 0:
        X = X[0 : X.shape[0] - reserve_last]
        y = y[0 : y.shape[0] - reserve_last]

    standard_split_limit = 60000 - reserve_last
    if n*(6/5) < standard_split_limit:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
        _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=max(100, n//5), stratify=y_tmp, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)
        print(
            "Requested n is too large for the standard MNIST split "
            f"(n={n}, standard split limit after reserve_last={standard_split_limit}, "
            f"loaded examples={X.shape[0]}); "
            f"using an 80/20 split instead with train_size={X_train.shape[0]} "
            f"and test_size={X_test.shape[0]}."
        )

    if random_labels:
        y_train = _randomize_labels(y_train, seed)

    return _as_tensor_dataset(X_train, X_test, y_train, y_test, d_in=784, device=device)
