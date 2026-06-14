import hashlib
import torch
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.datasets._openml",
)

DATASET_METADATA = {
    "digits": {"d_in": 64, "num_classes": 10},
    "mnist": {"d_in": 784, "num_classes": 10},
}

SYNTHETIC_BINARY_DATASETS = {"synthetic_isotropic", "synthetic_anisotropic"}


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
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

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
    90/10 stratified split and prints the resulting train size.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = _normalize_rows(mnist["data"], pixel_scale=255.0)
    y = mnist["target"].astype(np.int64)

    if reserve_last > 0:
        X = X[0 : X.shape[0] - reserve_last]
        y = y[0 : y.shape[0] - reserve_last]

    standard_split_limit = 63000 - reserve_last
    if n < standard_split_limit:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
        _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=max(100, n//5), stratify=y_tmp, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y, random_state=seed)
        print(
            "Requested n is too large for the standard MNIST split "
            f"(n={n}, standard split limit after reserve_last={standard_split_limit}, "
            f"loaded examples={X.shape[0]}); "
            f"using an 90/10 split instead with train_size={X_train.shape[0]} "
            f"and test_size={X_test.shape[0]}."
        )

    if random_labels:
        y_train = _randomize_labels(y_train, seed)

    return _as_tensor_dataset(X_train, X_test, y_train, y_test, d_in=784, device=device)



# -------------------------------------------------------------------------- #
# ------------------------ binary classification data ---------------------- #
# -------------------------------------------------------------------------- #

def load_binary_classification_data(
    dataset,
    n,
    negative_classes=None,
    positive_classes=None,
    random_labels=False,
    device="cpu",
    seed=42,
    reserve_last=1000,
):
    """Load a binary dataset, returning scalar labels in {-1, +1}."""
    if negative_classes is None:
        negative_classes = [0, 1, 2, 3, 4]
    if positive_classes is None:
        positive_classes = [5, 6, 7, 8, 9]

    if dataset == "digits":
        data = load_digits_data(n=n, random_labels=random_labels, device=device, seed=seed)
    elif dataset == "mnist":
        data = load_mnist_data(
            n=n,
            random_labels=random_labels,
            device=device,
            seed=seed,
            reserve_last=reserve_last,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    y = data["y_train"]
    neg_mask = torch.zeros_like(y, dtype=torch.bool) # len(neg_mask) = len(y). eventually, neg_mask[i] == 1 <---> y[i] in negative__classes
    pos_mask = torch.zeros_like(y, dtype=torch.bool) # len(pos_mask) = len(y). eventually, pos_mask[i] == 1 <---> y[i] in positive__classes
    for cls in negative_classes:
        neg_mask |= y == int(cls)
    for cls in positive_classes:
        pos_mask |= y == int(cls)

    # this part is in case some labels aren't mapped to positive/negative.
    # 'keep' maintains the indices with relevant labels. X anad y are 
    # filtered to include only these datapoints.
    keep = neg_mask | pos_mask
    if keep.sum().item() == 0:
        raise ValueError("Binary class split kept zero training examples.")
    X_train = data["X_train"][keep]
    y_binary = torch.where(pos_mask[keep], 1.0, -1.0) # from the indices under 'keep', positives map to 1, others to -1
    y_binary = y_binary.to(device=X_train.device, dtype=X_train.dtype)

    return {
        "d_in": data["d_in"],
        "d_out": 1,
        "X_train": X_train,
        "y_train_binary": y_binary.view(-1, 1),
        "n_effective": int(X_train.shape[0]),
    }
