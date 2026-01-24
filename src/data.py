from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def load_digits_data(n, random_labels=False, device="cpu", seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0 # scale to [0,1]
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1]) # normalize to \sqrt{d} norm
    X = X.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    X_test, _, y_test, _ = train_test_split(
        X_tmp, y_tmp, test_size=max(10, n//5), stratify=y_tmp, random_state=seed
    )

    if random_labels:
        y_train = np.random.randint(0, 10, size=n)

    X_train = torch.tensor(X_train, device=device)
    X_test  = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test  = torch.tensor(y_test, device=device)

    y_train_one_hot = torch.eye(10, device=device)[y_train]
    y_test_one_hot  = torch.eye(10, device=device)[y_test]

    return {
        "d_in": X_train[0].shape[1],
        "d_out": y_train_one_hot[0].shape[0],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_one_hot": y_train_one_hot,
        "y_test_one_hot": y_test_one_hot,
    }


def load_1d_regression_data(
    X_values=None,
    y_values=None,
    test_fraction=0,
    device="cpu",
    seed=42,
):
    if X_values is None or y_values is None:
        X_values = np.array([-1.5, -1.12, -0.74, -0.38, 0, 0.38, 0.74, 1.12, 1.5], dtype=np.float32)
        y_values = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)

    X = np.asarray(X_values, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(y_values, dtype=np.float32)

    X_train, X_test, y_train, y_test = X, X, y, y

    X_train = torch.tensor(X_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    return {
        "d_in": 1,
        "d_out": 1,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }