from sklearn.datasets import load_digits, fetch_openml
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

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=max(100, n//5), stratify=y_tmp, random_state=seed)

    if random_labels:
        y_train = np.random.randint(0, 10, size=n)

    X_train = torch.tensor(X_train, device=device)
    X_test  = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test  = torch.tensor(y_test, device=device)

    y_train_one_hot = torch.eye(10, device=device)[y_train]
    y_test_one_hot  = torch.eye(10, device=device)[y_test]

    return {
        "d_in": 64,
        "d_out": 10,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_one_hot": y_train_one_hot,
        "y_test_one_hot": y_test_one_hot,
    }


def load_mnist_data(n, random_labels=False, device="cpu", seed=42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(X.shape[1])
    X = X.astype(np.float32)
    y = mnist["target"].astype(np.int64)

    if n*(6/5) < 60000:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
        _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=max(100, n//5), stratify=y_tmp, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)
      
    if random_labels:
        y_train = np.random.randint(0, 10, size=y_train.shape[0])

    if random_labels:
        y_train = np.random.randint(0, 10, size=n)

    X_train = torch.tensor(X_train, device=device)
    X_test  = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test  = torch.tensor(y_test, device=device)

    y_train_one_hot = torch.eye(10, device=device)[y_train]
    y_test_one_hot  = torch.eye(10, device=device)[y_test]

    return {
        "d_in": 784,
        "d_out": 10,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_one_hot": y_train_one_hot,
        "y_test_one_hot": y_test_one_hot,
    }