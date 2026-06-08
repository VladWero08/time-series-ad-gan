import torch
import numpy as np
import typing as t


def normalization(X: np.ndarray) -> np.ndarray:
    """Min-max normalization to [-1, 1]."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_normalized


def overlaps(a: t.Tuple[int, int], b: t.Tuple[int, int]) -> bool:
    """Checks if the given intervals are overlapping."""
    return a[0] <= b[1] and b[0] <= a[1]


def intervals_to_points(y_intervals: t.List[t.List], n_labels: int) -> torch.Tensor:
    point_labels = torch.zeros(n_labels)
    for y_interval in y_intervals:
        start, end = y_interval[0], y_interval[1]
        point_labels[start:end] = 1
    return point_labels


def split_forecast(
        X: np.ndarray, 
        y: np.ndarray,
        sw: int,
        train_ratio: float = 0.5,
        val_ratio: float = 0.1,
        log: bool = True,
    ) -> t.Tuple[np.ndarray]:
        T = X.shape[0]
        train_end = int(T * train_ratio)
        val_end   = int(T * (train_ratio + val_ratio))

        X_train, y_train    = normalization(X[:train_end]), y[:train_end][sw:]
        X_val, y_val        = normalization(X[train_end:val_end]), y[train_end:val_end][sw:]
        X_test, y_test      = normalization(X[val_end:]), y[val_end:][sw:]  

        if log:
            print(f"Series shape : {X.shape}")
            print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
            print(f"Validation   : timesteps {train_end} -> {val_end}  ({val_end - train_end} steps)")
            print(f"Test         : timesteps {val_end} -> {T}  ({T - val_end} steps)\n")

        return X_train, y_train, X_val, y_val, X_test, y_test


def split_reconstruct(
        X: np.ndarray, 
        y: np.ndarray,
        sw: int,
        train_ratio: float = 0.5,
        val_ratio: float = 0.1,
        log: bool = True,
    ) -> t.Tuple[np.ndarray]:
        T = X.shape[0]
        train_end = int(T * train_ratio)
        val_end   = int(T * (train_ratio + val_ratio))
        cutoff = sw // 2

        X_train, y_train    = normalization(X[:train_end]), y[:train_end][cutoff:-cutoff]
        X_val, y_val        = normalization(X[train_end:val_end]), y[train_end:val_end][cutoff:-cutoff]
        X_test, y_test      = normalization(X[val_end:]), y[val_end:][cutoff:-cutoff]      

        if log:
            print(f"Series shape : {X.shape}")
            print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
            print(f"Validation   : timesteps {train_end} -> {val_end}  ({val_end - train_end} steps)")
            print(f"Test         : timesteps {val_end} -> {T}  ({T - val_end} steps)\n")

        return X_train, y_train, X_val, y_val, X_test, y_test