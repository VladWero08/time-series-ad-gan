import torch
import typing as t
import numpy as np
import matplotlib.pyplot as plt

from src.utils.data import overlaps


def evaluate_point_anomalies(y: torch.Tensor, y_hat: torch.Tensor) -> t.Tuple[float]:
    """
    Computes precision, recall, and F1 score for point anomaly detection.

    A point anomaly is an isolated anomalous timestep with no adjacent anomalies.
    Only isolated anomalies in `y` are evaluated; consecutive anomalies are ignored.

    Returns
    -------
    precision, recall, f1 : Tuple[float]
    """
    # handle the case when the time series does not contain any anomalies
    if all(y == 0) and all(y_hat == 0):
        return 1.0, 1.0, 1.0
    
    N = y.shape[0]

    def is_point_anomaly(i: int) -> bool:
        "An anomaly is a point anomaly if the surrounding points are not anomalies. Check whether the given index contains a point anomaly."
        if i == 0:
            return y[i] == 1 and y[i + 1] == 0
        elif i == N - 1:
            return y[i - 1] == 0 and y[i] == 1
        
        return y[i - 1] == 0 and y[i] == 1 and y[i + 1] == 0

    tp = fp = fn = 0

    for i in range(N):
        if is_point_anomaly(i):
            if y_hat[i] == 1:
                tp += 1
            else:
                fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
                fp += 1

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def evaluate_collective_anomalies(y_intervals: torch.Tensor, y_hat_intervals: torch.Tensor) -> t.Tuple[float]:
    """
    Computes precision, recall, and F1 score for collective time-series anomaly detection.

    Rather than comparing labels timestep by timestep, anomalies are grouped into contiguous intervals, and the following rules are used:
    - `TP`: when an anomalous window overlaps any predicted window;
    - `FN`: when an anomalous window does not overlap any predicted window;
    - `FP`: when a predicted window does not overlap any anomalous window.

    Returns
    -------
    precision, recall, f1 : Tuple[float]
    """
    # handle the case when the time series does not contain any anomalies
    if len(y_intervals) == 0 and len(y_hat_intervals) == 0:
        return 1.0, 1.0, 1.0
    
    tp = fp = fn = 0

    # TP / FN: for each true anomaly interval, check if any predicted interval overlaps it
    for y_interval in y_intervals:
        if any(overlaps(y_interval, y_hat_interval) for y_hat_interval in y_hat_intervals):
            tp += 1
        else:
            fn += 1

    # FP: for each predicted interval, check if any anomaly interval overlaps it
    for y_hat_interval in y_hat_intervals:
        if not any(overlaps(y_hat_interval, y_interval) for y_interval in y_intervals):
            fp += 1

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def plot_performance(
    X: torch.Tensor,
    X_preds: torch.Tensor,
) -> None:
    """
    Plots both the original `X` and predicted `X_preds` time-series. 

    Paramters:
    ----------
    X: torch.Tensor
        Original time-series of shape `(T, 1)`.
    X_preds: torch.Tensor
        Predicted time-series of shape `(T, 1)`.
    """
    def format(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if len(x.shape) > 1:
            x = x.flatten()
        return x

    X = format(X)
    X_preds = format(X_preds)

    plt.figure(figsize=(10, 5))
    plt.plot(X, color="blue", label="Time-Series")
    plt.plot(X_preds, color="orange", label="Prediction")
    plt.xlabel("Time Step")
    plt.legend()
    plt.tight_layout()
    plt.show()
