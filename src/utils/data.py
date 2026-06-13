import torch
import numpy as np
import typing as t
import math


def normalization(X: np.ndarray, X_train_min: np.ndarray, X_train_max: np.ndarray) -> np.ndarray:
    """Normalizes the values of the given array `X` to [-1, 1] using `X_min` and `X_max`."""
    X_normalized = 2 * (X - X_train_min) / (X_train_max - X_train_min + 1e-8) - 1
    return X_normalized


def overlaps(a: t.Tuple[int, int], b: t.Tuple[int, int]) -> bool:
    """Checks if the given intervals are overlapping."""
    return a[0] <= b[1] and b[0] <= a[1]


def intervals_to_points(y_intervals: t.List[t.List[int]], n_labels: int) -> torch.Tensor:
    """
    Transforms the anomaly intervals into a tensor with labels:
    - `0`: points outside intervals
    - `1`: points insinde the interval

    Parameters
    ----------
    y_intervals : t.List[t.List[int]]
        List of intervals with inclusive ends. (e.g. [[100, 105], [200, 300]])

    Return
    ------
    point_labels : torch.Tensor
        Tensor with labels 0 and 1 obtained from the given intervals.
    """
    point_labels = torch.zeros(n_labels)
    for y_interval in y_intervals:
        start, end = y_interval[0], y_interval[1]
        point_labels[start:end+1] = 1
    return point_labels


def split(
        X: np.ndarray,
        y: np.ndarray,
        sw: int,
        train_ratio: float = 0.5,
        val_ratio: float = 0.1,
        type: str = "forecast",
    ) -> t.Tuple[np.ndarray]:
        """
        Splits a time series and its anomaly labels into train, validation, and test subsets:
        - partitions `X` and `y` based on the provided ratios;
        - applies normalization to the splits;
        - slices the labels based on the prediction process used (forecast or reconstrucion).

        Parameters
        ----------
        X : np.ndarray of shape (T, n_features)
            The full time-series feature matrix.
        y : np.ndarray of shape (T,)
            The corresponding ground-truth anomaly labels.
        sw : int
            Sliding window size used by the sequence generator model.
        train_ratio : float, default=0.5
            The fraction of total data allocated for the training.
        val_ratio : float, default=0.1
            The fraction of total data allocated for the validation. 
            If set to 0.0, the validation split is skipped entirely.
        type: {"forecast", "reconstruct"}, default="forecast"
            - "forecast": removes first `sw` points from the label tensor.
            - "reconstruct": removes `sw // 2` points from both the start and end of the label tensor.

        Returns
        -------
        tuple of np.ndarray
            A dynamic flat tuple containing the split arrays in chronological order:
            - If test, val, and train are active: (X_train, y_train, X_val, y_val, X_test, y_test)
            - If validation is skipped: (X_train, y_train, X_test, y_test)
            - If train and validation consume the whole series: (X_train, y_train, X_val, y_val)
        """
        T = X.shape[0]
        has_val = val_ratio > 0.0
        has_test = (train_ratio + val_ratio) < 1.0
        val_size = int(T * val_ratio)
        
        train_end = int(T * train_ratio - max(0, sw + 1 - val_size) * int(has_val))
        val_end = int(train_end + max(sw + 1, val_size) * int(has_val))
      
        match type:
            case "forecast":
                # when forecasting, the first sw points will not be used
                y_slice = slice(sw, None)
            case "reconstruct":
                # when reconstructing, the first and last sw / 2 points will not be used
                cutoff = sw // 2
                y_slice = slice(cutoff, -cutoff)

        data_splits = []
        # slices used for train, validation and test
        data_slices = [
            (True, slice(0, train_end)),
            (has_val, slice(train_end, val_end)),
            (has_test, slice(val_end, None))
        ]

        for active, data_slice in data_slices:
            if active:
                X_split = X[data_slice]
                y_split = y[data_slice][y_slice]
                data_splits.append(X_split); data_splits.append(y_split)

        return tuple(data_splits)


def agg_reconstructions(X_rec: torch.Tensor, sw: int, ss: int) -> torch.Tensor:
    """
    Aggregates sliding window reconstruction to a per-timestep error by computing the mean over all windows that contain each timestep.

    Parameters
    ----------
    X_rec : torch.Tensor
        Reconstruction values per window and per position within that window.
    sw : int
        Sliding window size.
    ss : int
        Sliding window step size.

    Returns
    -------
    agg_errors : torch.Tensor of shape (T, features)
        Mean aggregated values for each original timestep.
    """
    N = X_rec.shape[0]
    T = (N - 1) * ss + sw
    agg_errors = []

    for i in range(T):
        # window k covers:                   [k * ss, k * ss + sw - 1]
        # window k contains timestep i if:   k * ss <= i <= k * ss + sw - 1
        # the inequality based on k:         (i - sw + 1) / ss <= k <= i / ss 
        k_min = max(0, int(np.ceil((i - sw + 1) / ss)))
        k_max = min(N - 1, int(np.floor(i / ss)))

        # position i in window k is at i-k*ss
        errors_at_i = torch.stack([
            X_rec[k, i - k * ss]
            for k in range(k_min, k_max + 1)
        ])
        median_error = torch.mean(errors_at_i, dim=0)
        agg_errors.append(median_error)

    return torch.stack(agg_errors)

