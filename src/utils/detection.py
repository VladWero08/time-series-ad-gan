import torch
import typing as t
import numpy as np


def detect_point_anomalies(
        train_errors: torch.Tensor, 
        test_errors: torch.Tensor,
        max_std: int = 4,
    ) -> torch.Tensor:
    """
    Computes the mean and standard deviation of forecasting errors on training data, then uses those to define what 'normality' is: 
    - all points that are no further than `max_std` standard deviations away from the mean are normal
    - all other points are considered outliers.

    Parameters
    ----------
    train_errors : torch.Tensor
        Forecasting errors for training data.
    test_errors : torch.Tensor
        Forecasting errors for testing data.
    max_std : int 
        The 'normal' distance accepted from the mean of training errors, measured in standard deviations 
        of forecasting training errors.

    Returns
    -------
    labels: torch.Tensor
        A tensor with the labels for the testing errors given as parameters.
    """
    labels = []
    mu = torch.mean(train_errors).item()
    std = torch.std(train_errors).item()

    for test_error in test_errors:
        is_anomaly = int((test_error > (mu + max_std * std)).item())
        labels.append(is_anomaly)

    labels = torch.tensor(labels)
    return labels


def get_anomaly_intervals(
        anomaly_labels: torch.Tensor,
        anomaly_scores: t.Optional[torch.Tensor] = None,
        prune: bool = False, 
        threshold: float = 0.1,
    ) -> t.List[t.Tuple[int, int]]:
    """
    Extracts the start and inclusive end index intervals of consecutive  anomalies (1s) from a binary tensor.
    
    To address the problem of high false positive rate, it uses a pruning technique that selects the maximum anomaly score from each sequence,
    sorts those value in a descending order, then it computes the decreasing percent of each anomaly sequence. When it finds a decreasing percent
    lower than `threshold`, it relabels all the remaining sequences as normal.       
    """
    if not isinstance(anomaly_labels, torch.Tensor):
        anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.float)

    # pad both ends with 0 to catch anomalies that start at index 0 or end at the final index
    padded = torch.cat([torch.tensor([0], device=anomaly_labels.device), anomaly_labels, torch.tensor([0], device=anomaly_labels.device)])
    
    # find transitions: 
    # diff == 1  means 0 -> 1 (start)
    # diff == -1 means 1 -> 0 (end)
    diff = padded[1:] - padded[:-1]
    
    starts = torch.where(diff == 1)[0].tolist()
    ends = (torch.where(diff == -1)[0] - 1).tolist()
    intervals = list(zip(starts, ends))
    # only keep intervals that have at least 2 points
    intervals = np.array([intervals[i] for i, (start, end) in enumerate(intervals) if (end - start) >= 1])

    if prune is True:
        # extract the maximum anomaly score for each interval
        intervals_max = [max(anomaly_scores[start:end+1]).item() for start, end in intervals]
        # sort the intervals and intervald maximums
        idx = np.argsort(intervals_max)[::-1]

        intervals = np.array(intervals)[idx]
        intervals_max = np.array(intervals_max)[idx]

        for i in range(1, len(intervals_max)):
            # compute the decrease percent
            pi = (intervals_max[i-1] - intervals_max[i]) / intervals_max[i-1]

            if pi < threshold:
                # prune the sequences following the first decrease percent that is too low
                intervals = intervals[:i]
                break

    intervals = intervals.tolist()
    return intervals


def merge_anomaly_intervals(intervals: t.List[t.Tuple[int, int]]) -> t.List[t.Tuple[int, int]]:
    """Merges the given list of intervals into a list of continous intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        # overlapping or consecutive
        if start <= prev_end + 1:  
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def detect_contextual_anomalies(anomaly_scores: torch.Tensor, q: float = 0.995) -> torch.Tensor:
    """
    Identifies contextual anomalies in time-series data using a dynamic, sliding-window thresholding technique.
    
    Computes a local threshold for overlapping windows of size `T / 3` moving at steps of `T / 30`, where `T` is the size of the anomaly scores tensor. 
    A data point is flagged as an anomaly (1) if its score deviates from the local window mean by more than `max_std` standard deviations.
    
    Returns
    -------
    anomaly_intervals: torch.Tensor
        A tensor with the anomaly intervals after detection and pruning.
    """
    T = anomaly_scores.shape[0]
    threshold_sw = T // 3
    threshold_ss = T // (3 * 10)     
    anomaly_intervals = []

    for i in range(0, T - threshold_sw + 1, threshold_ss):
        window = anomaly_scores[i:i+threshold_sw]
        threshold = torch.quantile(window, q)
        # use time-series size as labels size for easier index manipulation
        window_anomaly_labels = torch.zeros((T,))

        for j, anomaly_score in enumerate(window):
            is_anomaly = int((anomaly_score > threshold).item())
            window_anomaly_labels[i + j] = is_anomaly

        # merge the point anomalies into collective anomalies
        window_anomaly_intervals = get_anomaly_intervals(window_anomaly_labels, anomaly_scores, prune=True)
        anomaly_intervals.extend(window_anomaly_intervals)

    # merge the collective anomalies between them, as they might intersect 
    # due to being generated by overlapping thresholding windows
    anomaly_intervals = merge_anomaly_intervals(anomaly_intervals)

    return anomaly_intervals
