import numpy as np
import torch
import typing as t
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde, zscore

def point_wise_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.abs(y_pred - y_true)


def area_wise_error(y_pred: torch.Tensor, y_true: torch.Tensor, l: int = 3) -> torch.Tensor:    
    st = []
    n = y_pred.shape[0]

    for i in range(n):
        min_idx = max(0, i - l)
        max_idx = min(n, i + l + 1)

        abs_diff = torch.abs(y_pred[min_idx:max_idx] - y_true[min_idx:max_idx]).float()
        mean_abs_diff = abs_diff.mean(dim=0)

        st.append(mean_abs_diff)

    st = torch.stack(st)
    return st


def dtw_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    n, m = y_pred.shape[0], y_true.shape[0]
    
    # initialize the dp matrix for dtw
    dtw_matrix = torch.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = float("inf")
    dtw_matrix[0, 0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            # reconstruction error of the current point
            diff = abs(y_pred[i] - y_true[j])
            # lowest error from previous time-series data
            min_ = torch.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = diff + min_

    return dtw_matrix[n, m]


def agg_reconstruction_errors(reconstruction_errors: torch.Tensor, sw: int, ss: int) -> torch.Tensor:
    """
    Aggregates sliding window reconstruction errors to a per-timestep error by computing the median over all windows that contain each timestep.

    Parameters
    ----------
    reconstruction_errors : torch.Tensor
        Reconstruction error per window and per position within that window.
    sw : int
        Sliding window size.
    ss : int
        Sliding window step size.

    Returns
    -------
    agg_errors : torch.Tensor of shape (T,)
        Median aggregated error value for each original timestep.
    """
    N = reconstruction_errors.shape[0]
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
            reconstruction_errors[k, i - k * ss]
            for k in range(k_min, k_max + 1)
        ])
        median_error = torch.median(errors_at_i)
        agg_errors.append(median_error)

    return torch.stack(agg_errors)


def agg_gan_errors(
    reconstruction_errors: torch.Tensor,
    discriminator_errors: torch.Tensor,
    sw: int,
    ss: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    N = discriminator_errors.shape[0]
    T = (N - 1) * ss + sw

    # convert to numpy and invert the discriminator scores
    inverted_discriminator_errors = -1 * np.array(discriminator_errors).flatten()
    reconstruction_errors = reconstruction_errors.numpy()

    # lists where the errors will be merged together from multiple windows
    agg_re = []
    agg_dx = []

    for i in range(T):
        k_min = max(0, int(np.ceil((i - sw + 1) / ss)))
        k_max = min(N - 1, int(np.floor(i / ss)))
        
        # median of reconstruction errors
        errors_at_t = [reconstruction_errors[k, i - k * ss] for k in range(k_min, k_max + 1)]
        median_re = np.median(errors_at_t)
        agg_re.append(median_re)
        
        # kde of discriminator scores
        discriminators_at_t = [inverted_discriminator_errors[k] for k in range(k_min, k_max + 1)]
        
        if len(discriminators_at_t) > 1 and np.var(discriminators_at_t) > 1e-8:
            # fir kernel density estimation over the overlapping window discriminator scores
            kde = gaussian_kde(discriminators_at_t)

            # sample the maximum from the distribution
            space = np.linspace(min(discriminators_at_t), max(discriminators_at_t), 100)
            kde_max_score = space[np.argmax(kde(space))]
            agg_dx.append(kde_max_score)
        else:
            # use the mean if KDE cannot be applied
            agg_dx.append(np.mean(discriminators_at_t))

    Z_re = zscore(np.array(agg_re))
    Z_dx = zscore(np.array(agg_dx))
    anomaly_score = alpha * Z_re + (1 - alpha) * Z_dx

    return torch.from_numpy(anomaly_score).float()


def gradient_penalty(d, real: torch.Tensor, fake: torch.Tensor, device: str) -> torch.Tensor:
    """
    Interpolates the given `real` data with the `fake` data using a convex combination, and afterwards it feeds the
    discriminator `d` with the interpolated samples. It computes the norm of the gradients for the discriminator w.r.t the interpolated samples.
    
    Paramters
    ---------
    d: Discriminator | DiscriminatorG | DiscriminatorF
        The discriminator for which the gradient penalty is computed.
    real: torch.Tensor
        Batch of real time-series data.
    fake: torch.Tensor
        Batch of fake time-series data.
    device: str
        Device where to move the tensors. ("cpu" or "cuda")

    Returns
    -------
    gp: torch.Tensor
        Gradient penalty computed for the interpolated samples.
    """
    batch_size = real.size(0)

    # random interpolation weight α for each sample in the batch
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real)

    # interpolate between real and fake
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = d(interpolated)

    # compute gradients of critic output w.r.t. interpolated input
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # flatten gradients and compute their norm
    # NOTE: reshape ensures that the tensors is contiguous in memory by making a copy of it
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()

    return gp


def detect_point_anomalies(
        train_errors: torch.Tensor, 
        test_errors: torch.Tensor,
        max_std: int = 4,
    ) -> torch.Tensor:
    """
    Computes the mean and standard deviation of forecasting errors on training data, then uses those to define what 'normality' is: 
    - all points that are no further than `max_std` standard deviations away from the mean are normal
    - all other points are considered outliers.

    Parameters:
    -----------
    train_errors: torch.Tensor
        Forecasting errors for training data.
    test_errors: torch.Tensor
        Forecasting errors for testing data.
    max_std: int 
        The 'normal' distance accepted from the mean of training errors, measured in standard deviations 
        of forecasting training errors.

    Returns:
    --------
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


def overlaps(a: t.Tuple[int, int], b: t.Tuple[int, int]) -> bool:
    """Checks if the given intervals are overlapping."""
    return a[0] <= b[1] and b[0] <= a[1]


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


def detect_contextual_anomalies(
        anomaly_scores: torch.Tensor,
        max_std: int = 4,
    ) -> torch.Tensor:
    """
    Identifies contextual anomalies in time-series data using a dynamic, sliding-window thresholding technique.
    
    Computes a local threshold for overlapping windows of size `T / 3` moving at steps of `T / 30`, where `T` is the size of the anomaly scores tensor. 
    A data point is flagged as an anomaly (1) if its score deviates from the local window mean by more than `max_std` standard deviations.
    
    Returns:
    --------
    anomaly_intervals: torch.Tensor
        A tensor with the anomaly intervals after detection and pruning.
    """
    T = anomaly_scores.shape[0]
    threshold_sw = T // 3
    threshold_ss = T // (3 * 10)     
    anomaly_labels = [0 for _ in range(T)]

    for i in range(0, T - threshold_sw + 1, threshold_ss):
        window = anomaly_scores[i:i+threshold_sw]
        window_mean = torch.mean(window)
        window_std = torch.std(window) + 1e-8

        for j, anomaly_score in enumerate(window):
            is_anomaly = int((anomaly_score > (window_mean + max_std * window_std)).item())
            anomaly_labels[i + j] |= is_anomaly

    anomaly_labels = torch.tensor(anomaly_labels)
    anomaly_intervals = get_anomaly_intervals(anomaly_labels, anomaly_scores, prune=True)

    return anomaly_intervals


def evaluate_point_anomalies(y_true: torch.Tensor, y_predict: torch.Tensor) -> t.Tuple[float]:
    """
    Computes precision, recall, and F1 score for point anomaly detection.

    A point anomaly is an isolated anomalous timestep with no adjacent anomalies.
    Only isolated anomalies in y_true are evaluated; consecutive anomalies are ignored.

    Returns
    -------
    precision, recall, f1 : Tuple[float]
    """
    # handle the case when the time series does not contain any anomalies
    if len(y_true) == 0 and len(y_predict) == 0:
        return 1.0, 1.0, 1.0
    
    N = y_true.shape[0]

    def is_point_anomaly(i: int) -> bool:
        "An anomaly is a point anomaly if the surrounding points are not anomalies. Check whether the given index contains a point anomaly."
        if i == 0:
            return y_true[i] == 1 and y_true[i + 1] == 0
        elif i == N - 1:
            return y_true[i] == 1 and y_true[i - 1] == 0
        
        return y_true[i] == 1 and y_true[i - 1] == 0 and y_true[i + 1] == 0

    tp = fp = fn = 0

    for i in range(N):
        if is_point_anomaly(i):
            if y_predict[i] == 1:
                # y_true[i] = 1 and y_predict[i] = 1
                tp += 1
            else:
                # y_true[i] = 1 and y_predict[i] = 0
                fn += 1
        else:
            if y_predict[i] == 1:
                # y_true[i] = 0 and y_predict[i] = 1
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_collective_anomalies(y_true_intervals: torch.Tensor, y_predict_intervals: torch.Tensor) -> t.Tuple[float]:
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
    if len(y_true_intervals) == 0 and len(y_predict_intervals) == 0:
        return 1.0, 1.0, 1.0
    
    tp = fp = fn = 0

    # TP / FN: for each true anomaly interval, check if any predicted interval overlaps it
    for true_interval in y_true_intervals:
        if any(overlaps(true_interval, pred_interval) for pred_interval in y_predict_intervals):
            tp += 1
        else:
            fn += 1

    # FP: for each predicted interval, check if any anomaly interval overlaps it
    for pred_interval in y_predict_intervals:
        if not any(overlaps(pred_interval, true_interval) for true_interval in y_true_intervals):
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1    


def intervals_to_points(y_intervals: t.List[t.List], n_labels: int) -> torch.Tensor:
    point_labels = torch.zeros(n_labels)
    for y_interval in y_intervals:
        start, end = y_interval[0], y_interval[1]
        point_labels[start:end] = 1
    return point_labels


def plot_performance(
    X: torch.Tensor,
    X_preds: torch.Tensor,
    y: torch.Tensor,
) -> None:
    def format(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if len(x.shape) > 1:
            x = x.flatten()
        return x

    X = format(X)
    X_preds = format(X_preds)
    y = format(y)         
    y_idx = y == 1

    plt.figure(figsize=(10, 5))
    plt.plot(X, color="blue", label="Time-Series")
    plt.plot(X_preds, color="orange", label="Prediction")
    plt.scatter(np.where(y_idx)[0], X[y_idx], color="red", s=15, zorder=3, label="Anomaly")
    plt.xlabel("Time Step")
    plt.legend()
    plt.tight_layout()
    plt.show()    


if __name__ == "__main__":
    anomaly_scores = torch.zeros(200)
    anomaly_scores[[1, 2, 3, 4]] = 1
    anomaly_scores[[30, 31, 32, 34]] = 1
    anomaly_scores[[90, 91, 92, 93]] = 1
    anomaly_scores[[140, 141, 142, 143]] = 1

    intervals = get_anomaly_intervals(anomaly_scores)
    print(intervals)
    points = intervals_to_points(intervals, len(anomaly_scores))
    print(points)