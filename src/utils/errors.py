import torch
import numpy as np
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


def dtw_error(y_pred: torch.Tensor, y_true: torch.Tensor, window: int = 10) -> torch.Tensor:
    """
    Reference: https://github.com/sintel-dev/Orion/blob/master/orion/primitives/timeseries_errors.py
    """
    def dtw_distance(ts1: torch.Tensor, ts2: torch.Tensor) -> torch.Tensor:
        n, m = ts1.shape[0], ts2.shape[0]
        
        # initialize the dp matrix for dtw
        dtw_matrix = torch.full((n + 1, m + 1), float("inf"))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # reconstruction error of the current point
                diff = abs(ts1[i - 1] - ts2[j - 1])
                # lowest error from previous time-series data
                min_ = torch.min(torch.stack([
                    dtw_matrix[i - 1, j], 
                    dtw_matrix[i, j - 1], 
                    dtw_matrix[i - 1, j - 1]
                ]))
                dtw_matrix[i, j] = diff + min_

        return dtw_matrix[n, m]

    length_dtw = (window // 2) * 2 + 1
    half_len = length_dtw // 2
    T = y_true.shape[0]

    # pad inputs with zeros    
    pad_shape = (half_len, half_len)
    y_true_pad = torch.nn.functional.pad(y_true, pad_shape, mode='constant', value=0.0)
    y_pred_pad = torch.nn.functional.pad(y_pred, pad_shape, mode='constant', value=0.0)
    
    st = torch.zeros(T, device=y_true.device)
    
    # calculate local DTW within the window step-by-step
    for i in range(T - length_dtw + 1):
        true_window = y_true_pad[i : i + length_dtw]
        pred_window = y_pred_pad[i : i + length_dtw]
        
        # center point index 
        target_idx = i + half_len
        st[target_idx] = dtw_distance(true_window, pred_window)
        
    return st


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
        errors_at_i = [reconstruction_errors[k, i - k * ss] for k in range(k_min, k_max + 1)]
        median_re = np.median(errors_at_i)
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