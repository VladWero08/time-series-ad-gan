import torch
import numpy as np
from scipy.stats import gaussian_kde, zscore


def point_wise_error(X_true: torch.Tensor, X_pred: torch.Tensor) -> torch.Tensor:
    return torch.abs(X_true - X_pred)


def area_wise_error(X_true: torch.Tensor, X_pred: torch.Tensor, l: int = 3) -> torch.Tensor:    
    st = []
    N = X_true.shape[0]

    for i in range(N):
        min_idx = max(0, i - l)
        max_idx = min(N, i + l + 1)

        abs_diff = torch.abs(X_true[min_idx:max_idx] - X_pred[min_idx:max_idx]).float()
        mean_abs_diff = abs_diff.mean(dim=0)

        st.append(mean_abs_diff)

    st = torch.stack(st)
    return st


def dtw_error(X_true: torch.Tensor, X_pred: torch.Tensor, window: int = 10) -> torch.Tensor:
    """
    Computes DTW error per window, per timestep, per feature.

    Parameters
    ----------
    X_true, X_pred : torch.Tensor of shape (N, sw, features)

    Returns
    -------
    st : torch.Tensor of shape (N, sw, features)
    """
    X_true = X_true.cpu()
    X_pred = X_pred.cpu()

    def dtw_distance(ts1: torch.Tensor, ts2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts1, ts2: torch.Tensor of shape (sw, features)

        Returns
        -------
        torch.Tensor of shape (sw, features)
        """
        n, m, features = ts1.shape[0], ts2.shape[0], ts1.shape[1]
        
        dtw_matrix = torch.full((n + 1, m + 1, features), float("inf"), device=ts1.device)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diff = (ts1[i - 1] - ts2[j - 1]).abs()
                min_ = torch.min(
                    torch.stack([
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1]
                    ]), dim=0
                ).values
                dtw_matrix[i, j] = diff + min_

        return dtw_matrix[1:, 1:]  # (sw, features)

    N, sw, features = X_true.shape
    st = torch.zeros(N, sw, features, device=X_true.device)
    dtw_len = (window // 2) * 2 + 1
    dtw_half_len = dtw_len // 2

    # pad the sw dimension for each window: (N, sw, features) -> (N, sw + 2*dtw_half_len, features)
    X_true_pad = torch.nn.functional.pad(X_true, (0, 0, dtw_half_len, dtw_half_len), mode='constant', value=0.0)
    X_pred_pad = torch.nn.functional.pad(X_pred, (0, 0, dtw_half_len, dtw_half_len), mode='constant', value=0.0)

    for i in range(N):
        # local DTW within the window step-by-step along sw
        window_errors = torch.zeros(sw, features, device=X_true.device)
        
        for t in range(sw - dtw_len + 1):
            true_window = X_true_pad[i, t:t+dtw_len]   # (dtw_len, features)
            pred_window = X_pred_pad[i, t:t+dtw_len]   # (dtw_len, features)
            target_idx = t + dtw_half_len

            window_errors[target_idx] = dtw_distance(true_window, pred_window)[dtw_half_len, dtw_half_len]

        st[i] = window_errors

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
    discriminator_scores: torch.Tensor,
    sw: int,
    ss: int,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Aggregates sliding window reconstruction errors to a per-timestep error by computing the median and the discriminator scores to a  
    per-timestep score by extracting the maximum probability from the Gaussian KDE estimation.

    Parameters
    ----------
    reconstruction_errors : torch.Tensor
        Reconstruction error per window and per position within that window.
    discriminator_scores: torch.Tensor
        Discriminator scores per window.
    sw : int
        Sliding window size.
    ss : int
        Sliding window step size.

    Returns
    -------
    agg_errors : torch.Tensor of shape (T,)
        Median aggregated error value for each original timestep.
    """
    N = discriminator_scores.shape[0]
    T = (N - 1) * ss + sw

    # convert to numpy and invert the discriminator scores
    inverted_discriminator_scores = -1 * np.array(discriminator_scores).flatten()
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
        discriminators_at_t = [inverted_discriminator_scores[k] for k in range(k_min, k_max + 1)]
        
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
    d : Discriminator | DiscriminatorG | DiscriminatorF
        The discriminator for which the gradient penalty is computed.
    real : torch.Tensor
        Batch of real time-series data.
    fake : torch.Tensor
        Batch of fake time-series data.
    device : str
        Device where to move the tensors. ("cpu" or "cuda")

    Returns
    -------
    gp : torch.Tensor
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

