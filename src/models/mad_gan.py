import torch
import torch.nn as nn
import torch.optim as optim
import typing as t
import numpy as np
from torch.utils.data import DataLoader

from src.utils.preprocess import normalization, intervals_to_points, split, agg_reconstructions
from src.utils.errors import point_wise_error, area_wise_error, dtw_error, gradient_penalty, agg_gan_errors
from src.utils.evaluation import evaluate_point_anomalies, evaluate_collective_anomalies, plot_performance
from src.utils.detection import get_anomaly_intervals, detect_point_anomalies, detect_contextual_anomalies
from src.utils.signals import SignalsReconstructDataset


class Generator(nn.Module):
    def __init__(
        self,
        latent_size: int = 20,
        n_features: int = 1,
        hidden_size: int = 64
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_size*2,
            out_features=n_features,   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, latent_size)
        out, (hn, cn) = self.lstm(x)
        # (batch, signal_size, hidden_size * 2)
        out = self.fc(out)
        # (batch, signal_size, n_features)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        signal_size: int = 100,
        n_features: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1),
        )
        self.fc = nn.Linear(in_features=signal_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, n_features)
        x = x.permute(0, 2, 1)
        # (batch, n_features, signal_size)
        out = self.conv(x)
        # (batch, 1, signal_size)
        out = out.squeeze(1)
        # (batch, signal_size)
        out = self.fc(out)            
        # (batch, 1)
        return out


class MADGAN(nn.Module):
    def __init__(
            self, 
            signal_size: int = 100,
            latent_size: int = 20,
            n_features: int = 1,
            hidden_size: int = 64,
            device: str = "cpu",
        ) -> None:
            super().__init__()

            self.signal_size = signal_size
            self.latent_size = latent_size
            self.n_features = n_features
            self.hidden_size = hidden_size

            self.g = Generator(self.latent_size, self.n_features, self.hidden_size).to(device)
            self.d = Discriminator(self.signal_size, self.n_features, self.hidden_size).to(device)


def train(
    madgan: MADGAN,
    train_dl: DataLoader,
    epochs: int = 2000,
    n_critics: int = 5,
    lambda_gp: float = 10.0,
    lr_g: float = 1e-5,
    lr_d: float = 1e-5,
    beta1: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    optim_g = optim.Adam(madgan.g.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optim_d = optim.Adam(madgan.d.parameters(), lr=lr_d, betas=(beta1, 0.999))

    for epoch in range(epochs):
        loss_d_epoch = 0.0
        loss_g_epoch = 0.0

        ###############################################
        # (1) Update D Network: maximize D(X) - D(G(z)) 
        ###############################################
        for _ in range(n_critics):
            for real_data in train_dl:
                real_data = real_data.to(device)
                batch_size = real_data.size(0) 

                optim_d.zero_grad()
                
                # generate fake batch
                z = torch.randn(batch_size, madgan.signal_size, madgan.latent_size).to(device)
                fake_data = madgan.g(z)
                
                loss_real = madgan.d(real_data).mean()
                loss_fake = madgan.d(fake_data.detach()).mean()
                loss_gp = gradient_penalty(madgan.d, real_data, fake_data.detach(), device)

                # update the discriminator by minimizing -(D(X) - D(G(z))) + GP
                loss_d = loss_fake - loss_real + lambda_gp * loss_gp
                loss_d.backward()
                optim_d.step()

                # update epoch metrics
                loss_d_epoch += loss_d.item()

        ##########################################
        # (2) Update G network: maximize D(G(z))
        ##########################################
        for _ in range(len(train_dl)):
            optim_g.zero_grad()

            batch_size = real_data.size(0)
            z = torch.randn(batch_size, madgan.signal_size, madgan.latent_size).to(device)
            fake_data = madgan.g(z)
            
            loss_g = -madgan.d(fake_data).mean()
            loss_g.backward()
            optim_g.step()

            # update epoch metrics
            loss_g_epoch += loss_g.item()

        loss_d_epoch /= (len(train_dl) * n_critics)
        loss_g_epoch /= len(train_dl)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | D Loss: {loss_d_epoch:.6f} | G Loss: {loss_g_epoch:.6f}")


def find_best_latent(
    data: torch.Tensor,
    g: Generator,
    signal_size: int = 100,
    latent_size: int = 20,
    iters: int = 1000,
    lr: float = 1e-2,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Finds the latent variable `z` that resembles the given `data` sample the best using the generator; the solution
    minimizes 1 - similarity(data, G(z)), where similarity is the mean over absolute difference.
    """
    batch_size = data.shape[0]

    # initialize z as a learnable parameter
    z = torch.randn(batch_size, signal_size, latent_size, device=device, requires_grad=True)
    data = data.to(device)
    g = g.to(device).eval()
    
    optimizer = torch.optim.Adam([z], lr=lr)
    mse = nn.MSELoss(reduction='none')

    for _ in range(iters):
        optimizer.zero_grad()

        generated = g(z)
        loss = mse(generated, data).mean()
        loss.backward()
        optimizer.step()

    return z.detach()


def test(
    model: MADGAN,
    ds: SignalsReconstructDataset,
    sw: int = 100,
    ss: int = 1,
    rec_error_func: t.Callable = point_wise_error, 
    alpha: float = 0.5,
    device: str = "cpu",
) -> t.Tuple[torch.Tensor]:
    # find the best representation of each signal for the given dataset
    ds_best_z = find_best_latent(
        ds.X, 
        model.g, 
        signal_size=model.signal_size, 
        latent_size=model.latent_size, 
        device=device
    )
    
    with torch.no_grad():
        ds_X_rec = model.g(ds_best_z).detach()
        ds_X_rec_agg = agg_reconstructions(ds_X_rec, sw=sw, ss=ss)

        # compute the reconstruction error (T, sw, n_features)
        rec_errors = rec_error_func(ds.X, ds_X_rec)
        # compute the discriminator score (T, 1)
        disc_scores = model.d(ds_X_rec).detach()
        # compute the final anomaly score
        anomaly_scores = agg_gan_errors(rec_errors, disc_scores, sw=sw, ss=ss, alpha=alpha)

    return ds_X_rec_agg, anomaly_scores


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    X_test: t.Optional[np.ndarray] = None,
    y_test: t.Optional[np.ndarray] = None,
    train_ratio: float = 0.6,
    sw: int = 100,
    ss: int = 1,
    latent_size: int = 20,
    epochs: int = 2000,
    batch_size: int = 64,
    n_critics: int = 5,
    lr_g: float = 1e-5,
    lr_d: float = 1e-5,
    device: str = "cpu",
    anomaly_type: str = "point",
    verbose: bool = True,
) -> t.Dict[str, t.List]:
    cutoff = sw // 2
    if X_test is None or y_test is None:
        val_ratio = 0
        X_train, y_train, X_test, y_test = split(X, y, sw, train_ratio, val_ratio, type="reconstruct")
    else:
        X_train, y_train = normalization(X), y
        X_test = normalization(X_test)
        y_test = y_test[cutoff:-cutoff]

    # build sliding windows for train, val, test
    train_ds = SignalsReconstructDataset(X_train, sw=sw, ss=ss)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds  = SignalsReconstructDataset(X_test, sw=sw, ss=ss)

    # build and train discriminator and generator
    _, n_features = X.shape
    model = MADGAN(signal_size=sw, latent_size=latent_size, n_features=n_features, device=device)
    train(
        model, 
        train_dl, 
        epochs=epochs,
        n_critics=n_critics, 
        lr_g=lr_g,
        lr_d=lr_d,    
        device=device,
        verbose=verbose,
    )
    model.g.eval()
    model.d.eval()

    # for mad-gan, the errors for all aggregation functions will be computed
    metrics = {"point": [], "area": [], "dtw": []}
    rec_error_funcs = [
        (point_wise_error, "point"), 
        (area_wise_error, "area"), 
        (dtw_error, "dtw")
    ]

    for rec_error_func, rec_error_func_name in rec_error_funcs:
        X_test_preds, y_test_anomaly_scores = test(model, test_ds, sw=sw, ss=ss, rec_error_func=rec_error_func, device=device)
        y_test_anomaly_scores = y_test_anomaly_scores[cutoff:-cutoff]

        match anomaly_type:
            case "point":
                # compute the forecasting errors for the train set
                _, y_train_anomaly_scores = test(model, test_ds, sw=sw, ss=ss, rec_error_func=rec_error_func, device=device)
                # compute the test labels based on the forecasting errors obtained in training
                y_test_hat = detect_point_anomalies(y_train_anomaly_scores, y_test_anomaly_scores)

                precision, recall, f1 = evaluate_point_anomalies(y_test, y_test_hat)
            case "contextual":
                # compute the test anomaly sequences from test labels 
                y_test_intervals = get_anomaly_intervals(y_test)
                # compute the test anomaly sequences from test forecast errors
                y_test_hat_intervals = detect_contextual_anomalies(y_test_anomaly_scores)
                y_test_labels = intervals_to_points(y_test_hat_intervals, y_test.shape[0])

                precision, recall, f1 = evaluate_collective_anomalies(y_test_intervals, y_test_hat_intervals)
        
        metrics[rec_error_func_name] = [precision, recall, f1]

        if verbose:
            print()
            print(f"Metrics {rec_error_func_name}")
            print("-------")
            print(f"Precision = {precision:.4f}")
            print(f"Recall = {recall:.4f}")
            print(f"F1 = {f1:.4f}")

    if verbose:
        # for plotting, the test samples and their predictions need to be cutoff to match the predicted labels
        plot_performance(X=X_test[cutoff:-cutoff, 0], X_preds=X_test_preds[cutoff:-cutoff, 0])

    return metrics


if __name__ == "__main__":
    np.random.seed(42)

    T, n_features = 1000, 1
    ts = np.random.randn(T, n_features).cumsum(axis=0) * 0.1
    y = np.zeros(T)

    # inject spike anomalies
    anomaly_idx = [790, 800, 930]
    ts[anomaly_idx] += 15.0
    y[anomaly_idx] = 1

    run_pipeline(
        ts,
        y,
        sw=100,
        ss=1,
        epochs=10,
        batch_size=64,
        lr_g=1e-4,
        lr_d=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        plot=True,
    )
