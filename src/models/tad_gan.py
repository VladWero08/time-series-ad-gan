"""
TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks
===========================================================================
An independent implementation in PyTorch of a GAN-based reconstruction method for time-series anomaly detection, 
    following Geiger et al., “TadGAN: Time series anomaly detection using generative adversarial networks.” 
    In: 2020 ieee international conference on big data (big data). IEEE. 2020, pp. 33–43.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import typing as t
import numpy as np

from torch.utils.data import DataLoader

from src.utils.data import normalization, intervals_to_points, split, agg_reconstructions
from src.utils.errors import point_wise_error, gradient_penalty, agg_gan_errors
from src.utils.evaluation import evaluate_point_anomalies, evaluate_collective_anomalies, plot_performance
from src.utils.detection import get_anomaly_intervals, detect_point_anomalies, detect_contextual_anomalies
from src.utils.signals import SignalsReconstructDataset


class GeneratorG(nn.Module):
    def __init__(
        self,
        latent_size: int = 20,
        n_features: int = 1,
        hidden_size: int = 20, 
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_size*2,
            out_features=latent_size,   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, n_features)
        out, (hn, cn) = self.lstm(x)
        # (batch, signal_size, hidden_size)
        out = self.fc(out)
        # (batch, signal_size, latent_size)        
        return out


class GeneratorF(nn.Module):
    def __init__(
        self,
        latent_size: int = 20,
        n_features: int = 1,
        hidden_size: int = 64, 
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


class DiscriminatorX(nn.Module):
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
        out = x.permute(0, 2, 1)
        # (batch, n_features, signal_size)
        out = self.conv(out)
        # (batch, 1, signal_size)
        out = out.squeeze(1)
        # (batch, signal_size)
        out = self.fc(out)            
        # (batch, 1)          
        return out


class DiscriminatorZ(nn.Module):
    def __init__(
        self,
        signal_size: int = 100,
        latent_size: int = 20,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=latent_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1),
        )
        self.fc = nn.Linear(in_features=signal_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, latent_size)
        out = x.permute(0, 2, 1)
        # (batch, latent_size, signal_size)
        out = self.conv(out)
        # (batch, 1, signal_size)
        out = out.squeeze(1)
        # (batch, signal_size)
        out = self.fc(out)            
        # (batch, 1)          
        return out


class TadGAN(nn.Module):
    def __init__(
        self, 
        signal_size: int = 100,
        latent_size: int = 20,
        n_features: int = 1,
        g_hidden_size: int = 20,
        f_hidden_size: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.signal_size = signal_size
        self.latent_size = latent_size
        self.n_features = n_features
        self.g_hidden_size = g_hidden_size
        self.f_hidden_size = f_hidden_size

        self.gg = GeneratorG(self.latent_size, self.n_features, self.g_hidden_size).to(device)
        self.gf = GeneratorF(self.latent_size, self.n_features, self.f_hidden_size).to(device)
        self.dx = DiscriminatorX(self.signal_size, self.n_features).to(device)
        self.dz = DiscriminatorZ(self.signal_size, self.latent_size).to(device)
            

def train(
    tadgan: TadGAN,
    train_dl: DataLoader,
    epochs: int = 2000,
    n_critics: int = 5,
    lambda_gp: float = 10.0,
    lambda_fc: float = 10.0,
    lr_gg: float = 1e-6,
    lr_gf: float = 1e-6,
    lr_dx: float = 1e-6,
    lr_dz: float = 1e-6,
    beta1: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    optim_gg = optim.Adam(tadgan.gg.parameters(), lr=lr_gg, betas=(beta1, 0.999))
    optim_gf = optim.Adam(tadgan.gf.parameters(), lr=lr_gf, betas=(beta1, 0.999))
    optim_dx = optim.Adam(tadgan.dx.parameters(), lr=lr_dx, betas=(beta1, 0.999))
    optim_dz = optim.Adam(tadgan.dz.parameters(), lr=lr_dz, betas=(beta1, 0.999))

    mse = nn.MSELoss()

    for epoch in range(epochs):
        loss_gg_epoch = 0.0
        loss_gf_epoch = 0.0
        loss_fc_epoch = 0.0
        loss_dx_epoch = 0.0
        loss_dz_epoch = 0.0

        for _ in range(n_critics):
            # set each model to its corresponding mode to update the generators
            tadgan.dx.train(); tadgan.dz.train(); tadgan.gg.eval(); tadgan.gf.eval()

            for real_x_data in train_dl:
                batch_size = real_x_data.size(0)
                real_x_data = real_x_data.to(device)
                real_z_data = torch.randn(batch_size, tadgan.signal_size, tadgan.latent_size).to(device)
                
                ###################################################
                # (1) Update Dx Network: maximize Dx(X) - Dx(F(z)) 
                ###################################################
                optim_dx.zero_grad()

                # generate fake data from X 
                fake_x_data = tadgan.gf(real_z_data)

                loss_real = tadgan.dx(real_x_data).mean()
                loss_fake = tadgan.dx(fake_x_data.detach()).mean()
                loss_gp = gradient_penalty(tadgan.dx, real_x_data, fake_x_data.detach(), device)
                loss_dx = loss_fake - loss_real + lambda_gp * loss_gp
                loss_dx.backward()

                optim_dx.step()

                ###################################################
                # (2) Update Dz Network: maximize Dz(Z) - Dz(G(X))
                ###################################################
                optim_dz.zero_grad()
                
                # generate fake data from Z
                fake_z_data = tadgan.gg(real_x_data)

                loss_real = tadgan.dz(real_z_data).mean()
                loss_fake = tadgan.dz(fake_z_data.detach()).mean()
                loss_gp = gradient_penalty(tadgan.dz, real_z_data, fake_z_data.detach(), device)
                loss_dz = loss_fake - loss_real + lambda_gp * loss_gp
                loss_dz.backward()

                optim_dz.step()

                # metrics
                loss_dx_epoch += loss_dx.item()
                loss_dz_epoch += loss_dz.item()

        for real_x_data in train_dl:
            batch_size = real_x_data.size(0)
            real_x_data = real_x_data.to(device)
            real_z_data = torch.randn(batch_size, tadgan.signal_size, tadgan.latent_size).to(device)

            # set each model to its corresponding mode to update the generators
            tadgan.gg.train(); tadgan.gf.train(); tadgan.dz.eval(); tadgan.dx.eval()
            optim_gg.zero_grad()
            optim_gf.zero_grad()

            #############################################################
            # (3) Update G Network: maximize Dz(G(x)) - MSE(x - F(G(x)))
            #############################################################            

            fake_z_data = tadgan.gg(real_x_data)
            reconstructed_x_data = tadgan.gf(fake_z_data)
            # compute GAN loss for generator G
            loss_gg_gan = -tadgan.dz(fake_z_data).mean()
            # compute forward cycle loss
            loss_forward_cycle = mse(real_x_data, reconstructed_x_data).mean()
            loss_gg_total = loss_gg_gan + lambda_fc * loss_forward_cycle
            loss_gg_total.backward()

            ##########################################
            # (4) Update F Network: maximize Dx(F(z))
            ##########################################

            fake_x_data = tadgan.gf(real_z_data)
            # compute GAN loss for generator F
            loss_gf_gan = -tadgan.dx(fake_x_data).mean()
            loss_gf_gan.backward()

            # backward pass for both optimizers
            optim_gg.step()
            optim_gf.step()

            # metrics
            loss_gg_epoch += loss_gg_gan.item()
            loss_gf_epoch += loss_gf_gan.item()
            loss_fc_epoch += loss_forward_cycle.item()

        loss_gg_epoch /= len(train_dl)
        loss_gf_epoch /= len(train_dl)
        loss_fc_epoch /= len(train_dl)
        loss_dx_epoch /= (len(train_dl) * n_critics) 
        loss_dz_epoch /= (len(train_dl) * n_critics) 

        if verbose and (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d} | GG Loss: {loss_gg_epoch:.6f} | GF Loss: {loss_gf_epoch:.6f} | FC Loss: {loss_fc_epoch:.6f} | DX Loss: {loss_dx_epoch:.6f} | DZ Loss: {loss_dz_epoch:.6f}")


def test(
    model: TadGAN,
    dl: DataLoader,
    sw: int = 100,
    ss: int = 1,
    rec_error_funcs: t.List[t.Tuple[str, t.Callable]] = [("point", point_wise_error)],
    alpha: float = 0.5,
    device: str = "cpu",
) -> t.Tuple[torch.Tensor, t.Dict[str, torch.Tensor]]:
    # dictionary with all the reconstruction error functions used
    X_rec = []
    disc_scores = []
    rec_errors = {name: [] for name, _ in rec_error_funcs}

    model.eval()
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            # X -> G(X) -> F(G(X))
            xb_rec = model.gf(model.gg(xb)).detach()
            xb_disc_scores = model.dx(xb_rec)

            for name, error_func in rec_error_funcs:
                xb_rec_errors = error_func(xb, xb_rec)
                rec_errors[name].append(xb_rec_errors.detach())

            X_rec.append(xb_rec.detach())
            disc_scores.append(xb_disc_scores.detach())

    # concatenate along the batch dimension to maintain chronological sequence
    X_rec = torch.cat(X_rec, dim=0)
    X_rec = agg_reconstructions(X_rec, sw, ss).cpu()

    disc_scores = torch.cat(disc_scores, dim=0).cpu()
    rec_errors = {
        name: torch.cat(rec_error, dim=0).cpu() 
        for name, rec_error in rec_errors.items()
    }
    anomaly_scores = {
        name: agg_gan_errors(rec_error, disc_scores, sw=sw, ss=ss, alpha=alpha) 
        for name, rec_error in rec_errors.items()
    }

    return X_rec, anomaly_scores


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
    rec_error_funcs: t.List[t.Tuple[str, t.Callable]] = [("point", point_wise_error)],
    batch_size: int = 64,
    n_critics: int = 5,
    lr_gg: float = 1e-6,
    lr_gf: float = 1e-6,
    lr_dx: float = 1e-6,
    lr_dz: float = 1e-6,
    device: str = "cpu",
    anomaly_type: str = "point",
    verbose: bool = True,
) -> t.Tuple[float, float, float]:
    """
    Runs the complete time-series anomaly detection process.

    Parameters
    ----------
    X : np.ndarray
        The input data features for training or the entire dataset.
    y : np.ndarray
        The ground-truth anomaly labels for the input data.
    X_test : np.ndarray, optional
        A separate set of input features used only for testing.
    y_test : np.ndarray, optional
        The ground-truth anomaly labels for the test data.
    train_ratio : float, default 0.60
        The percentage of data to use for training the model.
    sw : int, default 100
        Sliding window size, how many past time steps the model looks at.
    ss : int, default 1
        Step size, how many steps the window moves forward each time).
    latent_size : int, default 20
        The size of the latent dimensional space in which the data will be encoded into, and decoded from.
    epochs : int, default 2000
        The maximum number of times the model loops through the entire training set.
    rec_error_funcs : t.List[t.Tuple[str, t.Callable]], default [("point", point_wise_error)]
        The list of tuple formed of reconstruction error name and function used to evaluate the model.
    batch_size : int, default 64
        The number of data samples processed together at one time.
    lr_gg: float, default 1e-6
        The learning rate for the optimizer of the generator G: X -> Z.
    lr_gf: float, default 1e-6
        The learning rate for the optimizer of the generator F: Z -> G.
    lr_dx: float, default 1e-6
        The learning rate for the optimizer of the discriminator X.
    lr_dz: float, default 1e-6
        The learning rate for the optimizer of the discriminator Z.
    device : str, default "cpu"
        The hardware to run the code on ("cpu" or "cuda").
    anomaly_type : str, default "point"
        The type of anomaly detection to use ("point" or "contextual").
    verbose : bool, default True
        If True, prints progress updates and final metrics.
    
    Returns
    -------
    t.Tuple[float, float, float]: [precision, recall, f1]
    """
    cutoff = sw // 2
    if X_test is None or y_test is None:
        val_ratio = 0
        X_train, y_train, X_test, y_test = split(X, y, sw, train_ratio, val_ratio, type="reconstruct")
    else:
        X_train, y_train = X, y
        y_test = y_test[cutoff:-cutoff]

    # normalization of the data
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    X_train = normalization(X_train, X_train_min, X_train_max)
    X_test = normalization(X_test, X_train_min, X_train_max)

    # build sliding windows for train, val, test
    train_ds = SignalsReconstructDataset(X_train, sw=sw, ss=ss)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_ds  = SignalsReconstructDataset(X_test, sw=sw, ss=ss)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    _, n_features = X.shape
    model = TadGAN(signal_size=sw, latent_size=latent_size, n_features=n_features, device=device)
    train(
        model, 
        train_dl, 
        epochs=epochs, 
        n_critics=n_critics,
        lr_gg=lr_gg,
        lr_gf=lr_gf,
        lr_dx=lr_dx,
        lr_dz=lr_dz,   
        device=device,
        verbose=verbose
    )
    model.gg.eval(); model.gf.eval(); model.dx.eval(); model.dz.eval()

    metrics = {name: [] for name, _ in rec_error_funcs}
    # infer the model for test data
    X_test_preds, y_test_anomaly_scores = test(model, test_dl, sw=sw, ss=ss, rec_error_funcs=rec_error_funcs, device=device)
    y_test_anomaly_scores = {name: rec_error[cutoff:-cutoff] for name, rec_error in y_test_anomaly_scores.items()}
    # infer the model for train data
    _, y_train_anomaly_scores = test(model, train_dl, sw=sw, ss=ss, rec_error_funcs=rec_error_funcs, device=device)
    y_train_anomaly_scores = {name: rec_error[cutoff:-cutoff] for name, rec_error in y_train_anomaly_scores.items()}

    for rec_error_func_name in metrics:
        match anomaly_type:
            case "point":
                # compute the test labels based on the forecasting errors obtained in training
                y_test_hat = detect_point_anomalies(y_train_anomaly_scores[rec_error_func_name], y_test_anomaly_scores[rec_error_func_name])

                precision, recall, f1 = evaluate_point_anomalies(y_test, y_test_hat)
            case "contextual":
                # compute the test anomaly sequences from test labels 
                y_test_intervals = get_anomaly_intervals(y_test)
                # compute the test anomaly sequences from test forecast errors
                y_test_hat_intervals = detect_contextual_anomalies(y_test_anomaly_scores[rec_error_func_name])
                y_test_hat = intervals_to_points(y_test_hat_intervals, y_test.shape[0])

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
    anomaly_idx = [800, 801, 802, 929, 930, 931]
    ts[anomaly_idx] += 15.0
    y[anomaly_idx] = 1

    run_pipeline(
        ts,
        y,
        sw=100,
        ss=1,
        epochs=200,
        batch_size=64,
        n_critics=2,
        lr_gg=1e-4,
        lr_gf=1e-4,
        lr_dx=1e-4,
        lr_dz=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        anomaly_type="contextual",
        verbose=True,
    )
