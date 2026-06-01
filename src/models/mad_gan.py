import torch
import torch.nn as nn
import torch.optim as optim
import typing as t
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models.signals import SignalsReconstructDataset 
from models.utils import point_wise_error, agg_reconstruction_errors

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
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, latent_size)
        out, (hn, cn) = self.lstm(x)
        # (batch, signal_size, hidden_size * 2)
        out = self.fc(out)
        # (batch, signal_size, n_features)
        out = self.activation(out)
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
        self.activation = nn.Sigmoid()

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
        out = self.activation(out)
        return out


def gradient_penalty(d: Discriminator, real: torch.Tensor, fake: torch.Tensor, device: str) -> torch.Tensor:
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
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()

    return gp


def train(
    train_dl: DataLoader,
    signal_size: int = 100,
    latent_size: int = 20,
    n_features: int = 1,
    epochs: int = 2000,
    n_critics: int = 5,
    lambda_gp: float = 10.0,
    lr_g: float = 1e-5,
    lr_d: float = 1e-5,
    beta1: float = 0.5,
    device: str = "cpu",
) -> t.Tuple[Generator, Discriminator]:
    g = Generator(latent_size=latent_size, n_features=n_features)
    d = Discriminator(signal_size=signal_size, n_features=n_features)

    optim_g = optim.Adam(g.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optim_d = optim.Adam(d.parameters(), lr=lr_d, betas=(beta1, 0.999))

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

                d.zero_grad()
                
                # generate fake batch
                z = torch.randn(batch_size, signal_size, latent_size).to(device)
                fake_data = g(z).detach()
                
                loss_real = d(real_data).mean()
                loss_fake = d(fake_data).mean()
                loss_gp = gradient_penalty(d, real_data, fake_data.detach(), device)

                # update the discriminator by minimizing -(D(X) - D(G(z))) + GP
                loss_d = loss_fake - loss_real + lambda_gp * loss_gp
                loss_d.backward()
                optim_d.step()

                # update epoch metrics
                loss_d_epoch += loss_d.item()

        ##########################################
        # (2) Update G network: maximize (D(G(z)))
        ##########################################
        for _ in range(len(train_dl)):
            g.zero_grad()

            batch_size = real_data.size(0)
            z = torch.randn(batch_size, signal_size, latent_size).to(device)
            fake_data = g(z)
            
            loss_g = -d(fake_data).mean()
            loss_g.backward()
            optim_g.step()

            # update epoch metrics
            loss_g_epoch += loss_g.item()

        loss_d_epoch /= (len(train_dl) * n_critics)
        loss_g_epoch /= len(train_dl)
        print(f"Epoch {epoch+1:3d} | D Loss: {loss_d_epoch:.6f} | G Loss: {loss_g_epoch:.6f}")

    return g, d


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

        generated = g(z).squeeze(0)
        loss = mse(generated, data)
        torch.mean(loss).backward()
        optimizer.step()

    return z.detach()


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    sw: int = 100,
    ss: int = 1,
    latent_size: int = 20,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu"
) -> np.ndarray:
    T, n_features = X.shape
    train_end = int(T * train_ratio)

    X_train, y_train    = X[:train_end], y[:train_end]
    X_test, y_test      = X[train_end:], y[train_end:]      

    print(f"Series shape : {X.shape}")
    print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
    print(f"Test         : timesteps {train_end} -> {T}  ({T - train_end} steps)\n")

    # build sliding windows for train, val, test
    train_ds = SignalsReconstructDataset(X_train, sw=sw, ss=ss)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds  = SignalsReconstructDataset(X_test, sw=sw, ss=ss)

    # build and train discriminator and generator
    g, d = train(
        train_dl, 
        signal_size=sw, 
        latent_size=latent_size, 
        n_features=n_features, 
        epochs=epochs,
        device=device
    )
    g.eval()
    d.eval()

    # compute reconstruction errors and plot them
    y_test_z = find_best_latent(test_ds.X, g)
    y_test_rec = g(y_test_z).detach()
    y_test_sw_errors = point_wise_error(y_test_rec, test_ds.X)
    y_test_errors = agg_reconstruction_errors(y_test_sw_errors, sw, ss)

    plt.figure(figsize=(10, 5))
    plt.plot(X[train_end:], color="blue", label="Time-Series")
    plt.plot(y_test_errors.detach(), color="orange", label="Reconstruction Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(999)

    T, n_features = 2000, 1
    ts = np.random.randn(T, n_features).cumsum(axis=0) * 0.1
    y = np.zeros(T)

    # inject spike anomalies
    anomaly_idx = [1790, 1800, 1930]
    ts[anomaly_idx] += 15.0
    y[anomaly_idx] = 1

    # min-max normalization to [-1, 1]
    ts_min = ts.min(axis=0)
    ts_max = ts.max(axis=0)
    ts_normalized = 2 * (ts - ts_min) / (ts_max - ts_min) - 1

    run_pipeline(
        ts_normalized,
        y,
        sw=100,
        ss=1,
        epochs=1,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
