import torch
import torch.nn as nn
import torch.optim as optim
import typing as t
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models.signals import SignalsReconstructDataset 
from models.utils import gradient_penalty
from models.utils import point_wise_error


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
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, signal_size, latent_size)
        out, (hn, cn) = self.lstm(x)
        # (batch, signal_size, hidden_size * 2)
        out = self.fc(out)
        # (batch, signal_size, n_features)
        out = self.activation(out)
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


def train(
    train_dl: DataLoader,
    signal_size: int = 100,
    latent_size: int = 20,
    n_features: int = 1,
    epochs: int = 2000,
    n_critics: int = 3,
    lambda_gp: float = 10.0,
    lambda_fc: float = 10.0,
    lr_gg: float = 1e-4,
    lr_gf: float = 1e-4,
    lr_dx: float = 1e-4,
    lr_dz: float = 1e-4,
    beta1: float = 0.5,
    device: str = "cpu",
) -> t.Tuple[GeneratorG, GeneratorF, DiscriminatorX, DiscriminatorZ]:
    gg = GeneratorG(latent_size=latent_size, n_features=n_features).to(device)
    gf = GeneratorF(latent_size=latent_size, n_features=n_features).to(device)
    dx = DiscriminatorX(signal_size=signal_size, n_features=n_features).to(device)
    dz = DiscriminatorZ(signal_size=signal_size, latent_size=latent_size).to(device)

    optim_gg = optim.Adam(gg.parameters(), lr=lr_gg, betas=(beta1, 0.999))
    optim_gf = optim.Adam(gf.parameters(), lr=lr_gf, betas=(beta1, 0.999))
    optim_dx = optim.Adam(dx.parameters(), lr=lr_dx, betas=(beta1, 0.999))
    optim_dz = optim.Adam(dz.parameters(), lr=lr_dz, betas=(beta1, 0.999))

    mse = nn.MSELoss()

    for epoch in range(epochs):
        loss_gg_epoch = 0.0
        loss_gf_epoch = 0.0
        loss_fc_epoch = 0.0
        loss_dx_epoch = 0.0
        loss_dz_epoch = 0.0

        for _ in range(n_critics):
            # set each model to its corresponding mode to update the generators
            dx.train(); dz.train(); gg.eval(); gf.eval()

            for real_x_data in train_dl:
                batch_size = real_x_data.size(0)
                real_x_data = real_x_data.to(device)
                real_z_data = torch.randn(batch_size, signal_size, latent_size).to(device)
                
                ################################################
                # (1) Update Dx Network: maximize Dx(X) - Dx(F(z)) 
                ################################################
                optim_dx.zero_grad()

                # generate fake data from X 
                fake_x_data = gf(real_z_data)

                loss_real = dx(real_x_data).mean()
                loss_fake = dx(fake_x_data.detach()).mean()
                loss_gp = gradient_penalty(dx, real_x_data, fake_x_data.detach(), device)
                loss_dx = loss_fake - loss_real + lambda_gp * loss_gp
                loss_dx.backward()

                optim_dx.step()

                ################################################
                # (2) Update Dz Network: maximize Dz(Z) - Dz(G(X))
                ################################################
                optim_dz.zero_grad()
                
                # generate fake data from Z
                fake_z_data = gg(real_x_data)

                loss_real = dz(real_z_data).mean()
                loss_fake = dz(fake_z_data.detach()).mean()
                loss_gp = gradient_penalty(dz, real_z_data, fake_z_data.detach(), device)
                loss_dz = loss_fake - loss_real + lambda_gp * loss_gp
                loss_dz.backward()

                optim_dz.step()

                # metrics
                loss_dx_epoch += loss_dx.item()
                loss_dz_epoch += loss_dz.item()

        for real_x_data in train_dl:
            batch_size = real_x_data.size(0)
            real_x_data = real_x_data.to(device)
            real_z_data = torch.randn(batch_size, signal_size, latent_size).to(device)

            # set each model to its corresponding mode to update the generators
            gg.train(); gf.train(); dz.eval(); dx.eval()
            optim_gg.zero_grad()
            optim_gf.zero_grad()

            #############################################################
            # (3) Update G Network: maximize Dz(G(x)) - MSE(x - F(G(x)))
            #############################################################            

            fake_z_data = gg(real_x_data)
            reconstructed_x_data = gf(fake_z_data)
            # compute GAN loss for generator G
            loss_gg_gan = -dz(fake_z_data).mean()
            # compute forward cycle loss
            loss_forward_cycle = mse(real_x_data, reconstructed_x_data).mean()

            ##########################################
            # (4) Update F Network: maximize Dx(F(z))
            ##########################################

            fake_x_data = gf(real_z_data)
            # compute GAN loss for generator F
            loss_gf_gan = -dx(fake_x_data).mean()
            
            # compute the total loss for the generators
            loss_generators = loss_gg_gan + loss_gf_gan + lambda_fc * loss_forward_cycle
            loss_generators.backward()

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
        print(f"Epoch {epoch+1:3d} | GG Loss: {loss_gg_epoch:.6f} | GF Loss: {loss_gf_epoch:.6f} | FC Loss: {loss_fc_epoch:.6f} | DX Loss: {loss_dx_epoch:.6f} | DZ Loss: {loss_dz_epoch:.6f}")

    return gg, gf, dx, dz


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    sw: int = 100,
    ss: int = 1,
    latent_size: int = 20,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
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

    gg, gf, dx, dz = train(
        train_dl,
        signal_size=sw,
        latent_size=latent_size,
        n_features=n_features,
        epochs=epochs,
        device=device
    )
    gg.eval(); gf.eval(); dx.eval(); dz.eval()

    # reconstruct a testing sample at random
    idx = np.random.randint(0, len(test_ds.X))
    sample = test_ds.X[idx].to(device)
    sample_rec = gf(gg(sample)).detach()
    sample_rec_error = point_wise_error(sample_rec, sample)

    plt.figure(figsize=(10, 5))
    plt.plot(sample, color="blue", label="Original")
    plt.plot(sample_rec, color="orange", label="Reconstruction")
    plt.plot(sample_rec_error, color="red", label="Error")
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
        epochs=10,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
