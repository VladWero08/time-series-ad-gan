import torch
import torch.nn as nn


class GeneratorG(nn.Module):
    def __init__(
        self,
        latent_size: int = 20,
        input_size: int = 100,
        hidden_size: int = 20, 
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=latent_size,   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)

        return out


class GeneratorF(nn.Module):
    def __init__(
        self,
        signal_size: int = 100,
        latent_size: int = 20,
        hidden_size: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(
            in_features=hidden_size * num_layers,
            out_features=signal_size,   
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)

        return out


class DiscriminatorX(nn.Module):
    def __init__(
        self,
        signal_size: int = 100,
        n_attrs: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_attrs, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1),
        )
        self.fc = nn.Linear(in_features=signal_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, signal_size, n_attrs)

        # (batch, n_attrs, signal_size)
        out = x.permute(0, 2, 1)
        # (batch, 1, signal_size)
        out = self.conv(out)
        # (batch, signal_size)
        out = out.squeeze(1)
        # (batch, 1)          
        out = self.fc(out)            

        return out


class DiscriminatorZ(nn.Module):
    def __init__(
        self,
        latent_size: int = 100,
        n_attrs: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_attrs, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=3, padding=1),
        )
        self.fc = nn.Linear(in_features=latent_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, latent_size, n_attrs)

        # (batch, n_attrs, latent_size)
        out = x.permute(0, 2, 1)
        # (batch, 1, latent_size)
        out = self.conv(out)
        # (batch, latent_size)
        out = out.squeeze(1)
        # (batch, 1)          
        out = self.fc(out)            

        return out