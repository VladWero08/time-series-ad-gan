import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        signal_size: int = 100,
        input_size: int = 20,
        hidden_size: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
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


class Discriminator(nn.Module):
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
