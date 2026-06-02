import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.models.utils import point_wise_error, detect_point_anomalies, agg_reconstruction_errors
from src.models.utils import evaluate_point_anomalies
from src.models.signals import SignalsReconstructDataset


class TSAD_LSTM_AE(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 80, 
        num_layers: int = 2, 
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        ) 

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, input_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_out, (hn, cn) = self.encoder(x)
        decoder_out, (_, _) = self.decoder(encoder_out, (hn, cn))
        out = self.fc(decoder_out)
        out = self.activation(out)

        return out


def train(
    model: TSAD_LSTM_AE,
    train_dl: DataLoader,
    val_ds: SignalsReconstructDataset,
    optimizer,
    criterion,
    epochs: int = 50,
    patience: int = 5,
    device: str = "cpu"
) -> TSAD_LSTM_AE:
    patience_counter = 0
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for xb in train_dl:
            xb = xb.to(device)
            
            optimizer.zero_grad()
            xb_ = model(xb)
            loss = criterion(xb_, xb)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_loss /= len(train_dl)

        # validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_ds.X.to(device))
            val_loss = criterion(val_pred, val_ds.X.to(device)).item()

        # metrics
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    return model


def test(
    model: TSAD_LSTM_AE,
    train_ds: SignalsReconstructDataset,
    test_ds: SignalsReconstructDataset, 
    sw: int = 100,
    ss: int = 1,
    device: str = "cpu",   
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        y_train_preds = model(train_ds.X.to(device)).cpu()
        y_train_sw_errors = point_wise_error(y_train_preds, train_ds.X)
        y_train_errors = agg_reconstruction_errors(y_train_sw_errors, sw, ss) 

        y_test_preds = model(test_ds.X.to(device)).cpu()
        y_test_sw_errors = point_wise_error(y_test_preds, test_ds.X)
        y_test_errors = agg_reconstruction_errors(y_test_sw_errors, sw, ss)

    y_test_labels = detect_point_anomalies(y_train_errors, y_test_errors)
    return y_test_labels


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.40,
    val_ratio: float = 0.10,
    sw: int = 250,
    ss: int = 1,
    hidden_size: int = 80,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu"
) -> None:
    """
    Full anomaly detection pipeline for a multivariate time series.

    Parameters
    ----------
    X : np.ndarray of shape (T, n_attributes)
        The full time series.
    y: np.ndarray of shape (T, )
        The anomaly labels of the time-series.
    train_ratio : float
        Fraction of data used for training (default 40% as in paper).
    val_ratio : float
        Fraction of data used for validation / early stopping (default 10%).

    Returns
    -------
    anomaly_matrix : np.ndarray of bool, shape (T, n_attributes)
        True where a timestep/channel pair is anomalous.
    """
    T, n_features = X.shape
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    X_train, y_train    = X[:train_end], y[:train_end]
    X_val, y_val        = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test      = X[val_end:], y[val_end:]      

    print(f"Series shape : {X.shape}")
    print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
    print(f"Validation   : timesteps {train_end} -> {val_end}  ({val_end - train_end} steps)")
    print(f"Test         : timesteps {val_end} -> {T}  ({T - val_end} steps)\n")

    # build sliding windows for train, val, test
    train_ds = SignalsReconstructDataset(X_train, sw=sw, ss=1)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds   = SignalsReconstructDataset(X_val,  sw=sw, ss=1)
    test_ds  = SignalsReconstructDataset(X_test, sw=sw, ss=1)

    # build and train the LSTM for this channel
    model = TSAD_LSTM_AE(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model = train(
        model, 
        train_dl, 
        val_ds, 
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs, 
        device=device
    )

    # compute the test labels based on the errors obtained in training
    y_test_labels = test(model, train_ds, test_ds, sw, ss, device)

    precision, recall, f1 = evaluate_point_anomalies(y_true=y_test, y_predict=y_test_labels)
    print("Metrics")
    print("-------")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 = {f1:.4f}")


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
        train_ratio=0.70,
        val_ratio=0.10,
        sw=100,
        ss=1,
        epochs=20,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
