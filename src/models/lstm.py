import numpy as np
import torch
import torch.nn as nn
import typing as t

from torch.utils.data import DataLoader

from src.models.utils import point_wise_error, get_anomaly_intervals, detect_point_anomalies, detect_contextual_anomalies
from src.models.utils import evaluate_point_anomalies, evaluate_collective_anomalies
from src.models.utils import intervals_to_points, plot_performance
from src.models.signals import SignalsForecastDataset


class TSAD_LSTM(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 80, 
        num_layers: int = 2, 
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.activation(out)

        return out


def train(
    model: TSAD_LSTM,
    train_dl: SignalsForecastDataset,
    val_ds: SignalsForecastDataset,
    optimizer,
    criterion,
    epochs: int = 35,
    patience: int = 5,
    device: str = "cpu"
) -> TSAD_LSTM:
    patience_counter = 0
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        train_loss /= len(train_dl)

        # validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_ds.X.to(device))
            val_loss = criterion(val_pred, val_ds.y.to(device)).item()

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
    model: TSAD_LSTM,
    ds: SignalsForecastDataset,
    device: str = "cpu",
) -> t.Tuple[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        ds_preds = model(ds.X.to(device)).cpu()
        ds_errors = point_wise_error(ds_preds, ds.y)

    return ds_preds, ds_errors


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.40,
    val_ratio: float = 0.10,
    sw: int = 250,
    ss: int = 1,
    hidden_size: int = 80,
    num_layers: int = 2,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 5,
    device: str = "cpu",
    anomaly_type: str = "point",
    plot: bool = False,
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
    """
    T, n_features = X.shape
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    X_train, y_train    = X[:train_end], y[:train_end][sw:]
    X_val, y_val        = X[train_end:val_end], y[train_end:val_end][sw:]
    X_test, y_test      = X[val_end:], y[val_end:][sw:]  

    print(f"Series shape : {X.shape}")
    print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
    print(f"Validation   : timesteps {train_end} -> {val_end}  ({val_end - train_end} steps)")
    print(f"Test         : timesteps {val_end} -> {T}  ({T - val_end} steps)\n")

    # build sliding windows for train, val, test
    train_ds = SignalsForecastDataset(X_train, sw, ss)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds   = SignalsForecastDataset(X_val,  sw, ss)
    test_ds  = SignalsForecastDataset(X_test, sw, ss)

    # build and train the LSTM for this channel
    model = TSAD_LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model = train(
        model, 
        train_dl, 
        val_ds, 
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs, 
        patience=patience,
        device=device
    )

    # compute the forecasting errors for the test set
    X_test_preds, y_test_errors = test(model, test_ds, device)

    match anomaly_type:
        case "point":
            # compute the forecasting errors for the train set
            _, y_train_errors = test(model, train_ds, device)
            # compute the test labels based on the forecasting errors obtained in training
            y_test_labels = detect_point_anomalies(y_train_errors, y_test_errors)

            precision, recall, f1 = evaluate_point_anomalies(y_true=y_test, y_predict=y_test_labels)
        case "contextual":
            # compute the test anomaly sequences from test labels 
            y_test_labels_intervals = get_anomaly_intervals(y_test) 
            # compute the test anomaly sequences from test forecast errors
            y_test_errors_intervals = detect_contextual_anomalies(y_test_errors)
            y_test_labels = intervals_to_points(y_test_errors_intervals, y_test.shape[0])

            precision, recall, f1 = evaluate_collective_anomalies(y_test_labels_intervals, y_test_errors_intervals)

    print()
    print("Metrics")
    print("-------")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 = {f1:.4f}")

    if plot:
        # for plotting, first sw test points need to be removed no forecast was made for them
        plot_performance(X=X_test[sw:], X_preds=X_test_preds, y=y_test_labels)


if __name__ == "__main__":
    np.random.seed(999)

    T, n_features = 2000, 1
    ts = np.random.randn(T, n_features).cumsum(axis=0) * 0.1
    y = np.zeros(T)

    # inject spike anomalies
    anomaly_idx = [1790, 1850, 1930]
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        plot=True,
    )
