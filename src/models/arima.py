import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from scipy.fft import fft

from models.utils import detect_point_anomalies

def fft_period(signal: np.array) -> int:
    """
    Estimates the dominant period of a univariate time series using the Fast Fourier Transform (FFT).

    The function first removes any linear trend from the signal to avoid low-frequency
    bias in the frequency spectrum, then applies FFT to identify the frequency with the
    highest magnitude. The dominant period is derived as the reciprocal of that frequency.

    Parameters
    ----------
    signal: np.array
        A 1D array representing the univariate time series. The values should be
        evenly spaced in time (e.g., daily, hourly readings).

    Returns
    -------
    period: int
        The dominant period of the given signal.
    """
    # linear detrending
    slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
    trend = np.arange(len(signal)) * slope + intercept 
    detrended = signal - trend 
    
    fft_values = fft(detrended)
    frequencies = np.fft.fftfreq(len(fft_values))

    # remove negative frequencies and sort
    positive_frequencies = frequencies[frequencies > 0]
    magnitudes = np.abs(fft_values)[frequencies > 0]

    # identify dominant frequency
    dominant_frequency = positive_frequencies[np.argmax(magnitudes)]
    # convert frequency to period (e.g., days, weeks, months, etc.)
    dominant_period = 1 / dominant_frequency
    dominant_period = np.floor(dominant_period)
    dominant_period = int(dominant_period)

    return dominant_period


def arima(ts_size: int = 100):
    """
    Fits an ARIMA model for a randomly generated univariate time-series.
    """
    # compute the split point between train and forecast
    train_size = int(ts_size * 0.8)
    forecast_size = ts_size - train_size

    # generate a random univariate time series
    time = np.arange(ts_size)
    trend = time * 0.2
    seasonality = 2 * np.sin(2 * np.pi * time / 12) 
    autoregressive = arma_generate_sample(ar=np.array([1.0, -0.5, 0.7]), ma=np.array([1]), nsample=ts_size, scale=1, burnin=1000)
    ts = trend + seasonality + autoregressive

    # find the periodicity of data automatically
    m = fft_period(ts)

    # search for the best ARIMA model
    model = pm.auto_arima(
        ts[:train_size],
        start_p=0, start_q=0,
        max_p=2, max_q=2,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None,
        m=m,
        test='adf',
        trace=True,         
        stepwise=True
    )
    print(model.summary())

    forecast = model.predict(n_periods=forecast_size)
    forecast_ci = model.predict(n_periods=forecast_size, return_conf_int=True)[1]

    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Original", marker='o', markersize=3)
    plt.plot(range(train_size, len(ts)), forecast, label='Forecast', marker='o', markersize=3, color='orange')
    plt.fill_between(range(train_size, len(ts)), forecast_ci[:, 0], forecast_ci[:, 1], color='tomato', alpha=0.2)
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def varima(ts_size: int = 100, ts_attrs: int = 3):
    """
    Fits a VARIMA(1, 1, 1) for a randomly generated multivariate time-series.
    """
    # compute the split point between train and forecast
    train_size = int(ts_size * 0.8)
    forecast_size = ts_size - train_size

    # generate a random multivariate time series
    data = np.random.randn(ts_size, ts_attrs)
    date_index = pd.date_range(start="2020-01-01", periods=ts_size, freq="D")
    ts = pd.DataFrame(data, index=date_index)
    ts = ts.diff().dropna()

    # fit the VARMAX(1, 1) model
    model = VARMAX(ts[:train_size], order=(1, 1))
    result = model.fit(maxiter=100, disp=False)

    forecast = result.forecast(steps=forecast_size)

    # plot original ts + forecast per attribute
    _, axes = plt.subplots(ts_attrs, 1, figsize=(12, 4 * (ts_attrs + 1)), sharex=False)

    if ts_attrs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        col = ts.columns[i]

        ax.plot(ts.index, ts[col], label="Original", color="steelblue", linewidth=1.5)
        ax.plot(forecast.index, forecast[col], label="Forecast", color="tomato", linewidth=1.5, linestyle="--")

        # shaded region to separate history from forecast
        ax.axvspan(forecast.index[0], forecast.index[-1], alpha=0.05, color="tomato")
        ax.axvline(forecast.index[0], color="gray", linestyle=":", linewidth=1)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("VARIMA(1,1,1) — Original vs Forecast", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def run_pipeline(
    X: np.ndarray,
    train_ratio: float = 0.70,
):
    T, n_attrs = X.shape
    train_end = int(T * train_ratio)

    X_train = X[:train_end]
    X_test  = X[train_end:]

    print(f"Series shape : {X.shape}")
    print(f"Train        : timesteps 0 -> {train_end}  ({train_end} steps)")
    print(f"Test         : timesteps {train_end} -> {T}  ({T - train_end} steps)\n")

    all_train_errors = []
    all_test_errors  = []

    for attr in range(n_attrs):
        print(f"Attribute {attr + 1}/{n_attrs}")

        train_signal = X_train[:, attr]
        test_signal  = X_test[:, attr]

        # find dominant period and fit on train
        m = fft_period(train_signal)
        print(f"Periodicity = {m}")

        print("[START] ARIMA Model Fitting...")
        model = ARIMA(train_signal, order=(1, 1, 1)).fit()
        print("[END] ARIMA Model Fitting!")

        # train errors: in-sample fitted values vs actuals
        train_errors = np.abs(train_signal - model.fittedvalues)

        # test errors: walk-forward — predict 1 step, then update with true value
        # this mirrors exactly what SignalsForecastDataset does with sliding windows
        test_preds  = []
        test_errors = []

        # walk-forward: grow history by one true value at each step
        history     = list(train_signal)
        test_errors = []

        print("[START] Prediction and update...")
        for t in range(len(test_signal)):
            pred = ARIMA(history, order=(1, 1, 1)).fit().forecast(steps=1)[0]
            err  = abs(test_signal[t] - pred)

            test_errors.append(err)
            history.append(test_signal[t])
        print("[END] Prediction and update!\n")

        all_train_errors.append(torch.tensor(train_errors, dtype=torch.float32))
        all_test_errors.append(torch.tensor(test_errors,   dtype=torch.float32))

    # aggregate errors across attributes (mean per timestep)
    train_errors_agg = torch.stack(all_train_errors, dim=1).mean(dim=1)
    test_errors_agg  = torch.stack(all_test_errors,  dim=1).mean(dim=1)

    y_test_labels      = detect_point_anomalies(train_errors_agg, test_errors_agg)
    y_test_anomaly_idx = torch.where(y_test_labels == 1)[0]
    X_test_plot        = torch.tensor(X_test[:, 0], dtype=torch.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(test_errors_agg,  color="orange", label="Reconstruction Error")
    plt.plot(X_test_plot,      color="blue",   label="Original")
    plt.scatter(y_test_anomaly_idx, X_test_plot[y_test_anomaly_idx], color="red", s=10, label="Anomaly")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(999)

    T, n_attrs = 2000, 1
    ts = np.random.randn(T, n_attrs).cumsum(axis=0) * 0.1
    # inject a spike anomaly
    ts[1790:1810] += 15.0

    run_pipeline(ts, train_ratio=0.70)

