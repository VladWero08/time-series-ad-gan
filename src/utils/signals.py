import torch
import typing as t
import numpy as np

from torch.utils.data import Dataset


class SignalsForecastDataset(Dataset):
    """
    Represents the datasets formed by sliding a window over a time series, using a window size `sw` and a step size `ss`.

    Given a time series `X`, it divides it into smaller subsequences, and the target `y` is the `sw + 1` value that needs to be forecasted. 
    It does not consider the forecasting for the first `sw - 1` points, as they do not possess enough neighbours to form a full window.
    """
    def __init__(
            self, 
            X: torch.tensor, 
            sw: int = 250, 
            ss: int = 1,
            in_features: t.Optional[t.List[int]] = None,
            out_features: t.Optional[t.List[int]] = None,     
        ) -> None:
        """
        Parameters
        ----------
        X: np.ndarray
            Time-series data of shape (T, n_attributes).
        sw: int
            Sliding window size.
        ss: int
            Sliding window step size.
        in_features: list
            List with the indexes of the features that will be given as input to the learning model.
        out_features: list
            List with the indexes of the features that will be given as output of the learning model.
        """
        self.sw = sw
        self.ss = ss
        n_features = X.shape[1]
        self.in_features = in_features if in_features is not None else list(range(n_features))
        self.out_features = out_features if out_features is not None else list(range(n_features))
        self.X, self.y = self.build_sliding_windows(X)

    def build_sliding_windows(self, X: torch.Tensor) -> t.Tuple[torch.Tensor]:
        """
        Transforms a 2D time-series array into overlapping sliding windows for forecasting tasks, in which a window of size `sw` must predict 
        the following value of the time-series (1-value forecast).

        Parameters
        ----------
        X: np.ndarray
            Time-series data of shape (T, n_attributes).

        Returns:
        --------
        X_windowed, y_windowed: t.Tuple[torch.Tensor]
            The dataset reconstructed using sliding windows.
        """
        X_windowed, y_windowed = [], []

        for t in range(0, len(X) - self.sw, self.ss):
            X_windowed.append(X[t:t+self.sw, self.in_features])
            y_windowed.append(X[t+self.sw, self.out_features])

        X_windowed = torch.tensor(np.array(X_windowed), dtype=torch.float32)
        y_windowed = torch.tensor(np.array(y_windowed), dtype=torch.float32)

        return X_windowed, y_windowed
    
    def __getitem__(self, index) -> t.Tuple[torch.Tensor]:
        X = self.X[index]
        y = self.y[index]

        return (X, y)
    
    def __len__(self) -> int:
        return self.X.shape[0]
        

class SignalsReconstructDataset(Dataset):
    """
    Represents the datasets formed by sliding a window over a time series, using a window size `sw` and a step size `ss`.

    Given a time series `X`, it divides it into smaller subsequences, and the target is to be able to reconstruct all subsequences in the dataset. 
    It does not consider the reconstruction for the first `sw - 1` points, as they do not possess enough neighbours to form a full window.
    """    
    def __init__(self, X: torch.tensor, sw: int = 100, ss: int = 1,) -> None:
        """
        Parameters
        ----------
        X: np.ndarray
            Time-series data of shape (T, n_attributes).
        sw: int
            Sliding window size.
        ss: int
            Sliding window step size.
        """
        self.sw = sw
        self.ss = ss
        self.X = self.build_sliding_windows(X)

    def build_sliding_windows(self, X: torch.Tensor) -> t.Tuple[torch.Tensor]:
        """
        Transforms a 2D time-series array into overlapping sliding windows, of window size `sw` and step size `ss`.

        Parameters
        ----------
        X: np.ndarray
            Time-series data of shape (T, n_attributes).

        Returns:
        --------
        X_windowed: torch.Tensor
            The new dataset built using sliding windows.
        """
        X_windowed = [X[t:t+self.sw, :] for t in range(0, len(X) - self.sw + 1, self.ss)]
        X_windowed = torch.tensor(np.array(X_windowed), dtype=torch.float32)

        return X_windowed
    
    def __getitem__(self, index) -> t.Tuple[torch.Tensor]:
        return self.X[index]
    
    def __len__(self) -> int:
        return self.X.shape[0]
