import numpy as np
from sklearn.preprocessing import StandardScaler
from src.get_data.data_handler import DataHandler


class FeatureMatrix:
    def __init__(self, data_handler: DataHandler):
        self.tickers = data_handler.tickers
        self.sclalers = {} #joao: fit sclarers per ticker 
        self.features = self._build_feature_matrix(data_handler)

    def _build_feature_matrix(self, data_handler: DataHandler) -> dict[str, np.ndarray]:
        """
        For each ticker, stack its 3 features into a 2D array of shape
        (n_timesteps, n_features), drop NaNs, and standardize.
        """
        feature_matrix = {} #joao: for current use case dict is fine but if we scale it becomes a problem since it is not memory optimized, a better option would be a standard list or a nparray

        for ticker in self.tickers:
            raw = self._stack_features(data_handler, ticker)
            clean = self._drop_nans(raw)
            scaled, scaler = self._standardize(clean) #joao: return scaler to to be utlized in regime.py
            feature_matrix[ticker] = scaled
            self.scalers["ticker"] #joao: store the scalers for regime.py  

        return feature_matrix

    def _stack_features(self, data_handler: DataHandler, ticker: str) -> np.ndarray:
        """Combine log returns, rolling volatility and z-score into a single 2D array."""
        log_returns       = data_handler.log_returns[ticker].values
        rolling_volatility = data_handler.rolling_volatility[ticker].values
        zscore            = data_handler.zscore[ticker].values

        return np.column_stack([log_returns, rolling_volatility, zscore])

    def _drop_nans(self, matrix: np.ndarray) -> np.ndarray:
        """Remove any row that contains at least one NaN (introduced by rolling windows)."""
        mask = ~np.isnan(matrix).any(axis=1)
        return matrix[mask]

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        """Zero mean, unit variance per feature column — HMMs are sensitive to scale."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)
        return scaler, scaled #joao: scaler was already fit, also return the new scaled 
