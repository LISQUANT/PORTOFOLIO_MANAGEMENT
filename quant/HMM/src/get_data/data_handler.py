from src.HMM.constants import DEFAULT_ROLLING_WINDOW, DEFAULT_START_DATE
import yfinance as yf
import numpy as np
import pandas as pd
import time

class DataHandler:
    def __init__(self, tickers: dict, start=DEFAULT_START_DATE, end=time.strftime("%Y-%m-%d"), rolling_window: int = DEFAULT_ROLLING_WINDOW):
        self.tickers = list(tickers.keys())
        self.rolling_window = rolling_window

        self.df = self._get_yfinance(start, end)
        self.log_returns = self._compute_log_returns()

        self.rolling_volatility = self._compute_rolling_volatility()
        self.zscore = self._compute_zscore()

    def _get_yfinance(self, start, end) -> pd.DataFrame:
        df = yf.download(self.tickers, start=start, end=end, auto_adjust=True)["Close"]
        return df

    def _compute_log_returns(self) -> pd.DataFrame:
        """Log return: log(Close_t / Close_t-1)"""
        return np.log(self.df / self.df.shift(1))

    def _compute_rolling_volatility(self) -> pd.DataFrame:
        """Rolling std of log returns over the window — proxy for local risk."""
        return self.log_returns.rolling(window=self.rolling_window).std()

    def _compute_zscore(self) -> pd.DataFrame:
        """Z-score of price vs its rolling mean — proxy for mean-reversion pressure."""
        rolling_mean = self.df.rolling(window=self.rolling_window).mean()
        rolling_std = self.df.rolling(window=self.rolling_window).std()
        return (self.df - rolling_mean) / rolling_std