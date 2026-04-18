from src.get_data.data_handler import DataHandler
from .detector.features import FeatureMatrix
from .detector.labler import RegimeLabeler
from .detector.regime_detector import RegimeDetector
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from src.HMM.constants import REGIME_COLORS, POSITION_MAP, DEFAULT_INITIAL_CAPITAL

class Backtester:
    """
    Simple regime-based strategy:
        Trending / Momentum          →  Long
        Mean-Reverting               →  Flat
        High Volatility / Distressed →  Short
    You can change these values in the constants.py file
    """
  
    def __init__(self,data_handler: DataHandler,feature_matrix: FeatureMatrix,detector: RegimeDetector, labeler: RegimeLabeler):
        
        self.data_handler    = data_handler
        self.feature_matrix  = feature_matrix
        self.detector        = detector
        self.labeler         = labeler
        
        self.initial_capital = DEFAULT_INITIAL_CAPITAL
        self.POSITION_MAP = POSITION_MAP

    def run(self, ticker: str) -> pd.DataFrame:
        """
        Run the backtest for a given ticker.
        Returns a DataFrame with daily positions, returns and portfolio value.
        """
        states   = self.detector.decode(ticker, self.feature_matrix.features[ticker])
        labels   = [self.labeler.get_label(ticker, s) for s in states]
        dates    = self._get_aligned_dates(ticker)
        close    = self._get_aligned_close(ticker)

        log_returns = np.log(close[1:] / close[:-1])

        # Position is determined by the PREVIOUS day's regime (avoid lookahead bias)
        positions = np.array([self.POSITION_MAP[label] for label in labels[:-1]])

        strategy_returns    = positions * log_returns
        cumulative_returns  = np.exp(np.cumsum(strategy_returns))
        portfolio_value     = self.initial_capital * cumulative_returns

        results = pd.DataFrame({
            "Date":              dates[1:],
            "Close":             close[1:],
            "Regime":            labels[1:],
            "Position":          positions,
            "Log Return":        log_returns,
            "Strategy Return":   strategy_returns,
            "Portfolio Value":   portfolio_value,
        }).set_index("Date")

        return results

    def summary(self, ticker: str) -> None:
        results          = self.run(ticker)
        close            = results["Close"]
        buy_hold_return  = (close.iloc[-1] / close.iloc[0] - 1) * 100

        total_return  = (results["Portfolio Value"].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = results["Strategy Return"].mean() * 252 * 100
        annual_vol    = results["Strategy Return"].std() * np.sqrt(252) * 100
        sharpe        = (annual_return / annual_vol) if annual_vol != 0 else 0
        max_drawdown  = self._compute_max_drawdown(results["Portfolio Value"])

        print(f"\n{'='*40}")
        print(f"  Backtest Summary  - {ticker}")
        print(f"{'='*40}")
        print(f"  Initial Capital   : ${self.initial_capital:>12,.2f}")
        print(f"  Final Value       : ${results['Portfolio Value'].iloc[-1]:>12,.2f}")
        print(f"  Total Return      : {total_return:>11.2f}%")
        print(f"  Buy & Hold Return : {buy_hold_return:>11.2f}%")
        print(f"  Annual Return     : {annual_return:>11.2f}%")
        print(f"  Annual Vol        : {annual_vol:>11.2f}%")
        print(f"  Sharpe Ratio      : {sharpe:>11.2f}")
        print(f"  Max Drawdown      : {max_drawdown:>11.2f}%")
        print(f"{'='*40}\n")

    def plot(self, ticker: str) -> None:
        results          = self.run(ticker)
        buy_hold_value   = self.initial_capital * (results["Close"] / results["Close"].iloc[0])

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(f"{ticker} — Regime Strategy vs Buy & Hold", fontsize=14, fontweight="bold")

        # Color background by regime
        for i in range(len(results) - 1):
            color = REGIME_COLORS[results["Regime"].iloc[i]]
            ax.axvspan(results.index[i], results.index[i + 1], alpha=0.15, color=color, linewidth=0)

        ax.plot(results.index, results["Portfolio Value"], color="black",  linewidth=1.2, zorder=5, label="Strategy")
        ax.plot(results.index, buy_hold_value,             color="grey",   linewidth=1.2, zorder=5, label="Buy & Hold", linestyle="--")

        ax.set_ylabel("Portfolio Value (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.set_xlim(results.index[0], results.index[-1])

        legend_patches = [
            mpatches.Patch(color=color, alpha=0.6, label=label)
            for label, color in REGIME_COLORS.items()
        ]
        regime_legend = ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
        ax.add_artist(regime_legend)
        ax.legend(loc="lower right", fontsize=8)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # Helpers

    def _compute_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Max peak-to-trough decline as a percentage."""
        rolling_max = portfolio_value.cummax()
        drawdown    = (portfolio_value - rolling_max) / rolling_max * 100
        return drawdown.min()

    def _get_aligned_dates(self, ticker: str) -> list:
        n = len(self.feature_matrix.features[ticker])
        return self.data_handler.df.index[-n:].tolist()

    def _get_aligned_close(self, ticker: str) -> np.ndarray:
        n = len(self.feature_matrix.features[ticker])
        return self.data_handler.df[ticker].values[-n:]