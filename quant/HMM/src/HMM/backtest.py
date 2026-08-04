from src.get_data.data_handler import DataHandler
from .detector.features import FeatureMatrix
from .detector.labler import RegimeLabeler
from .detector.regime_detector import RegimeDetector
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from src.HMM.constants import (
    REGIME_COLORS, POSITION_MAP, DEFAULT_INITIAL_CAPITAL,
    DEFAULT_TRAIN_SPLIT, DEFAULT_COST_BPS,
)

class Backtester:
    """
    Regime-based strategy — position per regime set in constants.POSITION_MAP.

    Two modes:

    out_of_sample=True (default, the honest one):
        The HMM is fitted ONLY on the first `train_split` fraction of the
        history. The test period is decoded CAUSALLY: the state at day t is
        the last state of a Viterbi pass over data up to and including t —
        no future observations are used. Performance is reported on the
        test period only.

    out_of_sample=False (in-sample, optimistic):
        Uses the full-history fit + full-sequence Viterbi from the detector.
        Both the training and the smoothing look ahead — treat the results
        as descriptive, never as expected live performance.

    Mechanics (both modes):
        - The position applied to the return from day t to t+1 is decided
          from the regime known at the END of day t (one-day lag).
        - Leverage is applied to SIMPLE returns (a 2x position earns twice
          the simple return, not twice the log return).
        - Transaction costs: `cost_bps` basis points on each unit of
          position change (|Δposition|).
    """

    def __init__(self, data_handler: DataHandler, feature_matrix: FeatureMatrix,
                 detector: RegimeDetector, labeler: RegimeLabeler,
                 train_split: float = DEFAULT_TRAIN_SPLIT,
                 cost_bps: float = DEFAULT_COST_BPS):

        self.data_handler    = data_handler
        self.feature_matrix  = feature_matrix
        self.detector        = detector
        self.labeler         = labeler

        self.initial_capital = DEFAULT_INITIAL_CAPITAL
        self.POSITION_MAP    = POSITION_MAP
        self.train_split     = train_split
        self.cost_bps        = cost_bps

    def run(self, ticker: str, out_of_sample: bool = True) -> pd.DataFrame:
        """
        Run the backtest for a given ticker.
        Returns a DataFrame with daily positions, returns and portfolio value.
        """
        X     = self.feature_matrix.features[ticker]
        dates = self._get_aligned_dates(ticker)
        close = self._get_aligned_close(ticker)

        if out_of_sample:
            split = int(len(X) * self.train_split)
            model = self.detector._fit_hmm(X[:split])
            train_states = model.predict(X[:split])
            label_map    = self.labeler._label_states(model.means_, train_states)

            states = self._causal_states(model, X, split)
            labels = [label_map[s] for s in states]

            dates, close = dates[split:], close[split:]
        else:
            states = self.detector.decode(ticker, X)
            labels = [self.labeler.get_label(ticker, s) for s in states]

        simple_returns = close[1:] / close[:-1] - 1

        # Position applied to return t → t+1 comes from day t's regime (one-day lag)
        positions = np.array([self.POSITION_MAP[label] for label in labels[:-1]])

        turnover  = np.abs(np.diff(positions, prepend=0.0))
        costs     = turnover * (self.cost_bps / 10_000)

        strategy_returns   = positions * simple_returns - costs
        portfolio_value    = self.initial_capital * np.cumprod(1 + strategy_returns)

        results = pd.DataFrame({
            "Date":            dates[1:],
            "Close":           close[1:],
            "Regime":          labels[1:],
            "Position":        positions,
            "Return":          simple_returns,
            "Strategy Return": strategy_returns,
            "Portfolio Value": portfolio_value,
        }).set_index("Date")

        return results

    def summary(self, ticker: str, out_of_sample: bool = True) -> None:
        results          = self.run(ticker, out_of_sample)
        close            = results["Close"]
        buy_hold_return  = (close.iloc[-1] / close.iloc[0] - 1) * 100

        n_days        = len(results)
        total_return  = (results["Portfolio Value"].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = ((results["Portfolio Value"].iloc[-1] / self.initial_capital)
                         ** (252 / n_days) - 1) * 100 if n_days > 0 else 0.0
        annual_vol    = results["Strategy Return"].std() * np.sqrt(252) * 100
        sharpe        = (annual_return / annual_vol) if annual_vol != 0 else 0
        max_drawdown  = self._compute_max_drawdown(results["Portfolio Value"])
        mode          = "OUT-OF-SAMPLE" if out_of_sample else "IN-SAMPLE (look-ahead!)"

        print(f"\n{'='*44}")
        print(f"  Backtest Summary  - {ticker}  [{mode}]")
        print(f"{'='*44}")
        print(f"  Initial Capital   : ${self.initial_capital:>12,.2f}")
        print(f"  Final Value       : ${results['Portfolio Value'].iloc[-1]:>12,.2f}")
        print(f"  Total Return      : {total_return:>11.2f}%")
        print(f"  Buy & Hold Return : {buy_hold_return:>11.2f}%")
        print(f"  Annual Return     : {annual_return:>11.2f}%  (CAGR)")
        print(f"  Annual Vol        : {annual_vol:>11.2f}%")
        print(f"  Sharpe Ratio      : {sharpe:>11.2f}")
        print(f"  Max Drawdown      : {max_drawdown:>11.2f}%")
        print(f"  Transaction Costs : {self.cost_bps:.0f} bps per unit turnover")
        print(f"{'='*44}\n")

    def plot(self, ticker: str, out_of_sample: bool = True) -> None:
        results          = self.run(ticker, out_of_sample)
        buy_hold_value   = self.initial_capital * (results["Close"] / results["Close"].iloc[0])

        fig, ax = plt.subplots(figsize=(14, 5))
        mode = "out-of-sample" if out_of_sample else "in-sample"
        fig.suptitle(f"{ticker} — Regime Strategy vs Buy & Hold ({mode})",
                     fontsize=14, fontweight="bold")

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

    def _causal_states(self, model, X: np.ndarray, start: int) -> np.ndarray:
        """
        State at each test day t decoded using ONLY observations up to t
        (Viterbi over the prefix, keep the last state). O(n²) but fine for
        daily data.
        """
        states = np.empty(len(X) - start, dtype=int)
        for i, t in enumerate(range(start, len(X))):
            states[i] = model.predict(X[: t + 1])[-1]
        return states

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
