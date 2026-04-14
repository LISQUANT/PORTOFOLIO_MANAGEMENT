import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
from src.get_data.data_handler import DataHandler
from src.HMM.detector.features import FeatureMatrix
from src.HMM.detector.labler import RegimeLabeler
from src.HMM.detector.regime_detector import RegimeDetector
from src.HMM.constants import REGIME_COLORS

class RegimeVisualizer:
    def __init__(self,data_handler: DataHandler, feature_matrix: FeatureMatrix,detector: RegimeDetector, labeler:RegimeLabeler ):
        self.data_handler = data_handler
        self.feature_matrix = feature_matrix
        self.detector = detector
        self.labeler = labeler

    def plot(self, ticker: str):
        """Renders the close price chart colored by detected regime."""
        states  = self.detector.decode(ticker, self.feature_matrix.features[ticker])
        labels  = [self.labeler.get_label(ticker, s) for s in states]
        dates   = self._get_aligned_dates(ticker)
        close   = self._get_aligned_close(ticker)
        current = self.labeler.get_current_regime(ticker, self.detector, self.feature_matrix)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle(
            f"{ticker} — HMM Regime Detection\nCurrent Regime: {current}",
            fontsize=14, fontweight="bold"
        )

        self._plot_price_with_regimes(ax, dates, close, labels)
        plt.tight_layout()
        plt.show()

    def _plot_price_with_regimes(self, ax, dates, close, labels):
        ax.set_title("Close Price by Regime", fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Color each segment by its regime
        for i in range(len(dates) - 1):
            color = REGIME_COLORS[labels[i]]
            ax.axvspan(dates[i], dates[i + 1], alpha=0.15, color=color, linewidth=0)

        ax.plot(dates, close, color="black", linewidth=1.2, zorder=5)
        ax.set_xlim(dates[0], dates[-1])

        legend_patches = [
            mpatches.Patch(color=color, alpha=0.6, label=label)
            for label, color in REGIME_COLORS.items()
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    def _get_aligned_dates(self, ticker: str) -> list:
        """Trim dates to match the NaN-dropped feature matrix length."""
        n = len(self.feature_matrix.features[ticker])
        return self.data_handler.df.index[-n:].tolist()

    def _get_aligned_close(self, ticker: str) -> np.ndarray:
        """Trim close prices to match the NaN-dropped feature matrix length."""
        n = len(self.feature_matrix.features[ticker])
        return self.data_handler.df[ticker].values[-n:]