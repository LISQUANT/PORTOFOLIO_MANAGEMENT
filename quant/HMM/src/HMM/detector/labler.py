import numpy as np
from .features import FeatureMatrix
from .regime_detector import RegimeDetector
from src.HMM.constants import REGIME_LABELS

FEATURE_LOG_RETURN  = 0
FEATURE_VOLATILITY  = 1
FEATURE_ZSCORE      = 2

class RegimeLabeler:
    def __init__(self, detector: RegimeDetector, feature_matrix: FeatureMatrix):
        self.tickers = detector.tickers
        self.labels  = self._label_all(detector, feature_matrix)

    def _label_all(self, detector: RegimeDetector, feature_matrix: FeatureMatrix) -> dict:
        """Build a {ticker: {state_int: label_str}} mapping for every ticker."""
        return {
            ticker: self._label_states(
                detector.models[ticker].means_,        
                detector.decode(ticker, feature_matrix.features[ticker])
            )
            for ticker in self.tickers
        }

    def _label_states(self, means: np.ndarray, states: np.ndarray) -> dict[int, str]:
        """
        Assign a human-readable label to each HMM state (0, 1, 2) based on
        the state's emission means across the 3 features.

        Rules (applied in priority order):
          1. Highest rolling volatility mean  → Distressed
          2. Highest log return mean          → Trending
          3. Remaining state                  → Mean-Reverting
        """
        n_states = means.shape[0]
        state_ids = list(range(n_states))

        # highest volatility gets the Distressed label
        distressed = int(np.argmax(means[:, FEATURE_VOLATILITY]))

        # among the remaining states, highest return = Trending
        remaining = [s for s in state_ids if s != distressed]
        trending  = max(remaining, key=lambda s: means[s, FEATURE_LOG_RETURN])

        # whatever is left is Mean-Reverting
        mean_reverting = [s for s in remaining if s != trending][0]

        return {
            distressed:    REGIME_LABELS["distressed"],
            trending:      REGIME_LABELS["trending"],
            mean_reverting: REGIME_LABELS["mean_reverting"],
        }

    def get_label(self, ticker: str, state: int) -> str:
        """Translate a raw HMM state integer into its human-readable regime name."""
        return self.labels[ticker][state]

    def get_current_regime(self, ticker: str, detector: RegimeDetector, feature_matrix: FeatureMatrix) -> str:
        """Return the labeled regime for the most recent timestep."""
        current_state = detector.current_regime(ticker, feature_matrix.features[ticker])
        return self.get_label(ticker, current_state)