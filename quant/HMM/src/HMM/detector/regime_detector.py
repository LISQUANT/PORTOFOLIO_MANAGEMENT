from .features import FeatureMatrix
from hmmlearn.hmm import GaussianHMM
from src.HMM.constants import DEFAULT_N_STATES, DEFAULT_N_ITER
import numpy as np

class RegimeDetector:
    def __init__(self, feature_matrix: FeatureMatrix, n_states: int = DEFAULT_N_STATES, n_iter: int = DEFAULT_N_ITER):
        self.n_states = n_states
        self.n_iter = n_iter
        self.scalers = feature_matrix.scalers # new addition for data pipeline
        self.tickers = feature_matrix.tickers
        self.models = self._fit_all(feature_matrix)

    def _fit_all(self, feature_matrix: FeatureMatrix) -> dict:
        """Fit one HMM per ticker and return a dict of {ticker: fitted model}."""
        return {
            ticker: self._fit_hmm(feature_matrix.features[ticker])
            for ticker in self.tickers
        }

    def _fit_hmm(self, X: np.ndarray) -> GaussianHMM:
        """
        Fit a Gaussian HMM to the observation matrix X.
        covariance_type='full' allows each state to have its own covariance structure,
        meaning the model can capture different volatility shapes per regime.
        """
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42
        )
        #This is the learning phase. The HMM looks at your log returns, volatility, and z-scores to find patterns.
        model.fit(X)
        return model

    def decode(self, ticker: str, X: np.ndarray) -> np.ndarray:
        """
        Run the Viterbi algorithm to find the most likely state sequence.
        Returns an array of state labels (0, 1, 2) for each timestep.
        """
        X_preprocessed = self._preprocess(ticker, X) #joao: other change here, the data is now decoded using processed data and not raw
        return self.models[ticker].predict(X_preprocessed)

    def current_regime(self, ticker: str, X: np.ndarray) -> int:
        """Return the regime label of the most recent timestep."""
        states = self.decode(ticker, X)
        return states[-1]

    def transition_matrix(self, ticker: str) -> np.ndarray:
        """
        The transition probability matrix learned by the HMM.
        Entry [i, j] = probability of moving from state i to state j.
        """
        return self.models[ticker].transmat_