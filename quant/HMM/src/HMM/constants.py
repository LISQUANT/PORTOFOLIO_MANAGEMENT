REGIME_COLORS = {
    "Trending / Momentum":          "#2ecc71",
    "Mean-Reverting":               "#3498db",
    "High Volatility / Distressed": "#e74c3c",
}

REGIME_LABELS = {
    "trending":       "Trending / Momentum",
    "mean_reverting": "Mean-Reverting",
    "distressed":     "High Volatility / Distressed",
}

POSITION_MAP = {
    "Trending / Momentum":          2.0,
    "Mean-Reverting":               1.0,
    "High Volatility / Distressed": -0.3,
}

DEFAULT_ROLLING_WINDOW  = 60
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_START_DATE      = "2020-01-01"
DEFAULT_N_STATES        = 3
DEFAULT_N_ITER          = 1000