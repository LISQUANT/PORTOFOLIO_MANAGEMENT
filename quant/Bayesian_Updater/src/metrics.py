"""
All derived metrics computed from the posterior.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from .config import ThesisParameters, ExitConfig

# ── Units fix ────────────────────────────────────────────────────────────────

def scale_sigma_L(realised_vol_ann: float, t_elapsed_years:  float,) -> float:

    t = max(t_elapsed_years, 1 / 252)   # at least 1 trading day
    return float(realised_vol_ann * np.sqrt(t))


# ── Realised vol ──────────────────────────────────────────────────────────────

def compute_realised_vol(price_series: pd.Series, window: int = 30,) -> float:
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 5:
        return float("nan")
    rv = float(log_ret.rolling(window, min_periods=5).std().iloc[-1] * np.sqrt(252))
    return rv if np.isfinite(rv) else float("nan")


# ── Probability to target ─────────────────────────────────────────────────────

def probability_reach_target(current_price: float, target_price: float, posterior_mu: float, posterior_sigma: float, years_remaining: float,) -> float:
    if years_remaining <= 0:
        return 1.0 if current_price >= target_price else 0.0

    if posterior_sigma <= 0:
        exp = posterior_mu * years_remaining
        return 1.0 if exp >= np.log(target_price / current_price) else 0.0

    d = (
        np.log(current_price / target_price) + posterior_mu * years_remaining
    ) / (posterior_sigma * np.sqrt(years_remaining))

    return float(stats.norm.cdf(d))


# ── Risk-adjusted hurdle ──────────────────────────────────────────────────────

def compute_hurdle(params: ThesisParameters, config: ExitConfig, realised_vol: Optional[float] = None,) -> float:
    if config.sharpe_equiv is not None:
        vol = (
            realised_vol
            if (realised_vol is not None and np.isfinite(realised_vol))
            else params.implied_vol
        )
        return float(params.risk_free_rate + config.sharpe_equiv * vol)

    return float(params.risk_free_rate + config.hurdle_premium)
