"""
All derived metrics computed from the posterior.

Units convention (see prior.py): the model parameter θ is the ANNUALISED
log drift. The posterior (mu, sigma) returned by the engine is therefore a
distribution over annual drift, not over total holding-period return.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from .config import ThesisParameters, ExitConfig


# ── Observation noise ─────────────────────────────────────────────────────────

def drift_observation_sigma(realised_vol_ann: float, t_elapsed_years: float) -> float:
    """
    Std of the observed drift  x/t  as an estimate of the true drift θ.

    The cumulative log-return satisfies  x_t ~ N(θ·t, σ²·t), so the drift
    estimate  x_t/t ~ N(θ, σ²/t)  →  sigma_L = σ/√t.

    Early in the holding period t is small → sigma_L is LARGE → a few days
    of price action carry almost no information about the annual drift and
    the prior dominates. As t grows the observation sharpens.
    """
    t = max(t_elapsed_years, 1 / 252)          # at least 1 trading day
    return float(max(realised_vol_ann / np.sqrt(t), 1e-4))


# ── Realised vol ──────────────────────────────────────────────────────────────

def compute_realised_vol(price_series: pd.Series, window: int = 30) -> float:
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 5:
        return float("nan")
    rv = float(log_ret.rolling(window, min_periods=5).std().iloc[-1] * np.sqrt(252))
    return rv if np.isfinite(rv) else float("nan")


# ── Probability to target ─────────────────────────────────────────────────────

def probability_reach_target(
    current_price:   float,
    target_price:    float,
    posterior_mu:    float,   # annualised drift (posterior mean)
    posterior_sigma: float,   # annualised drift (posterior std)
    years_remaining: float,
    path_vol:        float,   # annualised vol of the price path itself
) -> float:
    """
    P(S_T ≥ target) under the posterior predictive distribution.

    Remaining log-return over T_rem years:
        R = θ·T_rem + σ_path·√T_rem·Z
    with θ ~ N(μ_post, σ_post²) independent of Z, so
        R ~ N(μ_post·T_rem,  σ_post²·T_rem² + σ_path²·T_rem)

    Drift uncertainty compounds linearly with time (·T_rem inside the mean,
    hence ·T_rem² in the variance); path noise compounds with √time.
    """
    if years_remaining <= 0:
        return 1.0 if current_price >= target_price else 0.0

    required = float(np.log(target_price / current_price))
    mean_r   = posterior_mu * years_remaining
    var_r    = (posterior_sigma ** 2) * years_remaining ** 2 \
             + (path_vol ** 2) * years_remaining

    if var_r <= 0:
        return 1.0 if mean_r >= required else 0.0

    d = (mean_r - required) / np.sqrt(var_r)
    return float(stats.norm.cdf(d))


# ── Risk-adjusted hurdle ──────────────────────────────────────────────────────

def compute_hurdle(
    params: ThesisParameters,
    config: ExitConfig,
    realised_vol: Optional[float] = None,
) -> float:
    if config.sharpe_equiv is not None:
        vol = (
            realised_vol
            if (realised_vol is not None and np.isfinite(realised_vol))
            else params.implied_vol
        )
        return float(params.risk_free_rate + config.sharpe_equiv * vol)

    return float(params.risk_free_rate + config.hurdle_premium)
