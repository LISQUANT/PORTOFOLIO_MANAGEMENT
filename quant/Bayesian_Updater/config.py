"""
All dataclasses for the Bayesian Thesis Updater.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


RISK_FREE_RATE: float = 0.0432   # US Treasury — shared default


@dataclass
class ThesisParameters:
    ticker:          str
    entry_price:     float
    target_price:    float
    implied_vol:     float
    holding_years:   float = 1.0
    risk_free_rate:  float = RISK_FREE_RATE
    position_date:   date  = field(default_factory=date.today)


@dataclass
class ExitConfig:
    """
    Configurable exit thresholds.

    p_floor         : Exit if P(reach target) drops below this  (default 0.30)
    hurdle_premium  : Extra spread over rf for the hurdle rate   (default 0.07)
    sharpe_equiv    : If set: hurdle = rf + sharpe_equiv * vol
                      Overrides hurdle_premium when provided.
    lookback_days   : Rolling window for realised vol             (default 30)
    fetch_live_iv   : If True, fetch live IV (Bloomberg/yfinance) when
                      realised vol is unavailable. Keep False for backtests:
                      live IV on historical dates is look-ahead, and the
                      network calls are slow. (default False)
    """
    p_floor:        float           = 0.30
    hurdle_premium: float           = 0.07
    sharpe_equiv:   Optional[float] = None
    lookback_days:  int             = 30
    fetch_live_iv:  bool            = False


@dataclass
class UpdateResult:
    
    date:                  date
    ticker:                str
    current_price:         float

    # Prior state at this step (before update) — annualised drift units
    prior_mu:              float
    prior_sigma:           float

    # Posterior state (after update) — annualised drift units
    posterior_mu:          float
    posterior_sigma:       float

    # Derived metrics
    expected_return_ann:   float   # annualised expected remaining return
    probability_to_target: float   # P(S_T >= target | posterior)
    risk_adjusted_hurdle:  float   # annualised hurdle rate

    # Exit decision
    exit_signal:           bool
    exit_reason:           str

    # Time accounting
    days_held:             int
    years_remaining:       float
