"""
BayesianThesisUpdater — the main class that orchestrates all modules.

  1. Calls prior.py      to initialise the prior (annualised-drift space)
  2. Calls metrics.py    to compute realised vol and the drift-observation noise
  3. Calls update.py     to run the Bayesian update
  4. Calls metrics.py    again to compute P(target) and hurdle
  5. Evaluates exit conditions (horizon stop → probability floor → hurdle)
  6. Returns an UpdateResult

Model summary
-------------
Parameter: θ = annualised log drift of the position.
Prior:     θ ~ N( log(target/entry)/T,  (IV/√T)² )
Data:      observed drift  x_t/t  with noise  σ/√t   (x_t = cum. log-return)
Each update starts from the ORIGINAL prior because the daily observations
are overlapping cumulative returns — chaining posteriors would double-count
the same price path many times over.

Volatility source: realised vol (rolling) is preferred; if unavailable the
fallback is override_iv → live IV (only if config.fetch_live_iv) → thesis IV.
Live IV is never fetched during backtests unless explicitly enabled, both
for speed and because fetching today's IV for a historical date would be
look-ahead.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .config import ThesisParameters, ExitConfig, UpdateResult
from .prior   import build_prior
from .update  import bayesian_update
from .metrics import (
    compute_realised_vol,
    drift_observation_sigma,
    probability_reach_target,
    compute_hurdle,
)
from .data    import get_implied_vol


class BayesianThesisUpdater:

    def __init__(
        self,
        params: ThesisParameters,
        config: Optional[ExitConfig] = None,
    ):
        self.params = params
        self.config = config or ExitConfig()

        # Initialise prior — posterior starts equal to prior
        self.prior_mu,     self.prior_sigma     = build_prior(params)
        self.posterior_mu, self.posterior_sigma = self.prior_mu, self.prior_sigma

        self.history: list[UpdateResult] = []

    # ── Time helpers ──────────────────────────────────────────────────────────

    def _years_remaining(self, as_of: date) -> float:
        elapsed = (as_of - self.params.position_date).days / 365.25
        return max(0.0, self.params.holding_years - elapsed)

    def _years_elapsed(self, as_of: date) -> float:
        return (as_of - self.params.position_date).days / 365.25

    def _days_held(self, as_of: date) -> int:
        return max(0, (as_of - self.params.position_date).days)

    # ── Single update step ────────────────────────────────────────────────────

    def update(
        self,
        current_price: float,
        price_history: pd.Series,
        as_of:         Optional[date] = None,
        override_iv:   Optional[float] = None,
    ) -> UpdateResult:

        as_of = as_of or date.today()

        # ── 1. Observation: annualised drift since entry ──────────────────────
        t_elapsed = max(self._years_elapsed(as_of), 1 / 252)
        obs_x     = float(np.log(current_price / self.params.entry_price))
        obs_drift = obs_x / t_elapsed

        # ── 2. Volatility: realised → override → live IV (opt-in) → thesis ────
        rv = compute_realised_vol(price_history, self.config.lookback_days)

        if np.isfinite(rv):
            base_vol = rv
        elif override_iv is not None and override_iv > 0:
            base_vol = override_iv
        elif self.config.fetch_live_iv:
            base_vol = get_implied_vol(
                self.params.ticker, current_price, as_of, self.params.implied_vol
            )
        else:
            base_vol = self.params.implied_vol

        sigma_L = drift_observation_sigma(base_vol, t_elapsed)

        # ── 3. Bayesian update (always from the original prior) ───────────────
        prev_mu, prev_sigma = self.posterior_mu, self.posterior_sigma

        new_mu, new_sigma = bayesian_update(
            prior_mu    = self.prior_mu,
            prior_sigma = self.prior_sigma,
            observed_x  = obs_drift,
            sigma_L     = sigma_L,
        )

        self.posterior_mu    = new_mu
        self.posterior_sigma = new_sigma

        # ── 4. Derived metrics ────────────────────────────────────────────────
        years_rem = self._years_remaining(as_of)
        days_held = self._days_held(as_of)

        p_target = probability_reach_target(
            current_price   = current_price,
            target_price    = self.params.target_price,
            posterior_mu    = new_mu,
            posterior_sigma = new_sigma,
            years_remaining = years_rem,
            path_vol        = base_vol,
        )

        # Expected annualised remaining return = posterior drift.
        # (The model's drift is constant, so the annualised expectation for
        # the remaining period equals the posterior mean of θ.)
        exp_ann_return = new_mu

        hurdle = compute_hurdle(self.params, self.config, rv)

        # ── 5. Exit triggers (horizon stop checked first, explicitly) ─────────
        exit_signal = False
        exit_reason = ""

        if years_rem <= 0:
            exit_signal = True
            exit_reason = (
                f"Holding horizon reached ({self.params.holding_years:.2f}y) "
                f"— thesis expired, re-underwrite or close"
            )
        elif p_target < self.config.p_floor:
            exit_signal = True
            exit_reason = (
                f"P(reach target) = {p_target:.1%} "
                f"< floor {self.config.p_floor:.1%}"
            )
        elif exp_ann_return < hurdle:
            exit_signal = True
            exit_reason = (
                f"Expected return = {exp_ann_return:.1%} "
                f"< hurdle {hurdle:.1%}"
            )

        result = UpdateResult(
            date                  = as_of,
            ticker                = self.params.ticker,
            current_price         = current_price,
            prior_mu              = prev_mu,
            prior_sigma           = prev_sigma,
            posterior_mu          = new_mu,
            posterior_sigma       = new_sigma,
            expected_return_ann   = exp_ann_return,
            probability_to_target = p_target,
            risk_adjusted_hurdle  = hurdle,
            exit_signal           = exit_signal,
            exit_reason           = exit_reason,
            days_held             = days_held,
            years_remaining       = years_rem,
        )
        self.history.append(result)
        return result

    # ── Batch backtest ────────────────────────────────────────────────────────

    def run_backtest(self, price_series: pd.Series) -> pd.DataFrame:

        start  = pd.Timestamp(self.params.position_date)
        series = price_series.loc[price_series.index >= start]

        rows = []
        for ts, price in series.items():
            as_of   = ts.date() if hasattr(ts, "date") else ts
            history = price_series.loc[:ts]
            r       = self.update(float(price), history, as_of=as_of)
            rows.append(vars(r))

        return pd.DataFrame(rows).set_index("date")

    # ── Convenience ───────────────────────────────────────────────────────────

    def first_exit(self) -> Optional[UpdateResult]:
        """Return the first UpdateResult where exit_signal is True, or None."""
        for r in self.history:
            if r.exit_signal:
                return r
        return None

    def summary(self) -> str:
        """Human-readable summary of the current posterior state."""
        if not self.history:
            return "No updates recorded yet."
        r = self.history[-1]
        c = self.config
        return (
            f"\n{'═'*60}\n"
            f"  BAYESIAN THESIS UPDATER  |  {self.params.ticker}\n"
            f"{'═'*60}\n"
            f"  Entry Price         :  {self.params.entry_price:>10.2f}\n"
            f"  Target Price        :  {self.params.target_price:>10.2f}\n"
            f"  Current Price       :  {r.current_price:>10.2f}\n"
            f"  Days Held           :  {r.days_held:>10d}\n"
            f"  Years Remaining     :  {r.years_remaining:>10.2f}\n"
            f"{'─'*60}\n"
            f"  Prior   μ / σ  (ann):  {self.prior_mu:+.4f}  /  {self.prior_sigma:.4f}\n"
            f"  Post.   μ / σ  (ann):  {r.posterior_mu:+.4f}  /  {r.posterior_sigma:.4f}\n"
            f"{'─'*60}\n"
            f"  P(reach target)     :  {r.probability_to_target:>8.1%}  "
            f"[floor: {c.p_floor:.0%}]\n"
            f"  Exp. Ann. Return    :  {r.expected_return_ann:>8.1%}  "
            f"[hurdle: {r.risk_adjusted_hurdle:.1%}]\n"
            f"{'═'*60}\n"
            f"  EXIT:  "
            + ("⚠  " + r.exit_reason if r.exit_signal else "✓  No exit triggered")
            + f"\n{'═'*60}"
        )
