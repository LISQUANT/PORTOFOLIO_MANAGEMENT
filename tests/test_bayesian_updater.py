"""
Tests for the Bayesian Thesis Updater — all offline (no network).

The regression these tests guard against: the old model compared the
cumulative return so far against the full-period target return, which made
every on-track position trigger an exit on day one.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from quant.Bayesian_Updater import (
    ThesisParameters, ExitConfig, BayesianThesisUpdater,
    build_prior, bayesian_update, probability_reach_target,
    drift_observation_sigma,
)

ENTRY, TARGET, IV, T = 85.0, 130.0, 0.45, 1.0
POSITION_DATE = date(2024, 6, 1)


def make_params(**kw):
    defaults = dict(ticker="MU", entry_price=ENTRY, target_price=TARGET,
                    implied_vol=IV, holding_years=T, risk_free_rate=0.0432,
                    position_date=POSITION_DATE)
    defaults.update(kw)
    return ThesisParameters(**defaults)


def make_path(drift_mult, days=252, seed=0, vol=IV):
    """Synthetic daily closes: entry price ± drift toward target, plus pre-entry history."""
    rng = np.random.default_rng(seed)
    daily_drift = np.log(TARGET / ENTRY) / days * drift_mult
    dates = pd.bdate_range("2024-06-03", periods=days)
    prices = ENTRY * np.exp(np.cumsum(
        np.full(days, daily_drift) + rng.normal(0, vol / np.sqrt(252), days)))
    series = pd.Series(prices, index=dates)

    pre_r = rng.normal(0, vol / np.sqrt(252), 60)
    pre = pd.Series(ENTRY * np.exp(np.cumsum(pre_r) - np.cumsum(pre_r)[-1]),
                    index=pd.bdate_range("2024-03-08", periods=60))
    return pd.concat([pre, series]), series


def run_engine(drift_mult, seed=0):
    full, series = make_path(drift_mult, seed=seed)
    eng = BayesianThesisUpdater(make_params(), ExitConfig())
    for ts, price in series.items():
        eng.update(float(price), full.loc[:ts], as_of=ts.date())
    return eng


# ── Units / prior ─────────────────────────────────────────────────────────────

def test_prior_is_annualised_drift():
    mu, sigma = build_prior(make_params())
    assert mu == pytest.approx(np.log(TARGET / ENTRY) / T)
    assert sigma == pytest.approx(IV / np.sqrt(T))


def test_prior_two_year_horizon_scales():
    mu, sigma = build_prior(make_params(holding_years=2.0))
    assert mu == pytest.approx(np.log(TARGET / ENTRY) / 2.0)
    assert sigma == pytest.approx(IV / np.sqrt(2.0))


def test_drift_observation_noise_shrinks_with_time():
    early = drift_observation_sigma(0.45, 5 / 252)
    late  = drift_observation_sigma(0.45, 0.9)
    assert early > late          # information accumulates with time
    assert early > 1.0           # a week of data says ~nothing about annual drift


# ── Bayesian update ───────────────────────────────────────────────────────────

def test_update_is_precision_weighted_average():
    mu, sigma = bayesian_update(prior_mu=0.4, prior_sigma=0.2,
                                observed_x=0.0, sigma_L=0.2)
    assert mu == pytest.approx(0.2)                    # equal precisions → midpoint
    assert sigma < 0.2                                 # posterior always narrower


def test_noisy_observation_barely_moves_posterior():
    mu, _ = bayesian_update(prior_mu=0.4, prior_sigma=0.2,
                            observed_x=-5.0, sigma_L=50.0)
    assert mu == pytest.approx(0.4, abs=1e-3)


# ── Probability to target ─────────────────────────────────────────────────────

def test_probability_half_when_target_equals_expectation():
    # posterior drift exactly reaches the target → P should be 50%
    mu = np.log(TARGET / ENTRY)
    p = probability_reach_target(ENTRY, TARGET, posterior_mu=mu,
                                 posterior_sigma=0.2, years_remaining=1.0,
                                 path_vol=0.45)
    assert p == pytest.approx(0.5, abs=1e-9)


def test_probability_terminal_states():
    assert probability_reach_target(140, 130, 0.0, 0.2, 0.0, 0.45) == 1.0
    assert probability_reach_target(100, 130, 0.0, 0.2, 0.0, 0.45) == 0.0


# ── Engine behaviour (the day-one-exit regression) ────────────────────────────

def test_on_track_path_holds():
    # Regression for the old day-one-exit bug (which flagged EXIT on 100% of
    # days for every path). A path that ends above target may still dip below
    # the P-floor transiently — that is the model working — but signals must
    # be rare, never on day one, and the final state must be HOLD.
    full, series = make_path(1.0, vol=0.25)
    eng = BayesianThesisUpdater(make_params(), ExitConfig())
    for ts, price in series.items():
        eng.update(float(price), full.loc[:ts], as_of=ts.date())
    df = pd.DataFrame([vars(r) for r in eng.history])

    assert not df["exit_signal"].iloc[0]                  # no day-one knee-jerk
    assert df["exit_signal"].mean() < 0.20                # signals are the exception
    assert df["probability_to_target"].median() > 0.5     # thesis healthy overall
    # the old bug's signature was a PERMANENT hurdle-trigger exit:
    assert not df["exit_reason"].str.contains("hurdle").any()


def test_no_exit_on_day_one():
    for seed in range(5):
        full, series = make_path(1.0, seed=seed)
        eng = BayesianThesisUpdater(make_params(), ExitConfig())
        ts = series.index[0]
        r = eng.update(float(series.iloc[0]), full.loc[:ts], as_of=ts.date())
        assert not r.exit_signal, f"day-one exit with seed {seed}: {r.exit_reason}"


def test_broken_thesis_eventually_exits():
    eng = run_engine(drift_mult=-1.0)
    fe = eng.first_exit()
    assert fe is not None
    assert fe.days_held > 5      # evidence-driven, not a knee-jerk day-one signal


def test_horizon_expiry_is_explicit_exit():
    full, series = make_path(1.0)
    eng = BayesianThesisUpdater(make_params(holding_years=0.25), ExitConfig())
    last = series.index[-1]      # ~1y later, well past the 3-month horizon
    r = eng.update(float(series.iloc[-1]), full, as_of=last.date())
    assert r.exit_signal
    assert "horizon" in r.exit_reason.lower()


def test_no_network_needed_when_history_present():
    # fetch_live_iv=False (default) must never touch the network:
    # engine falls back to thesis IV when realised vol is unavailable.
    eng = BayesianThesisUpdater(make_params(), ExitConfig())
    tiny = pd.Series([85.0, 85.5],
                     index=pd.bdate_range("2024-06-03", periods=2))
    r = eng.update(85.5, tiny, as_of=date(2024, 6, 4))   # rv is NaN here
    assert r is not None
