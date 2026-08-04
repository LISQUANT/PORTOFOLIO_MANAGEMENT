"""Tests for risk/setup.py — offline, synthetic returns."""

import numpy as np
import pandas as pd
import pytest

from risk.setup import var_breach_metrics


def make_returns(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "AAA": rng.normal(0.001, 0.02, n),
        "BBB": rng.normal(0.0005, 0.01, n),
    }, index=idx)


WEIGHTS = {"AAA": 0.5, "BBB": 0.5}


def test_quiet_day_no_breach():
    rets = make_returns()
    rets.iloc[-1] = 0.0                       # flat day
    df, port = var_breach_metrics(rets, WEIGHTS)
    assert (df["Breach"] == "OK").all()
    assert port["Breach"] == "OK"


def test_critical_breach_flagged():
    rets = make_returns()
    rets.iloc[-1, 0] = -0.15                  # crash in AAA today
    df, port = var_breach_metrics(rets, WEIGHTS)
    assert "CRITICAL" in df.set_index("Ticker").loc["AAA", "Breach"]
    assert "CRITICAL" in port["Breach"]       # 50% weight drags the portfolio too


def test_var_window_excludes_today():
    # If today's crash were included in its own VaR window, a huge single-day
    # move could mask itself. VaR must be computed on data excluding today.
    rets = make_returns()
    rets.iloc[-1, 0] = -0.50
    df, _ = var_breach_metrics(rets, WEIGHTS)
    row = df.set_index("Ticker").loc["AAA"]
    assert float(row["VaR 99%"].rstrip("%")) / 100 > -0.10   # VaR unaffected by today


def test_insufficient_data_raises():
    rets = make_returns(n=10)
    with pytest.raises(ValueError):
        var_breach_metrics(rets, WEIGHTS)
