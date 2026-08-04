"""Tests for quant/ratios.py — offline, pure functions only."""

import numpy as np
import pandas as pd
import pytest

from quant.ratios import (
    sharpe_ratio, calmar_ratio, generate_signal,
    rv_bands, rv_anomaly_signal, iv_rv_signal,
)


def test_calmar_uses_relative_drawdown_and_cagr():
    # +10% then -20%, then exactly flat for a year:
    # wealth peaks at 1.10, troughs at 0.88 → max relative DD = 20%
    r = pd.Series([0.10, -0.20] + [0.0] * 250)
    wealth_end = 1.10 * 0.80
    n = len(r)
    expected_cagr = wealth_end ** (252 / n) - 1
    assert calmar_ratio(r) == pytest.approx(expected_cagr / 0.20)


def test_calmar_no_drawdown_is_nan():
    r = pd.Series([0.01] * 100)      # monotonic up → no drawdown
    assert np.isnan(calmar_ratio(r))


def test_sharpe_sign():
    up   = pd.Series(np.full(252, 0.003))
    down = pd.Series(np.full(252, -0.003))
    # constant series has zero std → NaN; add tiny noise
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-4, 252)
    assert sharpe_ratio(up + noise) > 0
    assert sharpe_ratio(down + noise) < 0


def test_generate_signal():
    assert generate_signal(2.0, 3.0) == "Buy"
    assert generate_signal(0.1, 3.0) == "Sell"
    assert generate_signal(2.0, 0.1) == "Sell"
    assert generate_signal(float("nan"), 1.0) == "Hold"


def test_rv_bands_and_alerts():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 300))
    b = rv_bands(r)
    assert b["lower"] < b["mean"] < b["upper"]
    assert rv_anomaly_signal(b["upper"] + 0.1, b["upper"], b["lower"]).startswith("Attention")
    assert rv_anomaly_signal(b["mean"], b["upper"], b["lower"]).startswith("Normal")


def test_iv_rv_signal():
    assert "IV >> RV" in iv_rv_signal(0.10)
    assert "RV >> IV" in iv_rv_signal(-0.10)
    assert "normal range" in iv_rv_signal(0.0)
