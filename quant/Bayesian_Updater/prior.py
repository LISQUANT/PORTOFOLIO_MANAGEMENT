"""
Constructs the Gaussian prior over the ANNUALISED log drift θ.

The prior encodes the analyst's thesis in drift space so that it is
directly comparable with the observed drift (cumulative return / time):
  - Centre  μ₀ = log(target / entry) / T   → the annual drift the thesis implies
  - Spread  σ₀ = IV / √T                   → thesis uncertainty, in drift units

Why these units: the thesis says the TOTAL log-return over T years is
log(target/entry) with uncertainty IV·√T. Dividing both by T converts the
total-return statement into an annualised-drift statement (std divides by T,
so IV·√T / T = IV/√T).
"""

from __future__ import annotations
import numpy as np
from .config import ThesisParameters


def build_prior(params: ThesisParameters) -> tuple[float, float]:
    T = max(params.holding_years, 1 / 252)   # avoid division by zero
    mu    = float(np.log(params.target_price / params.entry_price) / T)
    sigma = float(params.implied_vol / np.sqrt(T))
    return mu, sigma
