"""
Constructs the Gaussian prior over the expected total log-return.

The prior encodes the analyst's thesis:
  - Centre  μ₀ = log(target / entry)   → the return the thesis implies
  - Spread  σ₀ = IV × √T              → uncertainty driven by implied vol
"""

from __future__ import annotations
import numpy as np
from .config import ThesisParameters


def build_prior(params: ThesisParameters) -> tuple[float, float]:
    mu    = float(np.log(params.target_price / params.entry_price))
    sigma = float(params.implied_vol * np.sqrt(params.holding_years))
    return mu, sigma
