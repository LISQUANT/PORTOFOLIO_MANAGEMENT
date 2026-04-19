"""
Model
-----
Posterior:
    σ_n² = 1 / (1/σ₀² + 1/σ_L²)
    μ_n  = σ_n² × (μ₀/σ₀² + x/σ_L²)
The update is a precision-weighted average of two opinions:
  - The prior   contributes with precision  1/σ₀²  
  - The data    contributes with precision  1/σ_L²
"""
from __future__ import annotations
import numpy as np

def bayesian_update(prior_mu: float, prior_sigma: float, observed_x: float, sigma_L: float,) -> tuple[float, float]:

    sigma_L = max(float(sigma_L), 1e-6)   # numerical floor

    tau0 = prior_sigma ** 2
    tauL = sigma_L     ** 2

    sigma_n_sq = 1.0 / (1.0 / tau0 + 1.0 / tauL)
    mu_n       = sigma_n_sq * (prior_mu / tau0 + observed_x / tauL)

    return float(mu_n), float(np.sqrt(sigma_n_sq))
