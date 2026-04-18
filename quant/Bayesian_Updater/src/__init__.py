"""
Package structure

  config.py   — ThesisParameters, ExitConfig, UpdateResult  
  prior.py    — build_prior()                                
  update.py   — bayesian_update()                            
  metrics.py  — realised_vol, scale_sigma_L,                 
                probability_reach_target, compute_hurdle
  data.py     — Bloomberg stub, yfinance fallback,           
                get_implied_vol()
  engine.py   — BayesianThesisUpdater                        
"""

from .config  import ThesisParameters, ExitConfig, UpdateResult
from .prior   import build_prior
from .update  import bayesian_update
from .metrics import (
    compute_realised_vol,
    scale_sigma_L,
    probability_reach_target,
    compute_hurdle,
)
from .data    import get_implied_vol, fetch_iv_bloomberg, fetch_iv_yfinance
from .engine  import BayesianThesisUpdater

__all__ = [
    "ThesisParameters",
    "ExitConfig",
    "UpdateResult",
    "build_prior",
    "bayesian_update",
    "compute_realised_vol",
    "scale_sigma_L",
    "probability_reach_target",
    "compute_hurdle",
    "get_implied_vol",
    "fetch_iv_bloomberg",
    "fetch_iv_yfinance",
    "BayesianThesisUpdater",
]
