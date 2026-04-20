# Bayesian Thesis Updater

## What it does

For each equity position the Portfolio Team has a thesis: a target price and a
holding horizon. This module constructs a prior distribution from that thesis,
updates it daily with new price data via Bayesian inference, and fires exit
signals when the thesis is no longer supported.

**Two exit triggers:**
- `P(reach target | posterior) < 30%` — the probability of hitting the target has collapsed
- `Expected annualised remaining return < rf + hurdle` — the remaining upside no longer justifies the risk

---

## File structure

```
Bayesian_Updater/
├── __init__.py     public API — import everything from here
├── config.py       ThesisParameters, ExitConfig, UpdateResult
├── prior.py        build_prior()
├── update.py       bayesian_update() — the core math
├── metrics.py      realised vol, scale_sigma_L, P(target), hurdle rate
├── data.py         Bloomberg stub + yfinance fallback
└── engine.py       BayesianThesisUpdater — orchestrates all modules
```

Each file has one responsibility. The engine calls them in order — no file
knows about any other except through the engine.

---

## Usage

```python
from quant.Bayesian_Updater import BayesianThesisUpdater, ThesisParameters, ExitConfig
from datetime import date

params = ThesisParameters(
    ticker        = "MU",
    entry_price   = 85.0,
    target_price  = 130.0,
    implied_vol   = 0.45,              # from Bloomberg (decimal: 45% → 0.45)
    holding_years = 1.0,
    risk_free_rate= 0.0432,
    position_date = date(2024, 6, 1),
)

config = ExitConfig(
    p_floor        = 0.30,   # exit if P(target) < 30%
    hurdle_premium = 0.07,   # exit if return < rf + 7%
    lookback_days  = 30,     # rolling window for realised vol
)

engine = BayesianThesisUpdater(params, config)
df     = engine.run_backtest(price_series)   # price_series: pd.Series daily closes
print(engine.summary())
```

---

## The math

### Prior

The prior encodes the analyst's thesis as a Gaussian over the expected total
log-return:

```
μ₀ = log(target / entry)    ← centre on thesis return
σ₀ = IV × √T               ← uncertainty driven by implied vol and horizon
```

`σ₀` uses `√T` because volatility scales with the square root of time — the
longer the horizon, the wider the distribution, but it grows slower than
linearly because random daily moves partially cancel each other out.

### Observation

Every day the model observes the cumulative log-return from entry:

```
x = log(current_price / entry_price)
```

This is always measured from the same anchor point (entry). It is not a daily
return — it is the total return so far.

### Likelihood

The noise on the observation is the realised volatility scaled to the elapsed
time:

```
σ_L = realised_vol × √t_elapsed
```

This scaling is critical. The observation `x` has accumulated noise over
`t_elapsed` years. A raw annual vol figure would over-estimate the noise early
in the holding period and under-estimate it late. Scaling by `√t_elapsed`
brings `σ_L` into the same units as `x`.

**Why this matters:**
- High realised vol → large `σ_L` → noisy observation → posterior barely moves → prior holds
- Low realised vol → small `σ_L` → clean signal → posterior updates fast → exit fires

### Bayesian update (conjugate Normal-Normal)

```
σ_n² = 1 / (1/σ₀² + 1/σ_L²)        ← posterior variance
μ_n  = σ_n² × (μ₀/σ₀² + x/σ_L²)    ← posterior mean
```

This is a precision-weighted average of two opinions: the prior (your thesis)
and the data (what the market delivered). Precision = 1/σ². The tighter the
distribution, the more it weighs in.

The posterior is always narrower than both the prior and the likelihood —
combining two uncertain sources always reduces total uncertainty.

### Probability to target

```
d = [ log(S_t / K) + μ · T_rem ] / (σ · √T_rem)
P(S_T ≥ K) = Φ(d)
```

This is the Black-Scholes d₂ formula. It assumes the stock price at the end
of the remaining holding period follows a log-normal distribution with
parameters from the posterior. `Φ` is the standard normal CDF.

### Risk-adjusted hurdle

```
hurdle = risk_free_rate + hurdle_premium
       = 4.32% + 7.00% = 11.32%
```

Optional Sharpe mode (set `sharpe_equiv` in `ExitConfig`):
```
hurdle = risk_free_rate + sharpe_equiv × realised_vol
```

---

## Bloomberg IV

`data.py` contains a `blpapi` stub. When connected to a terminal, replace
the `fetch_iv_bloomberg()` body with the live call (full instructions in
`data.py` comments).

**Automatic fallback chain:**
```
1. Bloomberg blpapi    ← live, most accurate
2. yfinance ATM IV     ← free, options chain
3. Thesis IV           ← Portfolio Team input, last resort
```

---

## Parameters to update before running

These are the only values you need to change. Everything else is automatic.

### 1. Position parameters — `common_data/portfolio.py`

Change these for every new position you open:

| Parameter | Where | What to put |
|---|---|---|
| `ticker` | `portfolio.py` | Yahoo Finance ticker e.g. `"MU"`, `"MC.PA"` |
| `entry_price` | `portfolio.py` | The exact price you bought at |
| `target_price` | `portfolio.py` | Your analyst's target price |
| `implied_vol` | `portfolio.py` | 30-day IV from Bloomberg as a decimal (e.g. 35% → `0.35`) |
| `holding_years` | `portfolio.py` | How long you plan to hold e.g. `1.0` for 1 year |
| `position_date` | `portfolio.py` | The date you opened the position e.g. `date(2025, 4, 1)` |
| `weight` | `portfolio.py` | Portfolio weight as decimal e.g. `0.20` for 20% |

```python
# common_data/portfolio.py
PositionConfig(
    ticker        = "MU",               # ← change
    name          = "Micron Technology",
    entry_price   = 85.0,               # ← change
    target_price  = 130.0,              # ← change
    implied_vol   = 0.45,               # ← change (from Bloomberg)
    holding_years = 1.0,                # ← change
    position_date = date(2024, 6, 1),   # ← change
    weight        = 0.20,               # ← change
)
```

### 2. Exit thresholds — `run.py`

Change these if you want stricter or looser exit conditions:

| Parameter | Default | Meaning |
|---|---|---|
| `p_floor` | `0.30` | Exit if P(reach target) drops below 30% |
| `hurdle_premium` | `0.07` | Exit if expected return < rf + 7% |
| `lookback_days` | `30` | Rolling window for realised vol calculation |

```python
# run.py
CONFIG = ExitConfig(
    p_floor        = 0.30,   # ← change to make more/less aggressive
    hurdle_premium = 0.07,   # ← change to raise/lower the return bar
    lookback_days  = 30,     # ← change if you want faster/slower vol response
)
```

### 3. Backtest date range — `run.py`

```python
# run.py
START_DATE = "2024-01-01"   # ← start of price history to fetch
END_DATE   = "2025-04-01"   # ← end of backtest period
```

Set `START_DATE` to before your `position_date` so there is enough price
history to compute the initial realised volatility (at least 30 days before).

### 4. Risk-free rate — `common_data/portfolio.py`

```python
risk_free_rate = 0.0432   # ← update when US Treasury rates change
```

Currently set to 4.32% (US 5-year Treasury). Update this when rates move
significantly.

---

## Output fields

`engine.run_backtest(price_series)` returns a `pd.DataFrame` with one row per
trading day and these columns:

| Column | Description |
|---|---|
| `current_price` | Daily close price |
| `posterior_mu` | Updated posterior mean |
| `posterior_sigma` | Updated posterior std |
| `probability_to_target` | P(S_T ≥ target) |
| `expected_return_ann` | Annualised expected remaining return |
| `risk_adjusted_hurdle` | Hurdle rate that day |
| `exit_signal` | True if either trigger fired |
| `exit_reason` | String describing which trigger fired |
| `days_held` | Days since position opened |
| `years_remaining` | Years left in holding period |
