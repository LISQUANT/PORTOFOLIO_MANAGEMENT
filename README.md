# PORTOFOLIO_MANAGEMENT

Portfolio risk trigger system: not a trader or a bot — a set of daily checks
that fire a signal whenever the portfolio moves outside its risk parameters,
flagging that an exit is possible now.

## Daily use

```
python daily_check.py
```

Runs three checks on the latest close and appends any triggered alerts to
`signals.log`:

1. **VaR breach** — today's return (per asset and for the weighted
   portfolio, in USD) vs the trailing 95%/99% historical VaR.
2. **Thesis status** — the Bayesian Thesis Updater's exit triggers:
   P(reach target) below floor, expected return below hurdle, or holding
   horizon expired.
3. **Volatility anomalies** — 30d realised vol outside its ±2σ bands, and
   large IV−RV spreads.

Positions (ticker, entry, target, IV, horizon, weight) live in
`common_data/portfolio.py` — keep them current; everything reads from there.

## Repository map

| Path | What it is |
|---|---|
| `common_data/portfolio.py` | The portfolio: one `PositionConfig` per position |
| `common_data/ohlcv.py` | Bloomberg OHLCV → formatted Excel export (needs a terminal) |
| `daily_check.py` | **The daily monitor** — run this every trading day |
| `run.py` | Historical backtest of the Bayesian Thesis Updater |
| `quant/Bayesian_Updater/` | Bayesian exit-signal engine (see its README for the math) |
| `quant/ratios.py` | Trailing Sharpe/Calmar signals + vol anomaly detection |
| `quant/HMM/` | HMM regime detection research (separate README inside) |
| `risk/setup.py` | VaR breach monitor + PDF risk report |
| `tests/` | Offline pytest suite (`pytest` from the repo root) |

## Setup

```
pip install -r requirements.txt
pytest            # verify the math before trusting the signals
```
