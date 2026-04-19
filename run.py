"""
Usage
-----
    python run.py
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

from common_data.portfolio import POSITIONS
from quant.Bayesian_Updater.config  import ThesisParameters, ExitConfig
from quant.Bayesian_Updater.engine  import BayesianThesisUpdater


# ── Settings ──────────────────────────────────────────────────────────────────

START_DATE = "2024-01-01"
END_DATE   = "2025-04-01"

CONFIG = ExitConfig(
    p_floor        = 0.30,   # exit if P(reach target) < 30%
    hurdle_premium = 0.07,   # exit if expected return < rf + 7%
    lookback_days  = 30,     # rolling window for realised vol
)


# ── Step 1: Fetch prices ──────────────────────────────────────────────────────

tickers = [p.ticker for p in POSITIONS]

print("=" * 60)
print("  LISQUANT — Bayesian Thesis Updater")
print("=" * 60)
print(f"\nFetching prices for: {', '.join(tickers)}\n")

raw    = yf.download(tickers, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw


# ── Step 2: Run Bayesian updater for each position ────────────────────────────

results = {}   # ticker → DataFrame
engines = {}   # ticker → BayesianThesisUpdater

for pos in POSITIONS:

    series = closes[pos.ticker].dropna()

    params = ThesisParameters(
        ticker         = pos.ticker,
        entry_price    = pos.entry_price,
        target_price   = pos.target_price,
        implied_vol    = pos.implied_vol,
        holding_years  = pos.holding_years,
        risk_free_rate = pos.risk_free_rate,
        position_date  = pos.position_date,
    )
    
    engine             = BayesianThesisUpdater(params, CONFIG)
    df                 = engine.run_backtest(series)
    results[pos.ticker] = df
    engines[pos.ticker] = engine

    print(engine.summary())

    fe = engine.first_exit()
    if fe:
        print(f"  First exit triggered : {fe.date}")
        print(f"  Reason               : {fe.exit_reason}\n")
    else:
        print("  No exit triggered during backtest period.\n")


# ── Step 3: Exit signal summary table ─────────────────────────────────────────

print("=" * 60)
print("  EXIT SIGNAL SUMMARY")
print("=" * 60)
print(f"  {'Ticker':<8}  {'P(target)':>9}  {'Exp. Return':>11}  {'Hurdle':>8}  Status")
print(f"  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*8}  {'─'*20}")

for pos in POSITIONS:
    df   = results[pos.ticker]
    fe   = engines[pos.ticker].first_exit()
    last = df.iloc[-1]

    status = f"EXIT {fe.date}" if fe else "HOLD"
    print(
        f"  {pos.ticker:<8}  "
        f"{last['probability_to_target']:>8.1%}  "
        f"{last['expected_return_ann']:>+10.1%}  "
        f"{last['risk_adjusted_hurdle']:>7.1%}  "
        f"{status}"
    )

print("=" * 60)


# ── Step 4: Plot ──────────────────────────────────────────────────────────────
"""
n   = len(POSITIONS)
fig, axes = plt.subplots(3, n, figsize=(5 * n, 11))
fig.suptitle("LISQUANT — Bayesian Thesis Updater", fontsize=14, fontweight="bold")

for col, pos in enumerate(POSITIONS):
    df  = results[pos.ticker]
    ax0 = axes[0][col]
    ax1 = axes[1][col]
    ax2 = axes[2][col]

    # ── Price ──
    ax0.plot(df.index, df["current_price"], color="steelblue", lw=1.2)
    ax0.axhline(pos.target_price, color="green",  ls="--", lw=1,   label="Target")
    ax0.axhline(pos.entry_price,  color="orange", ls=":",  lw=1,   label="Entry")
    exits = df[df["exit_signal"]]
    if not exits.empty:
        ax0.axvline(pd.Timestamp(exits.index[0]), color="red",
                    lw=1.5, alpha=0.8, label="Exit")
    ax0.set_title(f"{pos.ticker}", fontweight="bold")
    ax0.set_ylabel("Price ($)")
    ax0.legend(fontsize=6)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax0.tick_params(axis="x", rotation=30, labelsize=7)

    # ── P(reach target) ──
    ax1.plot(df.index, df["probability_to_target"] * 100,
             color="purple", lw=1.2)
    ax1.axhline(CONFIG.p_floor * 100, color="red", ls="--",
                lw=1, label=f"Floor {CONFIG.p_floor:.0%}")
    ax1.fill_between(
        df.index,
        df["probability_to_target"] * 100,
        CONFIG.p_floor * 100,
        where=df["probability_to_target"] < CONFIG.p_floor,
        color="red", alpha=0.15,
    )
    ax1.set_ylabel("P(target) %")
    ax1.legend(fontsize=6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.tick_params(axis="x", rotation=30, labelsize=7)

    # ── Expected return vs hurdle ──
    ax2.plot(df.index, df["expected_return_ann"] * 100,
             color="teal", lw=1.2, label="Exp. return")
    ax2.plot(df.index, df["risk_adjusted_hurdle"] * 100,
             color="red", ls="--", lw=1, label="Hurdle")
    ax2.fill_between(
        df.index,
        df["expected_return_ann"] * 100,
        df["risk_adjusted_hurdle"] * 100,
        where=df["expected_return_ann"] < df["risk_adjusted_hurdle"],
        color="orange", alpha=0.2,
    )
    ax2.set_ylabel("Return %")
    ax2.legend(fontsize=6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2.tick_params(axis="x", rotation=30, labelsize=7)

plt.tight_layout()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
print(f"\nChart saved → output.png")
plt.show()
"""