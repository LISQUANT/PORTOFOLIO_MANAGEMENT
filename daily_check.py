"""
Daily portfolio risk monitor — the single entry point for the trigger system.

Run once per trading day (manually or via a scheduler):

    python daily_check.py

Checks, in order:
  1. VaR breach     — did today's move (per asset AND weighted portfolio)
                      fall beyond the trailing 95%/99% historical VaR?
  2. Thesis status  — Bayesian Thesis Updater on the latest close:
                      P(reach target) vs floor, expected return vs hurdle,
                      holding-horizon expiry.
  3. Vol anomalies  — 30d realised vol vs its ±2σ bands, and IV−RV spread
                      (IV fetched live from yfinance options; skipped
                      gracefully if unavailable).

Every triggered alert is printed AND appended to signals.log with a
timestamp, so the log is the audit trail of what the system said and when.
To wire this to email/Telegram, extend `send_alerts()`.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from common_data.portfolio import POSITIONS
from quant.Bayesian_Updater.config import ThesisParameters, ExitConfig
from quant.Bayesian_Updater.engine import BayesianThesisUpdater
from quant.ratios import rv_bands, rv_anomaly_signal, iv_rv_signal, fetch_atm_iv
from risk.setup import var_breach_metrics, EUR_QUOTED, FX_PAIR

LOOKBACK_START = "2024-01-01"   # enough history for 252d VaR + 30d vol
SIGNALS_LOG    = "signals.log"

EXIT_CONFIG = ExitConfig(
    p_floor        = 0.30,
    hurdle_premium = 0.07,
    lookback_days  = 30,
    fetch_live_iv  = False,      # realised vol is primary; flip on with a terminal
)


# ── Alerting ──────────────────────────────────────────────────────────────────

def send_alerts(alerts: list[str]) -> None:
    """
    Deliver triggered alerts. Currently: console + signals.log.
    Extend here for email / Telegram / Slack.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(SIGNALS_LOG, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(f"{stamp}  {a}\n")


# ── Checks ────────────────────────────────────────────────────────────────────

def check_var_breaches(returns: pd.DataFrame, weights: dict) -> list[str]:
    alerts = []
    df, portfolio = var_breach_metrics(returns, weights)

    print("\n── 1. VaR breach check " + "─" * 40)
    print(df.to_string(index=False))
    print(pd.DataFrame([portfolio]).to_string(index=False))

    for _, row in df.iterrows():
        if row["Breach"] != "OK":
            alerts.append(f"[VAR] {row['Ticker']}: {row['Breach']} "
                          f"(today {row['Today']}, VaR95 {row['VaR 95%']})")
    if portfolio["Breach"] != "OK":
        alerts.append(f"[VAR] PORTFOLIO: {portfolio['Breach']} "
                      f"(today {portfolio['Today']}) — possible exit point")
    return alerts


def check_theses(closes: pd.DataFrame) -> list[str]:
    alerts = []
    print("\n── 2. Bayesian thesis status " + "─" * 34)

    for pos in POSITIONS:
        series = closes[pos.ticker].dropna()
        if series.empty:
            alerts.append(f"[DATA] {pos.ticker}: no price data")
            continue

        params = ThesisParameters(
            ticker         = pos.ticker,
            entry_price    = pos.entry_price,
            target_price   = pos.target_price,
            implied_vol    = pos.implied_vol,
            holding_years  = pos.holding_years,
            risk_free_rate = pos.risk_free_rate,
            position_date  = pos.position_date,
        )
        engine = BayesianThesisUpdater(params, EXIT_CONFIG)

        # Each update is independent (static prior), so one update on the
        # latest close with full history gives today's state directly.
        last_ts = series.index[-1]
        as_of   = last_ts.date() if hasattr(last_ts, "date") else date.today()
        r = engine.update(float(series.iloc[-1]), series, as_of=as_of)

        status = f"EXIT — {r.exit_reason}" if r.exit_signal else "HOLD"
        print(f"  {pos.ticker:<8} P(target) {r.probability_to_target:>6.1%}   "
              f"exp.ret {r.expected_return_ann:>+7.1%}   "
              f"hurdle {r.risk_adjusted_hurdle:>6.1%}   {status}")

        if r.exit_signal:
            alerts.append(f"[THESIS] {pos.ticker}: {r.exit_reason}")
    return alerts


def check_vol_anomalies(closes: pd.DataFrame) -> list[str]:
    alerts = []
    print("\n── 3. Volatility anomalies " + "─" * 36)

    for pos in POSITIONS:
        series = closes[pos.ticker].dropna()
        if len(series) < 90:
            continue
        returns = series.pct_change().dropna()
        bands   = rv_bands(returns)
        rv_msg  = rv_anomaly_signal(bands["rv"], bands["upper"], bands["lower"])

        atm_iv = fetch_atm_iv(pos.ticker, float(series.iloc[-1]))
        spread = atm_iv - bands["rv"] if np.isfinite(atm_iv) else float("nan")
        iv_msg = iv_rv_signal(spread) if np.isfinite(spread) else "N/A - no IV data"

        print(f"  {pos.ticker:<8} RV {bands['rv']:.2%}  "
              f"band [{bands['lower']:.2%}, {bands['upper']:.2%}]  | {rv_msg}")

        if rv_msg.startswith("Attention"):
            alerts.append(f"[VOL] {pos.ticker}: {rv_msg}")
        if np.isfinite(spread) and abs(spread) > 0.05:
            alerts.append(f"[VOL] {pos.ticker}: {iv_msg} (spread {spread:+.2%})")
    return alerts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print(f"  DAILY PORTFOLIO RISK CHECK — {date.today()}")
    print("=" * 64)

    tickers = [p.ticker for p in POSITIONS]
    weights = {p.ticker: p.weight for p in POSITIONS}

    raw = yf.download(tickers + [FX_PAIR], start=LOOKBACK_START,
                      auto_adjust=True, progress=False)["Close"]
    closes = raw[tickers].ffill()
    fx     = raw[FX_PAIR].ffill()

    # USD-based returns for the risk layer (EUR lines converted)
    usd = closes.copy()
    for t in EUR_QUOTED:
        if t in usd.columns:
            usd[t] = usd[t] * fx
    returns = usd.pct_change().dropna(how="all")

    alerts: list[str] = []
    alerts += check_var_breaches(returns, weights)
    alerts += check_theses(closes)          # thesis math in local currency
    alerts += check_vol_anomalies(closes)

    print("\n" + "=" * 64)
    if alerts:
        print(f"  ⚠  {len(alerts)} SIGNAL(S) TRIGGERED — possible exit points")
        for a in alerts:
            print(f"   • {a}")
        send_alerts(alerts)
        print(f"\n  Appended to {SIGNALS_LOG}")
    else:
        print("  ✓  All checks passed — nothing out of line today.")
    print("=" * 64)


if __name__ == "__main__":
    main()
