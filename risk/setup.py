"""
Daily VaR breach monitor.

Two layers of checks, per asset and for the weighted portfolio:

  1. BREACH CHECK (the daily trigger): did TODAY's return fall below the
     historical 95% / 99% VaR estimated on the trailing window (excluding
     today)? A breach means the portfolio moved outside its expected risk
     envelope — a possible exit/de-risk point.

  2. DISTRIBUTION CHECK: is the asset's VaR itself wider than the static
     risk budget (thresholds)? This flags assets whose return distribution
     has become too fat for the mandate, independent of today's move.

All returns are measured in USD: EUR-quoted lines are converted with EURUSD
before computing returns, so FX risk is included. (No flat FX commission is
applied — a constant multiplicative cost cancels out in returns and has no
effect on VaR.)

Usage:
    python risk/setup.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fpdf import FPDF

# Allow running as `python risk/setup.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_data.portfolio import POSITIONS  # noqa: E402

# Yahoo tickers quoted in EUR that must be converted to USD.
# NOTE: LVMH is "MC.PA" (Paris). Plain "MC" on Yahoo is Moelis & Company!
# The US listing "ASML" is already in USD — do NOT convert it.
EUR_QUOTED = ["MC.PA"]
FX_PAIR    = "EURUSD=X"


# ── Pure computation (testable without network) ──────────────────────────────

def var_breach_metrics(
    returns: pd.DataFrame,
    weights: dict[str, float],
    threshold_95: float = 0.04,
    threshold_99: float = 0.06,
    lookback: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """
    returns  : daily returns per ticker (columns), USD-based
    weights  : {ticker: portfolio weight}; normalised internally

    Returns (per-asset DataFrame, portfolio dict).
    """
    if returns.empty or len(returns) < 30:
        raise ValueError(
            f"Not enough return data to compute VaR "
            f"({len(returns)} rows) — check the data download."
        )

    w = pd.Series(weights, dtype=float)
    w = w / w.sum()

    port_ret = (returns[w.index] * w).sum(axis=1)

    rows = []
    for ticker in returns.columns:
        rows.append(_asset_row(ticker, returns[ticker].dropna(),
                               threshold_95, threshold_99, lookback))

    portfolio = _asset_row("PORTFOLIO", port_ret.dropna(),
                           threshold_95, threshold_99, lookback)

    return pd.DataFrame(rows), portfolio


def _asset_row(name, rets, t95, t99, lookback):
    today_ret = float(rets.iloc[-1])
    hist      = rets.iloc[:-1].tail(lookback)   # VaR window EXCLUDES today

    var95 = float(np.percentile(hist, 5))
    var99 = float(np.percentile(hist, 1))

    breach = "OK"
    if today_ret < var99:
        breach = "CRITICAL (beyond 99% VaR)"
    elif today_ret < var95:
        breach = "WARNING (beyond 95% VaR)"

    tail_stress = var99 / var95 if var95 != 0 else 0.0

    return {
        "Ticker":       name,
        "Today":        f"{today_ret:+.2%}",
        "VaR 95%":      f"{var95:.2%}",
        "VaR 99%":      f"{var99:.2%}",
        "Breach":       breach,
        "Dist 95":      "WIDE" if abs(var95) > t95 else "OK",
        "Dist 99":      "WIDE" if abs(var99) > t99 else "OK",
        "Tail Stress":  f"{tail_stress:.2f}x",
    }


# ── Data layer ────────────────────────────────────────────────────────────────

def fetch_usd_returns(tickers, eur_quoted=EUR_QUOTED, start="2024-01-01") -> pd.DataFrame:
    """Download closes, convert EUR-quoted lines to USD, return daily returns."""
    print(f"--- Fetching Data for {tickers} ---")
    data = yf.download(tickers + [FX_PAIR], start=start,
                       auto_adjust=True, progress=False)["Close"]

    prices = data[tickers].ffill()
    fx     = data[FX_PAIR].ffill()          # USD per EUR

    for t in eur_quoted:
        if t in prices.columns:
            prices[t] = prices[t] * fx

    print("--- Data Processing Complete ---")
    return prices.pct_change().dropna(how="all")


# ── PDF report ────────────────────────────────────────────────────────────────

def export_to_pdf(df: pd.DataFrame, portfolio: dict,
                  t95: float, t99: float, filename="Risk_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, "Portfolio Risk Report", ln=True, align='C')
    pdf.ln(6)

    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        f"Breach check: today's return vs trailing 95%/99% historical VaR (window excludes today).\n"
        f"Distribution check: VaR width vs risk budget ({t95:.1%} @95%, {t99:.1%} @99%).\n"
        f"Base currency: USD (EUR lines converted at EURUSD spot).\n"
        f"Portfolio row uses position weights from common_data/portfolio.py.\n")
    pdf.ln(4)

    table = pd.concat([df, pd.DataFrame([portfolio])], ignore_index=True)

    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(0, 0, 0)
    cols = table.columns
    column_width = 190 / len(cols)
    for col in cols:
        pdf.cell(column_width, 9, col, border=1, align='C')
    pdf.ln()

    pdf.set_font("Arial", size=7)
    for _, row in table.iterrows():
        for col in cols:
            pdf.cell(column_width, 9, str(row[col]), border=1, align='C')
        pdf.ln()

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5,
        "Note: 'Tail Stress' is the 99%/95% VaR ratio. Values above 1.5x suggest "
        "fat tails (leptokurtosis). A CRITICAL breach on the PORTFOLIO row is a "
        "possible exit / de-risk signal.")

    pdf.output(filename)
    print(f"--- PDF Report Generated: {filename} ---")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(threshold_95=0.04, threshold_99=0.06):
    tickers = [p.ticker for p in POSITIONS]
    weights = {p.ticker: p.weight for p in POSITIONS}

    returns = fetch_usd_returns(tickers)
    df, portfolio = var_breach_metrics(returns, weights,
                                       threshold_95, threshold_99)

    print("\n--- Per-Asset ---")
    print(df.to_string(index=False))
    print("\n--- Portfolio ---")
    print(pd.DataFrame([portfolio]).to_string(index=False))

    export_to_pdf(df, portfolio, threshold_95, threshold_99)


if __name__ == "__main__":
    main()
