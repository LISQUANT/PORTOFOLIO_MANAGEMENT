"""
Ratio & volatility anomaly monitor.

Signals are computed on a TRAILING window (default 252 trading days) so they
reflect the asset's recent behaviour, not its 15-year average — a static
full-history Sharpe can never fire a "something changed today" trigger.
Full-history values are still printed for context.

Importable functions (used by daily_check.py and tests):
    sharpe_ratio, calmar_ratio, generate_signal,
    rv_bands, rv_anomaly_signal, iv_rv_signal

Usage:
    python quant/ratios.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

STOCKS = {
    "Micron":    "MU",
    "Microsoft": "MSFT",
    "Eli Lilly": "LLY",
    "ASML":      "ASML",
    "LVMH":      "MC.PA",
    "SPY":       "SPY",
}

START_DATE         = "2010-01-01"
END_DATE           = None            # None → today
RISK_FREE_RATE     = 0.0432          # US Treasury
WINDOW             = 30              # rolling window for realised vol
SIGNAL_WINDOW      = 252             # trailing window for Sharpe/Calmar signal
BAND_MULTIPLIER    = 2.0
LOWER_LIMIT_SHARPE = 0.7
LOWER_LIMIT_CALMAR = 1.0


# ── Ratios ────────────────────────────────────────────────────────────────────

def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    excess_returns = daily_returns - (risk_free_rate / 252)
    sd = excess_returns.std()
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.sqrt(252) * excess_returns.mean() / sd)


def calmar_ratio(daily_returns: pd.Series) -> float:
    """
    Annualised return / max drawdown.

    Drawdown is measured on the wealth curve as a RELATIVE peak-to-trough
    decline, and the numerator is annualised (CAGR) — both fixes vs the old
    version, which subtracted cumulative returns arithmetically and used the
    total (non-annualised) return.
    """
    wealth = (1 + daily_returns).cumprod()
    n = len(daily_returns)
    if n == 0 or wealth.iloc[-1] <= 0:
        return float("nan")

    cagr = wealth.iloc[-1] ** (252 / n) - 1

    running_max = wealth.cummax()
    max_dd = ((running_max - wealth) / running_max).max()

    return float(cagr / max_dd) if max_dd > 0 else float("nan")


def generate_signal(sharpe: float, calmar: float,
                    min_sharpe: float = LOWER_LIMIT_SHARPE,
                    min_calmar: float = LOWER_LIMIT_CALMAR) -> str:
    if not (np.isfinite(sharpe) and np.isfinite(calmar)):
        return "Hold"
    if sharpe < min_sharpe or calmar < min_calmar:
        return "Sell"
    if sharpe > min_sharpe and calmar > min_calmar:
        return "Buy"
    return "Hold"


# ── Volatility anomalies ──────────────────────────────────────────────────────

def rv_bands(daily_returns: pd.Series, window: int = WINDOW,
             band_multiplier: float = BAND_MULTIPLIER) -> dict:
    """Latest realised vol vs its rolling mean ± k·std bands."""
    rv       = daily_returns.rolling(window).std() * np.sqrt(252)
    rv_mean  = rv.rolling(window).mean()
    rv_std   = rv.rolling(window).std()

    return {
        "rv":    float(rv.iloc[-1]),
        "mean":  float(rv_mean.iloc[-1]),
        "upper": float(rv_mean.iloc[-1] + band_multiplier * rv_std.iloc[-1]),
        "lower": float(rv_mean.iloc[-1] - band_multiplier * rv_std.iloc[-1]),
    }


def rv_anomaly_signal(rv: float, upper: float, lower: float) -> str:
    if rv > upper:
        return "Attention - RV above upper band (high volatility)"
    if rv < lower:
        return "Attention - RV below lower band (low volatility)"
    return "Normal - RV within the expected range"


def iv_rv_signal(spread: float) -> str:
    if spread > 0.05:
        return "IV >> RV - uncertainty in market pricing"
    if spread < -0.05:
        return "RV >> IV - options appear to be cheap"
    return "IV = RV - spread within the normal range"


def fetch_atm_iv(stock_symbol: str, current_price: float) -> float:
    """Nearest-expiry ATM call IV from yfinance; NaN if unavailable."""
    try:
        ticker = yf.Ticker(stock_symbol)
        options_dates = ticker.options
        if not options_dates:
            return float("nan")
        calls = ticker.option_chain(options_dates[0]).calls.copy()
        calls["diff"] = abs(calls["strike"] - current_price)
        iv = float(calls.loc[calls["diff"].idxmin(), "impliedVolatility"])
        return iv if np.isfinite(iv) and iv > 0 else float("nan")
    except Exception:
        return float("nan")


# ── Report ────────────────────────────────────────────────────────────────────

def analyse(name: str, stock_symbol: str) -> None:
    print(f"\n{'═' * 55}")
    print(f"  {name} ({stock_symbol})")
    print(f"{'═' * 55}")

    data = yf.download(stock_symbol, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)
    if data.empty:
        print(f"No price data returned for {stock_symbol}")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    returns = data["Close"].pct_change().dropna()
    recent  = returns.tail(SIGNAL_WINDOW)

    # Signal is based on the trailing window; full history shown for context
    sharpe_recent = sharpe_ratio(recent)
    calmar_recent = calmar_ratio(recent)
    signal        = generate_signal(sharpe_recent, calmar_recent)

    print(f"Sharpe (last {SIGNAL_WINDOW}d) : {sharpe_recent:.4f}"
          f"   [full history: {sharpe_ratio(returns):.4f}]")
    print(f"Calmar (last {SIGNAL_WINDOW}d) : {calmar_recent:.4f}"
          f"   [full history: {calmar_ratio(returns):.4f}]")
    print(f"Signal              : {signal}")

    bands = rv_bands(returns)
    current_price = float(data["Close"].iloc[-1])
    atm_iv = fetch_atm_iv(stock_symbol, current_price)
    spread = atm_iv - bands["rv"]

    print(f"\nVolatility Analysis (rolling {WINDOW}d window)")
    print(f"Realized Volatility (30d) : {bands['rv']:.4f}")
    print(f"ATM Implied Volatility    : "
          + (f"{atm_iv:.4f}" if np.isfinite(atm_iv) else "N/A"))
    print(f"IV - RV Spread            : "
          + (f"{spread:.4f}" if np.isfinite(spread) else "N/A"))
    print(f"RV Mean                   : {bands['mean']:.4f}")
    print(f"Upper Band                : {bands['upper']:.4f}")
    print(f"Lower Band                : {bands['lower']:.4f}")

    print(f"\nAnomaly Alerts")
    print(f"RV Alert    : {rv_anomaly_signal(bands['rv'], bands['upper'], bands['lower'])}")
    print(f"IV/RV Alert : {iv_rv_signal(spread) if np.isfinite(spread) else 'N/A - no IV data'}")


def main() -> None:
    for name, stock_symbol in STOCKS.items():
        try:
            analyse(name, stock_symbol)
        except Exception as e:
            print(f"  [ERROR] {name} ({stock_symbol}): {e}")


if __name__ == "__main__":
    main()
