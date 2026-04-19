"""
All external data fetching lives here.
  fetch_iv_bloomberg() : Pull 30-day ATM IV from Bloomberg (stub)
  fetch_iv_yfinance()   : yfinance ATM IV fallback
  get_implied_vol()     : Resolution chain — Bloomberg → yfinance → thesis IV
"""

from __future__ import annotations
from datetime import date
from typing import Optional

import numpy as np
import yfinance as yf


# ── Bloomberg stub ────────────────────────────────────────────────────────────

def fetch_iv_bloomberg(ticker: str, as_of: date) -> Optional[float]:
    """
    Pull 30-day ATM implied volatility from Bloomberg terminal.
    """
    try:
        import blpapi

        session_options = blpapi.SessionOptions()
        session_options.setServerHost("localhost")
        session_options.setServerPort(8194)

        session = blpapi.Session(session_options)
        session.start()
        session.openService("//blp/refdata")

        refDataService = session.getService("//blp/refdata")
        request = refDataService.createRequest("ReferenceDataRequest")

        request.getElement("securities").appendValue(f"{ticker} US Equity")
        request.getElement("fields").appendValue("30DAY_IMPVOL_100%MNY_DF")

        session.sendRequest(request)

        while True:
            event = session.nextEvent(500)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData").getValueAsElement(0)
                    field_data = sec_data.getElement("fieldData")
                    if field_data.hasElement("30DAY_IMPVOL_100%MNY_DF"):
                        raw = field_data.getElementAsFloat("30DAY_IMPVOL_100%MNY_DF")
                        session.stop()
                        return float(raw) / 100.0

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()
        return None

    except Exception as e:
        print(f"  [Bloomberg] {ticker}: {e}")
        return None

# ── yfinance fallback ─────────────────────────────────────────────────────────

def fetch_iv_yfinance(ticker: str, current_price: float) -> Optional[float]:
    try:
        t             = yf.Ticker(ticker)
        options_dates = t.options
        if not options_dates:
            return None
        calls         = t.option_chain(options_dates[0]).calls.copy()
        calls["diff"] = abs(calls["strike"] - current_price)
        iv            = float(calls.loc[calls["diff"].idxmin(), "impliedVolatility"])
        return iv if (np.isfinite(iv) and iv > 0) else None
    except Exception:
        return None


def get_implied_vol(ticker: str, current_price: float, as_of: date, thesis_iv: float,) -> float:
    iv = fetch_iv_bloomberg(ticker, as_of)
    if iv and iv > 0:
        return iv

    iv = fetch_iv_yfinance(ticker, current_price)
    if iv and iv > 0:
        return iv

    return thesis_iv
