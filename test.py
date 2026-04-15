import pandas as pd
import numpy as np
import yfinance as yf

stock_symbol = "MU"  # Micron Technology, might be changed 
start_date = "2010-01-01"
end_date = "2025-12-31"
risk_free_rate = 0.0432  # US 10-year gov treasury yield as of June 2024
window           = 30         # in days
band_multiplier  = 2.0        


# Daily prices 
data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
data['Daily Return'] = data['Close'].pct_change()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
    data['Daily Return'] = data['Close'].pct_change()

def sharpe_ratio(daily_returns, risk_free_rate):
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calmar_ratio(daily_returns):
    cumulative_return = (1 + daily_returns).cumprod() - 1
    max_drawdown = (cumulative_return.cummax() - cumulative_return).max()
    return cumulative_return.iloc[-1] / max_drawdown if max_drawdown != 0 else np.nan

sharpe = sharpe_ratio(data['Daily Return'].dropna(), risk_free_rate)
calmar = calmar_ratio(data['Daily Return'].dropna())

lower_limit_sharpe = 0.7
lower_limit_calmar = 1.0


def generate_signal(sharpe, calmar):
    if sharpe < lower_limit_sharpe or calmar < lower_limit_calmar:
        return "Sell"
    elif sharpe > lower_limit_sharpe and calmar > lower_limit_calmar:
        return "Buy"
    else:
        return "Hold"

returns = data["Daily Return"].dropna()
sharpe  = sharpe_ratio(returns, risk_free_rate)
calmar  = calmar_ratio(returns)
signal  = generate_signal(sharpe, calmar)
 
print(f"Sharpe Ratio : {sharpe:.4f}")
print(f"Calmar Ratio : {calmar:.4f}")
print(f"Signal       : {signal}")

data["Realized_Volatility"] = data["Daily Return"].rolling(window).std() * np.sqrt(252)


ticker = yf.Ticker(stock_symbol)
options_dates = ticker.options
 
if not options_dates:
    raise ValueError("without information about this data")
 
nearest_expiry = options_dates[0]
opt_chain = ticker.option_chain(nearest_expiry)
calls = opt_chain.calls
 
current_price = float(data["Close"].iloc[-1])
calls["diff"] = abs(calls["strike"] - current_price)
atm_iv = float(calls.loc[calls["diff"].idxmin(), "impliedVolatility"])

data["RV_Rolling_Mean"] = data["Realized_Volatility"].rolling(window).mean()
data["RV_Rolling_Std"] = data["Realized_Volatility"].rolling(window).std()
data["Upper_Band"] = data["RV_Rolling_Mean"] + band_multiplier * data["RV_Rolling_Std"]
data["Lower_Band"] = data["RV_Rolling_Mean"] - band_multiplier * data["RV_Rolling_Std"]
 
latest_rv = float(data["Realized_Volatility"].iloc[-1])
upper = float(data["Upper_Band"].iloc[-1])
lower = float(data["Lower_Band"].iloc[-1])
rv_mean = float(data["RV_Rolling_Mean"].iloc[-1])

def rv_anomaly_signal(rv, upper, lower):
    if rv > upper:
        return "Attention - RV above upper band, might mean unusual high volatility"
    elif rv < lower:
        return "Attention - RV below lower band,might mean unusually low volatility"
    else:
        return "Normal - RV within the expected range"
 
 
def iv_rv_signal(spread):
    if spread > 0.05:
        return "IV >> RV - uncertainty in market pricing"
    elif spread < -0.05:
        return "RV >> IV - options appear to be cheap"
    else:
        return "IV = RV - spread within the normal range"

spread = atm_iv - latest_rv
 
print(f"\nVolatility Analysis (rolling {window}d window)")
print(f"Realized Volatility (30d): {latest_rv:.4f}")
print(f"ATM Implied Volatility: {atm_iv:.4f}")
print(f"IV - RV Spread: {spread:.4f}")
print(f"RV Mean: {rv_mean:.4f}")
print(f"Upper Band: {upper:.4f}")
print(f"Lower Band: {lower:.4f}")
 
print(f"\nAnomaly Alerts")
print(f"RV Alert: {rv_anomaly_signal(latest_rv, upper, lower)}")
print(f"IV/RV Alert: {iv_rv_signal(spread)}")