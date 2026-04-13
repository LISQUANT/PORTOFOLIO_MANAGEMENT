import pandas as pd
import numpy as np
import yfinance as yf

stock_symbol = "MU"  # Micron Technology, might be changed 
start_date = "2010-01-01"
end_date = "2025-12-31"
risk_free_rate = 0.0432  # US 10-year gov treasury yield as of June 2024

# Daily prices (auto_adjust=True is default, Close = adjusted close)
data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)

# Fix: use 'Close' instead of 'Adj Close'
data['Daily Return'] = data['Close'].pct_change()

# Fix: flatten multi-level columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
    data['Daily Return'] = data['Close'].pct_change()

# Sharpe ratio
def sharpe_ratio(daily_returns, risk_free_rate):
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

# Calmar ratio
def calmar_ratio(daily_returns):
    cumulative_return = (1 + daily_returns).cumprod() - 1
    max_drawdown = (cumulative_return.cummax() - cumulative_return).max()
    return cumulative_return.iloc[-1] / max_drawdown if max_drawdown != 0 else np.nan

# Ratios
sharpe = sharpe_ratio(data['Daily Return'].dropna(), risk_free_rate)
calmar = calmar_ratio(data['Daily Return'].dropna())

# Limits
lower_limit_sharpe = 0.7
lower_limit_calmar = 1.0

# Signal generation
def generate_signal(sharpe, calmar):
    if sharpe < lower_limit_sharpe or calmar < lower_limit_calmar:
        return "Sell"
    elif sharpe > lower_limit_sharpe and calmar > lower_limit_calmar:
        return "Buy"
    else:
        return "Hold"

signal = generate_signal(sharpe, calmar)
print(f'Sharpe Ratio: {sharpe:.4f}')
print(f'Calmar Ratio: {calmar:.4f}')
print(f'Signal: {signal}')