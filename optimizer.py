import numpy as np
import pandas as pd
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
data = yf.download(tickers, start='2020-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']
daily_returns = data.pct_change().dropna()

annual_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

print("Annual Returns:")
print(annual_returns)
print("\nCovariance Matrix:")
print(cov_matrix)