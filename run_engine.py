# run_engine.py
# This is the "Switchboard" that turns on the HRP Robot.

import numpy as np
import pandas as pd
import yfinance as yf

# Import your HRP robot from the engine room
from src.algorithms.mvo import MVOOptimizer

print("="*60)
print("STARTING THE HRP ROBOT TEST")
print("="*60)

# 1. Feed the robot with data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
print(f"\n[1] Fetching data for: {tickers}...")
data = yf.download(tickers, start='2020-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']
daily_returns = data.pct_change().dropna()

annual_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

print(f"\n[2] Data fetched. Annual Returns:")
print(annual_returns.round(4))

# 2. Build the HRP robot
print(f"\n[3] Initializing HRP Robot...")
mvo_robot = MVOOptimizer()

# 3. Tell the robot to optimize
print(f"[4] HRP Robot is walking the tree...")
weights = mvo_robot.optimize(daily_returns, cov_matrix)

# 4. Display the results
print(f"\n[5] FINAL OPTIMAL WEIGHTS:")
for ticker, weight in zip(tickers, weights):
    print(f"    {ticker}: {weight:.2%}")

print(f"\n[6] Verification: Sum of weights = {np.sum(weights):.4f} (should be 1.0)")
print("\n" + "="*60)
print("HRP ROBOT TEST COMPLETE!")
print("="*60)