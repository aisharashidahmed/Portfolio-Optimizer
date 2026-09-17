from src.algorithms.equal_weight import create_equal_weights
from src.algorithms.mvo import MVOOptimizer
from src.backtester import (
    calculate_portfolio_daily_returns,
    calculate_cumulative_returns,
)
from src.data_fetcher import prepare_market_data


tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
risk_free_rate = 0.04

# 1. Fetch two separate periods
training_data = prepare_market_data(
    tickers,
    "2020-01-01",
    "2024-01-01",
)

testing_data = prepare_market_data(
    tickers,
    "2024-01-01",
    "2025-01-01",
)

if training_data["missing_tickers"] or testing_data["missing_tickers"]:
    raise ValueError("Some requested tickers are unavailable")

# 2. Preserve the training column order
asset_names = list(training_data["daily_returns"].columns)

training_returns = training_data["daily_returns"][asset_names]
training_covariance = training_data["annual_covariance"].loc[
    asset_names,
    asset_names,
]

testing_returns = testing_data["daily_returns"][asset_names]

# 3. Learn weights using training data only
mvo = MVOOptimizer(
    risk_free_rate=risk_free_rate,
    max_weight=0.40,
)

mvo_weights = mvo.optimize(
    training_returns,
    training_covariance,
)

equal_weights = create_equal_weights(len(asset_names))

# 4. Apply frozen weights to unseen testing data
mvo_daily_returns = calculate_portfolio_daily_returns(
    testing_returns,
    mvo_weights,
)

equal_daily_returns = calculate_portfolio_daily_returns(
    testing_returns,
    equal_weights,
)

# 5. Compound the daily returns
mvo_cumulative_returns = calculate_cumulative_returns(
    mvo_daily_returns
)

equal_cumulative_returns = calculate_cumulative_returns(
    equal_daily_returns
)

# 6. Display the learned allocations and unseen performance
print("\nWeights learned from training data")
print("----------------------------------")

for ticker, mvo_weight, equal_weight in zip(
    asset_names,
    mvo_weights,
    equal_weights,
):
    print(
        f"{ticker}: "
        f"MVO {mvo_weight:.2%} | "
        f"Equal {equal_weight:.2%}"
    )

print("\nOut-of-sample results")
print("---------------------")
print(
    "MVO cumulative return:",
    f"{mvo_cumulative_returns.iloc[-1]:.2%}",
)
print(
    "Equal-weight cumulative return:",
    f"{equal_cumulative_returns.iloc[-1]:.2%}",
)
