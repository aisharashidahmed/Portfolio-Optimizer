from src.algorithms.equal_weight import create_equal_weights
from src.algorithms.mvo import MVOOptimizer
from src.data_fetcher import prepare_market_data
from src.metrics import calculate_portfolio_metrics


tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
start_date = "2020-01-01"
end_date = "2025-01-01"
risk_free_rate = 0.04

market_data = prepare_market_data(
    tickers,
    start_date,
    end_date,
)

daily_returns = market_data["daily_returns"]
expected_returns = (
    market_data["expected_annual_returns"].to_numpy()
)
annual_covariance_dataframe = market_data["annual_covariance"]
annual_covariance_array = annual_covariance_dataframe.to_numpy()

asset_names = list(daily_returns.columns)
asset_count = len(asset_names)

mvo = MVOOptimizer(
    risk_free_rate=risk_free_rate,
    max_weight=0.40,
)

mvo_weights = mvo.optimize(
    daily_returns,
    annual_covariance_dataframe,
)

equal_weights = create_equal_weights(asset_count)

mvo_metrics = calculate_portfolio_metrics(
    mvo_weights,
    expected_returns,
    annual_covariance_array,
    risk_free_rate,
)

equal_metrics = calculate_portfolio_metrics(
    equal_weights,
    expected_returns,
    annual_covariance_array,
    risk_free_rate,
)


def print_portfolio(name, weights, metrics):
    print(f"\n{name}")
    print("-" * len(name))

    for ticker, weight in zip(asset_names, weights):
        print(f"{ticker}: {weight:.2%}")

    print(
        "Expected annual return:",
        f"{metrics['expected_annual_return']:.2%}",
    )
    print(
        "Annual volatility:",
        f"{metrics['annual_volatility']:.2%}",
    )
    print(
        "Sharpe ratio:",
        f"{metrics['sharpe_ratio']:.2f}",
    )


print_portfolio("MVO Portfolio", mvo_weights, mvo_metrics)
print_portfolio(
    "Equal-Weight Portfolio",
    equal_weights,
    equal_metrics,
)

if market_data["missing_tickers"]:
    print("\nMissing tickers:", market_data["missing_tickers"])