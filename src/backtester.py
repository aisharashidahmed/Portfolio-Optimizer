import pandas as pd


def calculate_portfolio_daily_returns(test_returns, weights):
    if len(weights) != len(test_returns.columns):
        raise ValueError("Each asset must have a corresponding weight")

    weights_by_ticker = pd.Series(
        weights,
        index=test_returns.columns,
    )

    return test_returns.mul(weights_by_ticker).sum(axis=1)


def calculate_cumulative_returns(portfolio_daily_returns):
    return (1 + portfolio_daily_returns).cumprod() - 1
