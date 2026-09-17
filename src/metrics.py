import math


def calculate_expected_portfolio_return(weights, expected_returns):
    asset_count = len(weights)

    if len(weights) != len(expected_returns):
        raise ValueError("Each weight must have a corresponding return")

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    expected_portfolio_return = 0.0

    for i in range(asset_count):
        expected_portfolio_return += weights[i] * expected_returns[i]

    return expected_portfolio_return


def calculate_portfolio_volatility(weights, covariance_matrix):
    asset_count = len(weights)
    portfolio_variance = 0.0

    for i in range(asset_count):
        for j in range(asset_count):
            portfolio_variance += weights[i] * weights[j] * covariance_matrix[i][j]

    return math.sqrt(portfolio_variance)


def calculate_sharpe_ratio(
    expected_portfolio_return,
    portfolio_volatility,
    risk_free_rate,
):
    if portfolio_volatility <= 0:
        raise ValueError("Portfolio volatility must be greater than zero")

    return (expected_portfolio_return - risk_free_rate) / portfolio_volatility


def calculate_portfolio_metrics(
    weights,
    expected_returns,
    covariance_matrix,
    risk_free_rate,
):
    expected_annual_return = calculate_expected_portfolio_return(
        weights,
        expected_returns,
    )

    annual_volatility = calculate_portfolio_volatility(
        weights,
        covariance_matrix,
    )

    sharpe_ratio = calculate_sharpe_ratio(
        expected_annual_return,
        annual_volatility,
        risk_free_rate,
    )

    return {
        "expected_annual_return": expected_annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
    }
