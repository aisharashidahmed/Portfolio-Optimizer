import unittest

from src.metrics import (
    calculate_expected_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
)


class MetricsTests(unittest.TestCase):
    def test_calculates_expected_portfolio_return(self):
        result = calculate_expected_portfolio_return(
            weights=[0.50, 0.50],
            expected_returns=[0.12, 0.08],
        )

        self.assertAlmostEqual(result, 0.10)

    def test_calculates_portfolio_volatility(self):
        result = calculate_portfolio_volatility(
            weights=[0.50, 0.50],
            covariance_matrix=[
                [0.04, -0.02],
                [-0.02, 0.04],
            ],
        )

        self.assertAlmostEqual(result, 0.10)

    def test_calculates_sharpe_ratio(self):
        result = calculate_sharpe_ratio(
            expected_portfolio_return=0.10,
            portfolio_volatility=0.10,
            risk_free_rate=0.04,
        )

        self.assertAlmostEqual(result, 0.60)


if __name__ == "__main__":
    unittest.main()
