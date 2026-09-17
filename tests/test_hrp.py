"""Tests for Hierarchical Risk Parity."""

import numpy as np
import pandas as pd
import pytest

from src.algorithms.hrp import HRPOptimizer, correlation_from_covariance


def make_returns(seed: int = 0, n_obs: int = 750) -> pd.DataFrame:
    """Four assets: A and B move together, C and D are independent."""
    rng = np.random.default_rng(seed)

    common = rng.normal(0, 0.01, n_obs)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0, 0.002, n_obs),
            "B": common + rng.normal(0, 0.002, n_obs),
            "C": rng.normal(0, 0.020, n_obs),
            "D": rng.normal(0, 0.035, n_obs),
        }
    )


class TestCorrelationFromCovariance:
    def test_recovers_the_true_correlation_matrix(self):
        """Regression test.

        The original implementation called ``cov_matrix.corr()``, which
        correlates the covariance matrix's columns as though they were
        observations. On uncorrelated data it invented correlations near -0.4.
        """
        returns = make_returns()
        covariance = returns.cov() * 252

        recovered = correlation_from_covariance(covariance)

        pd.testing.assert_frame_equal(
            recovered, returns.corr(), atol=1e-12, check_exact=False
        )

    def test_diagonal_is_exactly_one(self):
        covariance = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]], index=["X", "Y"], columns=["X", "Y"]
        )
        np.testing.assert_allclose(np.diag(correlation_from_covariance(covariance)), 1.0)

    def test_scaling_an_asset_does_not_change_correlation(self):
        """Correlation is scale-invariant; covariance is not."""
        returns = make_returns()
        base = correlation_from_covariance(returns.cov())

        scaled = returns.copy()
        scaled["D"] *= 1000.0

        pd.testing.assert_frame_equal(
            correlation_from_covariance(scaled.cov()), base, atol=1e-10
        )

    def test_rejects_zero_variance_asset(self):
        covariance = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.0]], index=["X", "FLAT"], columns=["X", "FLAT"]
        )
        with pytest.raises(ValueError, match="non-positive variance"):
            correlation_from_covariance(covariance)


class TestHRPOptimizer:
    def test_weights_sum_to_one(self):
        returns = make_returns()
        weights = HRPOptimizer().optimize(returns, returns.cov() * 252)
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_are_long_only(self):
        returns = make_returns()
        weights = HRPOptimizer().optimize(returns, returns.cov() * 252)
        assert (weights >= 0).all()

    def test_returns_one_weight_per_asset(self):
        returns = make_returns()
        weights = HRPOptimizer().optimize(returns, returns.cov() * 252)
        assert weights.shape == (4,)

    def test_allocates_less_to_the_riskier_asset(self):
        """D has ~1.75x the volatility of C and both are uncorrelated noise."""
        returns = make_returns()
        covariance = returns.cov() * 252
        weights = pd.Series(
            HRPOptimizer().optimize(returns, covariance), index=covariance.columns
        )
        assert weights["D"] < weights["C"]

    def test_single_asset_gets_everything(self):
        returns = pd.DataFrame({"ONLY": np.random.default_rng(1).normal(0, 0.01, 100)})
        weights = HRPOptimizer().optimize(returns, returns.cov() * 252)
        np.testing.assert_allclose(weights, [1.0])

    def test_rejects_empty_universe(self):
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty universe"):
            HRPOptimizer().optimize(empty, empty)

    def test_identical_assets_split_evenly(self):
        """Two perfectly-correlated identical assets should share their branch."""
        rng = np.random.default_rng(7)
        series = rng.normal(0, 0.01, 500)
        returns = pd.DataFrame(
            {"A": series, "A_COPY": series, "B": rng.normal(0, 0.01, 500)}
        )
        covariance = returns.cov() * 252
        weights = pd.Series(
            HRPOptimizer().optimize(returns, covariance), index=covariance.columns
        )
        assert weights["A"] == pytest.approx(weights["A_COPY"], abs=1e-9)

    def test_weights_follow_covariance_column_order(self):
        """Reordering the inputs must permute the outputs, not reshuffle them."""
        returns = make_returns()
        covariance = returns.cov() * 252

        original = pd.Series(
            HRPOptimizer().optimize(returns, covariance), index=covariance.columns
        )

        order = ["D", "B", "C", "A"]
        shuffled_cov = covariance.loc[order, order]
        shuffled = pd.Series(
            HRPOptimizer().optimize(returns[order], shuffled_cov), index=order
        )

        for ticker in order:
            assert shuffled[ticker] == pytest.approx(original[ticker], abs=1e-9)
