"""Property-based tests.

Example-based tests check the cases we thought of. These assert invariants that
must hold for *every* input, and let Hypothesis search for the counterexample.
Each property here is one that, if violated, makes the output financially
meaningless rather than merely wrong in the last decimal place.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from src.algorithms import EqualWeightOptimizer, HRPOptimizer, MVOOptimizer
from src.algorithms.mvo import OptimizationFailedError
from src.backtesting.performance import (
    annualised_volatility,
    drawdown_series,
    max_drawdown,
    total_return,
    win_rate,
)
from src.metrics import calculate_portfolio_volatility
from src.risk.covariance import condition_number, ledoit_wolf_shrinkage

# Daily returns in a plausible range. Without bounds Hypothesis produces values
# like 1e300, which say nothing about whether the finance is right.
daily_return_values = st.floats(
    min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False
)

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def return_frames(draw, min_assets=2, max_assets=8, min_obs=60, max_obs=250):
    """A DataFrame of daily returns with no degenerate (constant) columns."""
    n_assets = draw(st.integers(min_assets, max_assets))
    n_obs = draw(st.integers(min_obs, max_obs))

    values = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs, n_assets),
            elements=daily_return_values,
        )
    )

    # A zero-variance column is not a tradable asset and every estimator here
    # legitimately rejects it.
    assume(np.all(values.std(axis=0) > 1e-6))

    return pd.DataFrame(values, columns=[f"A{i}" for i in range(n_assets)])


@st.composite
def continuous_return_frames(draw, min_assets=2, max_assets=8):
    """Returns drawn from a continuous distribution.

    Hypothesis's raw float arrays are full of exact duplicates, which produce
    tied correlation distances. Some properties only hold on generic data, so
    this strategy varies shape and seed while keeping the values continuous --
    Hypothesis still explores the input space, just not the degenerate corner.
    """
    n_assets = draw(st.integers(min_assets, max_assets))
    n_obs = draw(st.integers(80, 300))
    seed = draw(st.integers(0, 2**31 - 1))

    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_obs)
    betas = rng.uniform(0.3, 1.8, n_assets)
    values = np.outer(market, betas) + rng.normal(0, 0.008, (n_obs, n_assets))

    return pd.DataFrame(values, columns=[f"A{i}" for i in range(n_assets)])


class TestAllocationInvariants:
    """Whatever the data, an allocation must be a valid portfolio."""

    @pytest.mark.parametrize(
        "make_optimizer",
        [
            pytest.param(lambda: HRPOptimizer(), id="hrp"),
            pytest.param(lambda: EqualWeightOptimizer(), id="equal_weight"),
            pytest.param(lambda: MVOOptimizer(max_weight=1.0), id="mvo"),
        ],
    )
    @given(returns=return_frames())
    @SLOW
    def test_weights_form_a_valid_portfolio(self, make_optimizer, returns):
        covariance = returns.cov() * 252

        try:
            weights = make_optimizer().optimize(returns, covariance)
        except OptimizationFailedError:
            # MVO's solver legitimately cannot converge on some degenerate
            # covariance matrices. Refusing loudly is acceptable; returning an
            # invalid portfolio is not, which is what the assertions below pin.
            return

        assert len(weights) == returns.shape[1], "one weight per asset"
        assert np.all(np.isfinite(weights)), "no NaN or inf weights"
        assert np.all(weights >= -1e-9), "long-only: no short positions"
        assert weights.sum() == pytest.approx(1.0, abs=1e-6), "fully invested"

    @given(returns=return_frames(), cap=st.floats(min_value=0.3, max_value=1.0))
    @SLOW
    def test_mvo_never_breaches_its_weight_cap(self, returns, cap):
        """When MVO returns an allocation, the cap must hold.

        SLSQP is a local method and genuinely cannot converge on some
        degenerate covariance matrices -- Hypothesis finds them readily. That
        is a real limitation of mean-variance optimization, not a defect, so
        the optimizer raises a typed error pointing the caller at HRP. What is
        *not* acceptable is silently returning an allocation that breaks the
        constraint, which is what this asserts.
        """
        n_assets = returns.shape[1]
        # The cap must admit a feasible solution: n * cap >= 1.
        assume(n_assets * cap >= 1.0)

        try:
            weights = MVOOptimizer(max_weight=cap).optimize(returns, returns.cov() * 252)
        except OptimizationFailedError:
            return  # failing loudly is allowed; failing silently is not

        assert weights.max() <= cap + 1e-6
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-9)

    @given(returns=continuous_return_frames())
    @SLOW
    def test_hrp_is_invariant_to_asset_ordering(self, returns):
        """Reordering columns must permute weights, not change them.

        Restricted to distance matrices without near-ties. Hierarchical
        clustering has no canonical dendrogram when distances tie -- scipy
        breaks the tie by index order -- so a partially-tied matrix is
        genuinely order-dependent. HRPOptimizer handles the fully-degenerate
        case explicitly (see ``_distances_are_degenerate``); partial ties are
        an inherent property of the clustering algorithm and are documented
        rather than papered over.
        """
        covariance = returns.cov() * 252

        original = pd.Series(
            HRPOptimizer().optimize(returns, covariance), index=covariance.columns
        )

        order = list(reversed(covariance.columns))
        reordered = pd.Series(
            HRPOptimizer().optimize(returns[order], covariance.loc[order, order]),
            index=order,
        )

        for ticker in order:
            assert reordered[ticker] == pytest.approx(original[ticker], abs=1e-8)


class TestCovarianceInvariants:
    @given(returns=return_frames())
    @SLOW
    def test_shrinkage_intensity_is_always_a_proportion(self, returns):
        _, intensity = ledoit_wolf_shrinkage(returns)
        assert 0.0 <= intensity <= 1.0

    @given(returns=return_frames())
    @SLOW
    def test_shrunk_matrix_is_always_symmetric_psd(self, returns):
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        values = shrunk.to_numpy()

        np.testing.assert_allclose(values, values.T, atol=1e-15)
        assert np.linalg.eigvalsh(values).min() >= -1e-12, "no negative variance"

    @given(returns=return_frames())
    @SLOW
    def test_shrinkage_always_produces_an_invertible_matrix(self, returns):
        """With any positive shrinkage, the result is non-singular.

        This is the guarantee shrinkage actually provides, and it is the one
        that matters: a singular covariance matrix cannot be inverted, so
        mean-variance optimization has no solution at all.

        Note what is *not* claimed. The intuitive property "shrinkage always
        improves the condition number" is false, and Hypothesis disproved it
        twice while this suite was being written:

        * On an already well-conditioned matrix it can nudge the number up by
          a fraction of a percent (1.5622 -> 1.5843) -- noise, on a matrix
          nobody needed help with.
        * More interestingly, on a matrix whose ill-conditioning comes from
          variances spanning three orders of magnitude rather than from
          correlation structure (260.93 -> 263.02). The constant-correlation
          target preserves each asset's variance exactly, so it cannot repair
          conditioning caused by the variance spread.

        The narrower conditioning claim -- that shrinkage repairs a covariance
        matrix made singular by having fewer observations than assets -- is
        verified concretely in tests/test_covariance.py.
        """
        shrunk, intensity = ledoit_wolf_shrinkage(returns)
        assume(intensity > 0.01)

        eigenvalues = np.linalg.eigvalsh(shrunk.to_numpy())

        assert eigenvalues.min() > 0, "shrunk matrix must be invertible"
        assert np.isfinite(condition_number(shrunk))

    @given(returns=return_frames())
    @SLOW
    def test_variances_are_preserved_exactly(self, returns):
        """The constant-correlation target keeps each asset's own variance."""
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        np.testing.assert_allclose(
            np.diag(shrunk), np.diag(returns.cov() * 252), rtol=1e-9
        )


class TestRiskInvariants:
    @given(returns=return_frames())
    @SLOW
    def test_portfolio_volatility_is_never_negative(self, returns):
        """Variance is w'Σw, a quadratic form on a PSD matrix: always >= 0."""
        covariance, _ = ledoit_wolf_shrinkage(returns)
        n_assets = returns.shape[1]
        weights = [1.0 / n_assets] * n_assets

        volatility = calculate_portfolio_volatility(weights, covariance.to_numpy())

        assert volatility >= 0
        assert np.isfinite(volatility)

    @given(returns=return_frames())
    @SLOW
    def test_diversification_never_increases_risk(self, returns):
        """An equal-weight portfolio cannot be riskier than its worst asset.

        Correlations are at most 1, so the portfolio's volatility is bounded by
        the weighted average of individual volatilities.
        """
        covariance = (returns.cov() * 252).to_numpy()
        n_assets = len(covariance)
        weights = np.full(n_assets, 1.0 / n_assets)

        portfolio_volatility = calculate_portfolio_volatility(list(weights), covariance)
        worst_asset_volatility = np.sqrt(np.diag(covariance)).max()

        assert portfolio_volatility <= worst_asset_volatility + 1e-9


class TestPerformanceInvariants:
    @given(
        values=st.lists(daily_return_values, min_size=2, max_size=400),
    )
    @SLOW
    def test_drawdown_is_never_positive(self, values):
        assert max_drawdown(pd.Series(values)) <= 1e-12

    @given(values=st.lists(daily_return_values, min_size=2, max_size=400))
    @SLOW
    def test_drawdown_series_matches_its_minimum(self, values):
        series = pd.Series(values)
        assert drawdown_series(series).min() == pytest.approx(max_drawdown(series))

    @given(values=st.lists(daily_return_values, min_size=2, max_size=400))
    @SLOW
    def test_volatility_is_never_negative(self, values):
        assert annualised_volatility(pd.Series(values)) >= 0

    @given(values=st.lists(daily_return_values, min_size=1, max_size=400))
    @SLOW
    def test_win_rate_is_a_proportion(self, values):
        assert 0.0 <= win_rate(pd.Series(values)) <= 1.0

    @given(values=st.lists(daily_return_values, min_size=1, max_size=200))
    @SLOW
    def test_total_return_cannot_lose_more_than_everything(self, values):
        """Long-only with returns above -100% can never breach total loss."""
        assert total_return(pd.Series(values)) >= -1.0

    @given(
        values=st.lists(daily_return_values, min_size=2, max_size=200),
        scale=st.floats(min_value=0.1, max_value=10.0),
    )
    @SLOW
    def test_volatility_scales_linearly_with_return_magnitude(self, values, scale):
        series = pd.Series(values)
        assume(annualised_volatility(series) > 1e-9)

        assert annualised_volatility(series * scale) == pytest.approx(
            annualised_volatility(series) * scale, rel=1e-9
        )


class TestNumericalStability:
    @given(
        shape=array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=6),
        scale=st.sampled_from([1e-6, 1e-4, 1e-2, 1.0]),
    )
    @SLOW
    def test_estimators_survive_extreme_scales(self, shape, scale):
        """Prices quoted in KES are ~130x a USD figure; scale must not break anything."""
        n_obs = max(shape[0], 60)
        n_assets = shape[1]

        rng = np.random.default_rng(abs(hash((n_obs, n_assets, scale))) % 2**32)
        values = rng.normal(0, scale, (n_obs, n_assets))
        assume(np.all(values.std(axis=0) > 1e-15))

        returns = pd.DataFrame(values, columns=[f"A{i}" for i in range(n_assets)])
        shrunk, intensity = ledoit_wolf_shrinkage(returns)

        assert np.all(np.isfinite(shrunk.to_numpy()))
        assert 0.0 <= intensity <= 1.0
