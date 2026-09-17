"""Tests for the walk-forward backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.algorithms import EqualWeightOptimizer, HRPOptimizer, MVOOptimizer
from src.backtesting.engine import (
    BacktestConfigError,
    _drift_weights,
    compare_strategies,
    run_walk_forward,
)


def factor_returns(n_assets=6, n_obs=800, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.010, n_obs)
    betas = rng.uniform(0.6, 1.4, n_assets)
    data = np.outer(market, betas) + rng.normal(0, 0.008, (n_obs, n_assets))
    return pd.DataFrame(
        data,
        index=pd.date_range("2019-01-01", periods=n_obs, freq="B"),
        columns=[f"A{i}" for i in range(n_assets)],
    )


class TestNoLookahead:
    """The property that decides whether a backtest means anything."""

    def test_future_data_cannot_change_past_weights(self):
        """Corrupt everything after a cutoff; weights before it must not move.

        If any weight changed, the optimizer saw data from its own future and
        every performance number the engine produces is fiction.
        """
        returns = factor_returns(n_obs=700)
        cutoff = 500

        baseline = run_walk_forward(returns, HRPOptimizer(), training_days=252)

        corrupted = returns.copy()
        rng = np.random.default_rng(1234)
        corrupted.iloc[cutoff:] = rng.normal(0.05, 0.20, corrupted.iloc[cutoff:].shape)

        tampered = run_walk_forward(corrupted, HRPOptimizer(), training_days=252)

        before_cutoff = baseline.weights_history.index < returns.index[cutoff]
        pd.testing.assert_frame_equal(
            baseline.weights_history[before_cutoff],
            tampered.weights_history[before_cutoff],
        )

    def test_returns_before_the_cutoff_are_also_unchanged(self):
        returns = factor_returns(n_obs=700)
        cutoff = 500

        baseline = run_walk_forward(returns, EqualWeightOptimizer(), training_days=252)

        corrupted = returns.copy()
        corrupted.iloc[cutoff:] *= 10.0
        tampered = run_walk_forward(corrupted, EqualWeightOptimizer(), training_days=252)

        mask = baseline.net_returns.index < returns.index[cutoff]
        pd.testing.assert_series_equal(
            baseline.net_returns[mask], tampered.net_returns[mask]
        )


class TestWeightDrift:
    def test_a_winner_grows_its_share_of_the_portfolio(self):
        """50/50, one asset +10% and one flat -> 52.4% / 47.6%."""
        drifted = _drift_weights(np.array([0.5, 0.5]), np.array([0.10, 0.0]))
        np.testing.assert_allclose(drifted, [0.55 / 1.05, 0.50 / 1.05])

    def test_drifted_weights_still_sum_to_one(self):
        rng = np.random.default_rng(2)
        weights = rng.dirichlet(np.ones(5))
        drifted = _drift_weights(weights, rng.normal(0, 0.05, 5))
        assert drifted.sum() == pytest.approx(1.0)

    def test_equal_returns_leave_weights_untouched(self):
        weights = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(_drift_weights(weights, np.full(3, 0.07)), weights)

    def test_total_loss_yields_zero_weights(self):
        np.testing.assert_allclose(
            _drift_weights(np.array([0.5, 0.5]), np.array([-1.0, -1.0])), [0.0, 0.0]
        )

    def test_engine_lets_weights_drift_between_rebalances(self):
        """Weights must move on non-rebalance days rather than stay pinned."""
        returns = factor_returns(n_obs=500)
        result = run_walk_forward(
            returns, EqualWeightOptimizer(), training_days=252, rebalance_every=60
        )

        first_ten = result.weights_history.iloc[:10]
        assert not np.allclose(first_ten.iloc[0], first_ten.iloc[9])


class TestTransactionCosts:
    def test_zero_cost_makes_net_identical_to_gross(self):
        result = run_walk_forward(
            factor_returns(), HRPOptimizer(), transaction_cost_bps=0.0
        )
        pd.testing.assert_series_equal(
            result.net_returns, result.gross_returns, check_names=False
        )

    def test_higher_costs_monotonically_reduce_returns(self):
        returns = factor_returns()
        totals = [
            run_walk_forward(
                returns, HRPOptimizer(), transaction_cost_bps=bps
            ).performance.total_return
            for bps in (0.0, 10.0, 50.0, 200.0)
        ]
        assert totals == sorted(totals, reverse=True)

    def test_cost_drag_is_positive_when_trading_happens(self):
        result = run_walk_forward(
            factor_returns(), MVOOptimizer(max_weight=0.5), transaction_cost_bps=25.0
        )
        assert result.total_cost_drag > 0

    def test_first_rebalance_trades_the_whole_portfolio(self):
        """Going from all cash to fully invested is turnover of 1.0."""
        result = run_walk_forward(factor_returns(), EqualWeightOptimizer())
        assert result.turnover.iloc[0] == pytest.approx(1.0)

    def test_mvo_trades_more_than_equal_weight(self):
        """A documented weakness of mean-variance: unstable weights, high turnover."""
        returns = factor_returns()
        mvo = run_walk_forward(returns, MVOOptimizer(max_weight=0.5))
        equal = run_walk_forward(returns, EqualWeightOptimizer())
        assert mvo.average_turnover > equal.average_turnover

    def test_negative_costs_are_rejected(self):
        with pytest.raises(BacktestConfigError, match="cannot be negative"):
            run_walk_forward(factor_returns(), HRPOptimizer(), transaction_cost_bps=-1)


class TestRebalanceSchedule:
    def test_rebalance_count_follows_the_schedule(self):
        returns = factor_returns(n_obs=800)
        result = run_walk_forward(
            returns, EqualWeightOptimizer(), training_days=252, rebalance_every=50
        )
        out_of_sample = 800 - 252
        assert len(result.turnover) == int(np.ceil(out_of_sample / 50))

    def test_rebalancing_daily_trades_every_day(self):
        returns = factor_returns(n_obs=400)
        result = run_walk_forward(
            returns, EqualWeightOptimizer(), training_days=252, rebalance_every=1
        )
        assert len(result.turnover) == 400 - 252

    def test_equal_weight_rebalanced_daily_has_near_zero_turnover_after_the_first(self):
        """Daily rebalancing only has to undo one day of drift."""
        result = run_walk_forward(
            factor_returns(n_obs=400),
            EqualWeightOptimizer(),
            training_days=252,
            rebalance_every=1,
        )
        assert result.turnover.iloc[1:].max() < 0.10


class TestOutputShape:
    def test_out_of_sample_length_is_correct(self):
        returns = factor_returns(n_obs=600)
        result = run_walk_forward(returns, HRPOptimizer(), training_days=252)
        assert len(result.net_returns) == 600 - 252

    def test_weights_history_covers_every_out_of_sample_day(self):
        returns = factor_returns(n_obs=600)
        result = run_walk_forward(returns, HRPOptimizer(), training_days=252)
        assert len(result.weights_history) == len(result.net_returns)
        assert list(result.weights_history.columns) == list(returns.columns)

    def test_weights_are_long_only_and_fully_invested_at_rebalance(self):
        result = run_walk_forward(factor_returns(), HRPOptimizer())
        assert (result.weights_history.to_numpy() >= -1e-9).all()
        assert result.weights_history.sum(axis=1).max() == pytest.approx(1.0, abs=1e-6)

    def test_equity_curve_matches_compounded_returns(self):
        result = run_walk_forward(factor_returns(), EqualWeightOptimizer())
        expected = (1.0 + result.net_returns).cumprod()
        pd.testing.assert_series_equal(result.equity_curve(), expected)

    def test_summary_reports_diagnostics(self):
        result = run_walk_forward(factor_returns(), HRPOptimizer())
        summary = result.summary()
        assert summary["covariance_estimator"] == "ledoit_wolf"
        assert summary["n_rebalances"] > 0
        assert 0.0 <= summary["mean_shrinkage_intensity"] <= 1.0


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"training_days": 1}, "training_days"),
            ({"rebalance_every": 0}, "rebalance_every"),
            ({"training_days": 10_000}, "out-of-sample"),
        ],
    )
    def test_rejects_bad_configuration(self, kwargs, match):
        with pytest.raises(BacktestConfigError, match=match):
            run_walk_forward(factor_returns(), HRPOptimizer(), **kwargs)

    def test_rejects_empty_data(self):
        with pytest.raises(BacktestConfigError, match="empty"):
            run_walk_forward(pd.DataFrame(), HRPOptimizer())

    def test_rejects_a_single_asset(self):
        single = factor_returns(n_assets=1, n_obs=500)
        with pytest.raises(BacktestConfigError, match="at least 2 assets"):
            run_walk_forward(single, HRPOptimizer())


class TestCompareStrategies:
    def test_returns_one_row_per_strategy(self):
        table = compare_strategies(
            factor_returns(),
            {"equal": EqualWeightOptimizer(), "hrp": HRPOptimizer()},
        )
        assert list(table.index) == ["equal", "hrp"]
        assert "sharpe_ratio" in table.columns

    def test_all_strategies_see_identical_data(self):
        """Same optimizer under two names must produce identical results."""
        table = compare_strategies(
            factor_returns(),
            {"a": EqualWeightOptimizer(), "b": EqualWeightOptimizer()},
        )
        assert table.loc["a", "sharpe_ratio"] == pytest.approx(
            table.loc["b", "sharpe_ratio"]
        )
