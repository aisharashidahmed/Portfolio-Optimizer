"""Tests for performance statistics.

Most cases use hand-computable numbers so the expected value can be verified
on paper rather than by running the code being tested.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.performance import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarise,
    total_return,
    value_at_risk,
    win_rate,
)


class TestTotalReturn:
    def test_compounds_rather_than_adds(self):
        """+10% then +10% is +21%, not +20%."""
        assert total_return(pd.Series([0.10, 0.10])) == pytest.approx(0.21)

    def test_a_50_percent_loss_needs_a_100_percent_gain_to_recover(self):
        assert total_return(pd.Series([-0.50, 1.00])) == pytest.approx(0.0)

    def test_empty_series_returns_zero(self):
        assert total_return(pd.Series(dtype=float)) == 0.0


class TestAnnualisedReturn:
    def test_is_geometric_not_arithmetic(self):
        """+50% then -50% is a 25% loss; the arithmetic mean says break-even."""
        returns = pd.Series([0.50, -0.50])
        assert annualised_return(returns, periods_per_year=2) == pytest.approx(-0.25)

    def test_one_full_year_returns_the_period_return(self):
        returns = pd.Series([0.01] * 12)
        expected = 1.01**12 - 1
        assert annualised_return(returns, periods_per_year=12) == pytest.approx(expected)

    def test_total_wipeout_reports_minus_one(self):
        assert annualised_return(pd.Series([-1.0, 0.05])) == pytest.approx(-1.0)


class TestVolatility:
    def test_scales_with_the_square_root_of_time(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        expected = returns.std(ddof=1) * np.sqrt(252)
        assert annualised_volatility(returns) == pytest.approx(expected)

    def test_a_constant_series_has_zero_volatility(self):
        assert annualised_volatility(pd.Series([0.01] * 50)) == pytest.approx(0.0)

    def test_single_observation_has_no_volatility(self):
        assert annualised_volatility(pd.Series([0.01])) == 0.0


class TestDrawdown:
    def test_computes_peak_to_trough_by_hand(self):
        """+100% to 2.0, then -50% to 1.0: a 50% drawdown from the peak."""
        assert max_drawdown(pd.Series([1.0, -0.5])) == pytest.approx(-0.5)

    def test_a_monotonically_rising_series_never_draws_down(self):
        assert max_drawdown(pd.Series([0.01] * 20)) == pytest.approx(0.0)

    def test_measures_from_the_peak_not_the_start(self):
        """Up 100%, down 50%, up 100%: the drawdown is 50%, not 0%."""
        assert max_drawdown(pd.Series([1.0, -0.5, 1.0])) == pytest.approx(-0.5)

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(3)
        drawdowns = drawdown_series(pd.Series(rng.normal(0.001, 0.02, 500)))
        assert (drawdowns <= 1e-15).all()

    def test_drawdown_series_aligns_with_the_input(self):
        returns = pd.Series(
            [0.1, -0.2, 0.05], index=pd.date_range("2024-01-01", periods=3)
        )
        assert list(drawdown_series(returns).index) == list(returns.index)


class TestRiskAdjustedRatios:
    def test_sharpe_is_excess_return_over_volatility(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
        expected = (annualised_return(returns) - 0.03) / annualised_volatility(returns)
        assert sharpe_ratio(returns, risk_free_rate=0.03) == pytest.approx(expected)

    def test_sharpe_is_zero_when_there_is_no_volatility(self):
        assert sharpe_ratio(pd.Series([0.01] * 30)) == 0.0

    def test_sortino_exceeds_sharpe_when_upside_dominates(self):
        """Sortino ignores upside swings, so a series with big gains and small
        losses is penalised less by Sortino than by Sharpe."""
        returns = pd.Series([0.05, 0.06, -0.005, 0.07, -0.004, 0.05, -0.006] * 20)
        assert sortino_ratio(returns) > sharpe_ratio(returns)

    def test_sortino_is_zero_without_enough_losing_periods(self):
        assert sortino_ratio(pd.Series([0.01] * 30)) == 0.0

    def test_calmar_is_return_over_worst_drawdown(self):
        returns = pd.Series([0.02, -0.05, 0.03, 0.01] * 30)
        expected = annualised_return(returns) / abs(max_drawdown(returns))
        assert calmar_ratio(returns) == pytest.approx(expected)

    def test_calmar_is_zero_without_a_drawdown(self):
        assert calmar_ratio(pd.Series([0.01] * 20)) == 0.0


class TestTailRisk:
    def test_var_is_the_fifth_percentile(self):
        returns = pd.Series(np.linspace(-0.10, 0.10, 101))
        assert value_at_risk(returns, 0.95) == pytest.approx(np.percentile(returns, 5))

    def test_cvar_is_at_least_as_bad_as_var(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        assert conditional_value_at_risk(returns) <= value_at_risk(returns)

    def test_both_are_zero_for_an_empty_series(self):
        empty = pd.Series(dtype=float)
        assert value_at_risk(empty) == 0.0
        assert conditional_value_at_risk(empty) == 0.0


class TestWinRate:
    def test_counts_positive_periods(self):
        assert win_rate(pd.Series([0.01, -0.01, 0.02, -0.03])) == pytest.approx(0.5)

    def test_zero_returns_do_not_count_as_wins(self):
        assert win_rate(pd.Series([0.0, 0.0, 0.01])) == pytest.approx(1 / 3)


class TestSummarise:
    def test_reports_every_statistic(self):
        rng = np.random.default_rng(7)
        stats = summarise(pd.Series(rng.normal(0.0004, 0.012, 600)))

        assert set(stats.to_dict()) == {
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "value_at_risk_95",
            "conditional_value_at_risk_95",
            "win_rate",
            "n_periods",
        }
        assert stats.n_periods == 600

    def test_ignores_missing_observations(self):
        stats = summarise(pd.Series([0.01, np.nan, 0.02]))
        assert stats.n_periods == 2
