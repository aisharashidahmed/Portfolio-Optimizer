"""Tests for FX alignment and currency risk decomposition."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.currency import (
    decompose_currency_risk,
    kes_risk_free_advantage,
    summarise_currency_exposure,
)
from src.ingestion.fx import (
    convert_kes_returns_to_usd,
    convert_usd_returns_to_kes,
    fx_returns,
    fx_returns_on_calendar,
)


def business_days(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def make_rate(
    n: int = 500, start_rate: float = 100.0, drift: float = 0.0002, seed: int = 0
):
    """A plausible KES-per-USD path on a daily (7-day) calendar."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    steps = rng.normal(drift, 0.004, n)
    return pd.Series(start_rate * np.cumprod(1 + steps), index=index, name="usd_kes")


class TestCalendarAlignment:
    """The bug this module exists to prevent."""

    def test_returns_telescope_to_the_endpoint_move(self):
        """Compounding aligned returns must recover the actual rate change."""
        rate = make_rate(400)
        target = business_days(200, "2020-01-06")

        aligned = fx_returns_on_calendar(rate, target)
        on_target = (
            rate.reindex(rate.index.union(target)).ffill().reindex(target).dropna()
        )

        expected = on_target.iloc[-1] / on_target.iloc[0] - 1
        assert (1 + aligned).prod() - 1 == pytest.approx(expected, rel=1e-12)

    def test_naive_inner_join_loses_currency_moves(self):
        """The bias, demonstrated.

        Currency markets trade on days equity markets do not. Inner-joining
        pre-computed FX returns onto equity dates silently discards the moves
        that happened in between, even though an investor holding across the
        gap really did bear them.

        On real 2015-2024 USD/KES this inflated measured depreciation from
        41.6% to 67.7%.
        """
        rate = make_rate(600)
        # An equity calendar that misses many currency-market days.
        target = pd.DatetimeIndex(rate.index[::3])

        naive = fx_returns(rate).reindex(target).dropna()
        correct = fx_returns_on_calendar(rate, target)

        on_target = rate.reindex(target)
        truth = on_target.iloc[-1] / on_target.iloc[0] - 1

        assert (1 + correct).prod() - 1 == pytest.approx(truth, rel=1e-12)
        # The naive path drops two-thirds of the moves and misses badly.
        assert abs((1 + naive).prod() - 1 - truth) > abs((1 + correct).prod() - 1 - truth)

    def test_handles_a_target_calendar_with_gaps(self):
        rate = make_rate(300)
        target = business_days(100, "2020-01-06")
        aligned = fx_returns_on_calendar(rate, target)

        assert len(aligned) == len(target) - 1
        assert aligned.notna().all()

    def test_result_is_indexed_on_the_target_calendar(self):
        rate = make_rate(300)
        target = business_days(50, "2020-02-03")
        aligned = fx_returns_on_calendar(rate, target)
        assert aligned.index.isin(target).all()


class TestCurrencyConversion:
    def test_returns_compound_rather_than_add(self):
        """+10% asset and +10% currency is +21% in KES, not +20%."""
        index = business_days(2)
        asset = pd.Series([0.0, 0.10], index=index)
        fx = pd.Series([0.0, 0.10], index=index)

        converted = convert_usd_returns_to_kes(asset, fx)
        assert converted.iloc[1] == pytest.approx(0.21)

    def test_conversion_round_trips(self):
        rate = make_rate(300)
        target = business_days(150, "2020-01-06")
        fx = fx_returns_on_calendar(rate, target)

        rng = np.random.default_rng(1)
        usd = pd.Series(rng.normal(0.0004, 0.01, len(fx)), index=fx.index)

        kes = convert_usd_returns_to_kes(usd, fx)
        back = convert_kes_returns_to_usd(kes, fx)

        pd.testing.assert_series_equal(back, usd, rtol=1e-10, check_names=False)

    def test_a_flat_currency_leaves_returns_unchanged(self):
        index = business_days(20)
        rng = np.random.default_rng(2)
        usd = pd.Series(rng.normal(0, 0.01, 20), index=index)
        flat_fx = pd.Series(0.0, index=index)

        converted = convert_usd_returns_to_kes(usd, flat_fx)
        pd.testing.assert_series_equal(converted, usd, rtol=1e-12, check_names=False)


class TestDecomposition:
    def test_the_three_parts_sum_to_the_total(self):
        """r_kes = r_asset + r_fx + interaction, exactly."""
        rate = make_rate(400)
        target = business_days(200, "2020-01-06")
        rng = np.random.default_rng(3)
        usd = pd.Series(rng.normal(0.0004, 0.012, len(target)), index=target)

        d = decompose_currency_risk(usd, rate)

        assert d.asset_return_usd + d.currency_return + d.interaction == pytest.approx(
            d.total_return_kes, abs=1e-12
        )

    def test_a_depreciating_shilling_helps_a_dollar_holder(self):
        rate = make_rate(400, drift=0.0005, seed=4)  # shilling weakening
        target = business_days(200, "2020-01-06")
        usd = pd.Series(0.0, index=target)  # asset flat, isolate the currency

        d = decompose_currency_risk(usd, rate)

        assert d.currency_return > 0
        assert d.currency_helped
        assert d.total_return_kes == pytest.approx(d.currency_return, abs=1e-9)

    def test_a_strengthening_shilling_hurts_a_dollar_holder(self):
        rate = make_rate(400, drift=-0.0005, seed=5)
        target = business_days(200, "2020-01-06")
        usd = pd.Series(0.0, index=target)

        d = decompose_currency_risk(usd, rate)
        assert d.currency_return < 0
        assert not d.currency_helped

    def test_uncorrelated_currency_risk_adds_to_volatility(self):
        """With near-zero correlation, total variance is the sum of parts."""
        rate = make_rate(600, seed=6)
        target = business_days(300, "2020-01-06")
        rng = np.random.default_rng(7)
        usd = pd.Series(rng.normal(0, 0.012, len(target)), index=target)

        d = decompose_currency_risk(usd, rate)

        assert abs(d.correlation) < 0.2
        assert d.total_volatility_kes > d.asset_volatility
        assert not d.currency_reduced_risk

    def test_currency_share_of_variance_is_a_proportion(self):
        rate = make_rate(500, seed=8)
        target = business_days(250, "2020-01-06")
        rng = np.random.default_rng(9)
        usd = pd.Series(rng.normal(0.0003, 0.015, len(target)), index=target)

        d = decompose_currency_risk(usd, rate)
        assert 0.0 <= d.currency_share_of_variance <= 1.0

    def test_a_calmer_asset_has_a_larger_currency_share(self):
        """The safer the asset, the more the currency dominates its risk."""
        rate = make_rate(600, seed=10)
        target = business_days(300, "2020-01-06")
        rng = np.random.default_rng(11)

        calm = pd.Series(rng.normal(0, 0.004, len(target)), index=target)
        wild = pd.Series(rng.normal(0, 0.030, len(target)), index=target)

        calm_share = decompose_currency_risk(calm, rate).currency_share_of_variance
        wild_share = decompose_currency_risk(wild, rate).currency_share_of_variance

        assert calm_share > wild_share

    def test_rejects_a_return_series_passed_as_the_rate(self):
        """A guard against reintroducing the alignment bug."""
        rate = make_rate(300)
        target = business_days(150, "2020-01-06")
        usd = pd.Series(0.0, index=target)

        with pytest.raises(ValueError, match="looks like a return series"):
            decompose_currency_risk(usd, fx_returns(rate))

    def test_rejects_insufficient_overlap(self):
        rate = make_rate(300)
        lonely = pd.Series([0.01], index=pd.DatetimeIndex(["2030-01-01"]))

        with pytest.raises(ValueError, match="at least 2 overlapping"):
            decompose_currency_risk(lonely, rate)

    def test_reports_the_number_of_periods_used(self):
        rate = make_rate(400)
        target = business_days(200, "2020-01-06")
        usd = pd.Series(0.0, index=target)

        assert decompose_currency_risk(usd, rate).n_periods == len(target) - 1


class TestSummarise:
    def test_one_row_per_asset(self):
        rate = make_rate(400)
        target = business_days(200, "2020-01-06")
        rng = np.random.default_rng(12)
        frame = pd.DataFrame(
            {t: rng.normal(0.0004, 0.012, len(target)) for t in ["AAPL", "MSFT", "SPY"]},
            index=target,
        )

        table = summarise_currency_exposure(frame, rate)

        assert sorted(table.index) == ["AAPL", "MSFT", "SPY"]
        assert "currency_share_of_variance" in table.columns

    def test_every_asset_sees_the_same_currency_return(self):
        rate = make_rate(400)
        target = business_days(200, "2020-01-06")
        rng = np.random.default_rng(13)
        frame = pd.DataFrame(
            {t: rng.normal(0, 0.01, len(target)) for t in ["A", "B"]}, index=target
        )

        table = summarise_currency_exposure(frame, rate)
        assert table["currency_return"].nunique() == 1


class TestRiskFreeAdvantage:
    def test_a_high_kes_rate_is_offset_by_expected_depreciation(self):
        """Kenyan T-bills yielding 16% are not free money.

        Uncovered interest parity: a currency expected to weaken must pay more.
        With 16% KES, 4.5% USD, and 11% expected depreciation, almost nothing
        is left over.
        """
        advantage = kes_risk_free_advantage(
            kes_rate=0.16, usd_rate=0.045, expected_depreciation=0.11
        )
        assert abs(advantage) < 0.02

    def test_depreciation_can_wipe_out_the_nominal_premium(self):
        advantage = kes_risk_free_advantage(
            kes_rate=0.16, usd_rate=0.045, expected_depreciation=0.25
        )
        assert advantage < 0

    def test_a_stable_currency_leaves_the_premium_intact(self):
        advantage = kes_risk_free_advantage(
            kes_rate=0.16, usd_rate=0.045, expected_depreciation=0.0
        )
        assert advantage == pytest.approx(0.115)
