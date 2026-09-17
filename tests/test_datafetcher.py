import unittest

import numpy as np
import pandas as pd
import pytest

from src.data_fetcher import (
    MIN_OBSERVATIONS,
    NoMarketDataError,
    calculate_annual_covariance,
    calculate_daily_returns,
    calculate_expected_annual_returns,
    fetch_price_data,
    prepare_market_data,
)


class DataFetcherTests(unittest.TestCase):
    def test_calculates_daily_returns(self):
        prices = pd.DataFrame(
            {
                "AAPL": [100.00, 110.00, 99.00],
                "MSFT": [100.00, 105.00, 110.25],
            }
        )

        expected = pd.DataFrame(
            {
                "AAPL": [0.10, -0.10],
                "MSFT": [0.05, 0.05],
            },
            index=[1, 2],
        )

        result = calculate_daily_returns(prices)

        pd.testing.assert_frame_equal(result, expected)

    def test_calculates_expected_annual_returns(self):
        daily_returns = pd.DataFrame(
            {
                "AAPL": [0.01, 0.03],
                "MSFT": [-0.01, 0.01],
            }
        )

        expected = pd.Series(
            {
                "AAPL": 5.04,
                "MSFT": 0.00,
            }
        )

        result = calculate_expected_annual_returns(daily_returns)

        pd.testing.assert_series_equal(result, expected)

    def test_calculates_annual_covariance(self):
        daily_returns = pd.DataFrame(
            {
                "AAPL": [0.01, 0.03],
                "MSFT": [-0.01, 0.01],
            }
        )

        expected = pd.DataFrame(
            {
                "AAPL": [0.0504, 0.0504],
                "MSFT": [0.0504, 0.0504],
            },
            index=["AAPL", "MSFT"],
        )

        result = calculate_annual_covariance(daily_returns)

        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# Provider-boundary tests. yfinance is faked so these never touch the network.
# ---------------------------------------------------------------------------


def multi_ticker_frame(tickers, n_obs=120, nan_columns=()):
    """Mimic yfinance's MultiIndex response for several tickers."""
    index = pd.date_range("2023-01-02", periods=n_obs, freq="B")
    rng = np.random.default_rng(0)

    columns = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    frame = pd.DataFrame(
        rng.uniform(90, 110, size=(n_obs, len(columns))), index=index, columns=columns
    )
    for ticker in nan_columns:
        frame[("Close", ticker)] = np.nan
    return frame


class TestFetchPriceData:
    def test_requires_at_least_one_ticker(self):
        with pytest.raises(ValueError, match="At least one ticker"):
            fetch_price_data([], "2023-01-01", "2023-06-01")

    def test_raises_when_provider_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(
            "src.data_fetcher.yf.download", lambda *a, **k: pd.DataFrame()
        )
        with pytest.raises(NoMarketDataError, match="No price data"):
            fetch_price_data(["AAPL", "MSFT"], "2023-01-01", "2023-06-01")

    def test_reports_all_nan_ticker_as_missing_instead_of_failing(self, monkeypatch):
        """One bad symbol must not sink the whole request."""
        frame = multi_ticker_frame(["AAPL", "DELISTED"], nan_columns=["DELISTED"])
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        prices, missing = fetch_price_data(
            ["AAPL", "DELISTED"], "2023-01-01", "2023-06-01"
        )

        assert missing == ["DELISTED"]
        assert list(prices.columns) == ["AAPL"]

    def test_drops_tickers_with_too_few_observations(self, monkeypatch):
        frame = multi_ticker_frame(["AAPL", "THIN"])
        frame.loc[frame.index[MIN_OBSERVATIONS - 5 :], ("Close", "THIN")] = np.nan
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        prices, missing = fetch_price_data(["AAPL", "THIN"], "2023-01-01", "2023-06-01")

        assert "THIN" in missing
        assert list(prices.columns) == ["AAPL"]

    def test_raises_when_every_ticker_is_unusable(self, monkeypatch):
        frame = multi_ticker_frame(["A", "B"], nan_columns=["A", "B"])
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        with pytest.raises(NoMarketDataError, match="Every requested ticker"):
            fetch_price_data(["A", "B"], "2023-01-01", "2023-06-01")

    def test_handles_flat_columns_from_a_single_ticker(self, monkeypatch):
        """yfinance returns flat columns for one ticker and a MultiIndex for many."""
        index = pd.date_range("2023-01-02", periods=120, freq="B")
        flat = pd.DataFrame(
            {"Close": np.linspace(100, 120, 120), "Volume": 1_000}, index=index
        )
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: flat)

        prices, missing = fetch_price_data(["AAPL"], "2023-01-01", "2023-06-01")

        assert list(prices.columns) == ["AAPL"]
        assert missing == []

    def test_aligns_calendars_across_assets(self, monkeypatch):
        """A holiday on one exchange must not leave a NaN row in the output."""
        frame = multi_ticker_frame(["NYSE_CO", "NSE_CO"])
        frame.loc[frame.index[10], ("Close", "NSE_CO")] = np.nan
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        prices, _ = fetch_price_data(["NYSE_CO", "NSE_CO"], "2023-01-01", "2023-06-01")

        assert not prices.isna().to_numpy().any()
        assert len(prices) == len(frame)  # ffilled, not dropped


class TestPrepareMarketData:
    def test_returns_every_downstream_key(self, monkeypatch):
        frame = multi_ticker_frame(["AAPL", "MSFT"])
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        data = prepare_market_data(["AAPL", "MSFT"], "2023-01-01", "2023-06-01")

        assert set(data) == {
            "prices",
            "daily_returns",
            "expected_annual_returns",
            "annual_covariance",
            "missing_tickers",
        }

    def test_covariance_is_square_and_matches_the_universe(self, monkeypatch):
        frame = multi_ticker_frame(["AAPL", "MSFT", "GOOGL"])
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        data = prepare_market_data(["AAPL", "MSFT", "GOOGL"], "2023-01-01", "2023-06-01")
        covariance = data["annual_covariance"]

        assert covariance.shape == (3, 3)
        assert list(covariance.columns) == list(data["daily_returns"].columns)

    def test_rejects_a_window_with_too_little_history(self, monkeypatch):
        frame = multi_ticker_frame(["AAPL", "MSFT"], n_obs=10)
        monkeypatch.setattr("src.data_fetcher.yf.download", lambda *a, **k: frame)

        with pytest.raises(NoMarketDataError):
            prepare_market_data(["AAPL", "MSFT"], "2023-01-01", "2023-01-15")
