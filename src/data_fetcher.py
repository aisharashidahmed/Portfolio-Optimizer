"""Market data acquisition.

The single boundary between this project and external price data. Everything
downstream consumes the dict returned by :func:`prepare_market_data`, so
provider quirks are normalised here and nowhere else.
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Fewer observations than this and the covariance matrix is not worth trusting.
MIN_OBSERVATIONS = 30


class NoMarketDataError(ValueError):
    """Raised when the provider returned nothing usable."""


def _extract_close_prices(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull a ticker-per-column close-price frame out of yfinance's response.

    yfinance changes shape depending on how many tickers were requested: a
    MultiIndex for several, flat columns for one. Normalise both to a DataFrame.
    """
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Close" not in raw_data.columns.get_level_values(0):
            raise NoMarketDataError("Provider response contained no Close prices")
        prices = raw_data["Close"]
    else:
        if "Close" not in raw_data.columns:
            raise NoMarketDataError("Provider response contained no Close prices")
        prices = raw_data[["Close"]].rename(columns={"Close": tickers[0]})

    return prices if isinstance(prices, pd.DataFrame) else prices.to_frame()


def fetch_price_data(
    tickers: list[str], start_date: str, end_date: str
) -> tuple[pd.DataFrame, list[str]]:
    """Download adjusted close prices.

    Returns the price frame and the tickers that came back unusable, rather
    than failing the whole request because one symbol was delisted or typo'd.
    """
    if not tickers:
        raise ValueError("At least one ticker is required")

    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,  # splits/dividends folded into Close
        progress=False,
    )

    if raw_data is None or raw_data.empty:
        raise NoMarketDataError(
            f"No price data returned for {tickers} between {start_date} and {end_date}"
        )

    prices = _extract_close_prices(raw_data, tickers)

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns or prices[ticker].isna().all()
    ]
    prices = prices.drop(columns=missing_tickers, errors="ignore")

    # A column that is mostly gaps will poison the covariance estimate.
    sparse = [c for c in prices.columns if prices[c].notna().sum() < MIN_OBSERVATIONS]
    if sparse:
        logger.warning("Dropping tickers with too few observations: %s", sparse)
        missing_tickers.extend(sparse)
        prices = prices.drop(columns=sparse)

    if prices.empty or prices.shape[1] == 0:
        raise NoMarketDataError(f"Every requested ticker was unusable: {missing_tickers}")

    # Forward-fill single-day holes (exchange holidays differ across markets --
    # this matters once NSE and NYSE data sit side by side), then drop any date
    # still incomplete so every asset shares one aligned calendar.
    prices = prices.ffill().dropna(how="any")

    return prices, missing_tickers


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def calculate_expected_annual_returns(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def calculate_annual_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def prepare_market_data(tickers: list[str], start_date: str, end_date: str) -> dict:
    """Fetch prices and derive every statistic the optimizers need."""
    prices, missing_tickers = fetch_price_data(tickers, start_date, end_date)

    daily_returns = calculate_daily_returns(prices)

    if len(daily_returns) < MIN_OBSERVATIONS:
        raise NoMarketDataError(
            f"Only {len(daily_returns)} usable observations; need at least "
            f"{MIN_OBSERVATIONS} to estimate a covariance matrix"
        )

    return {
        "prices": prices,
        "daily_returns": daily_returns,
        "expected_annual_returns": calculate_expected_annual_returns(daily_returns),
        "annual_covariance": calculate_annual_covariance(daily_returns),
        "missing_tickers": missing_tickers,
    }
