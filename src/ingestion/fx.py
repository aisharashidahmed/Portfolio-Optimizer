"""Foreign exchange rates.

A Kenyan investor's return on a US asset is not the asset's return. It is the
asset's return compounded with the KES/USD move, and over the last decade the
currency term has been large: the shilling went from 90.5 to 128.2 per dollar
between 2015 and 2024, so simply holding dollars returned roughly 42% in KES
terms before any asset appreciation at all.

Yahoo quotes this pair as ``KES=X``, in KES per USD. A *rising* quote means the
shilling weakened, which makes dollar assets worth more in shillings.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Yahoo's symbol for KES per USD.
USD_KES_SYMBOL = "KES=X"

TRADING_DAYS_PER_YEAR = 252


class FXDataError(RuntimeError):
    """Exchange rate data could not be obtained."""


def fetch_usd_kes(start_date: str, end_date: str) -> pd.Series:
    """Daily KES per USD.

    Returns a series indexed by date. Higher means a weaker shilling.
    """
    raw = yf.download(
        USD_KES_SYMBOL,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise FXDataError(
            f"No USD/KES data between {start_date} and {end_date}. "
            f"The pair is quoted as {USD_KES_SYMBOL!r}."
        )

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    rate = close.dropna()
    rate.name = "usd_kes"

    if rate.empty:
        raise FXDataError("USD/KES series was entirely missing values")

    # A rate outside this band means the symbol changed meaning, not that the
    # shilling moved.
    if not (10.0 < float(rate.median()) < 1000.0):
        raise FXDataError(
            f"Implausible USD/KES median of {float(rate.median()):.2f}; "
            f"the source may have changed quoting convention"
        )

    return rate


def fx_returns(rate: pd.Series) -> pd.Series:
    """Daily percentage change in the exchange rate.

    Positive means the shilling weakened, which *helps* a Kenyan holding
    dollar assets.
    """
    returns = rate.pct_change().dropna()
    returns.name = "fx_return"
    return returns


def fx_returns_on_calendar(rate: pd.Series, target_index: pd.Index) -> pd.Series:
    """Exchange-rate returns measured between consecutive *target* dates.

    This exists because of a bias that is easy to introduce and hard to see.

    The currency market trades on days the US equity market is closed. Taking
    daily FX returns and then inner-joining them onto equity trading days
    silently **discards** the currency moves that happened on those extra days
    -- and those moves really did affect an investor holding the asset across
    the gap.

    Measured on 2015-2024 USD/KES, the discarded days compounded to -15.6%, so
    the naive join reported 67.7% cumulative depreciation against a true 41.6%.
    Every currency-attributed return was overstated by more than half.

    The fix is to reindex the *rate* onto the target calendar and difference it
    there, so the return from one equity trading day to the next spans all
    intervening currency movement -- which is exactly the exposure an investor
    holding over that period actually bears.
    """
    target_index = pd.DatetimeIndex(target_index)

    # Forward-fill across the union so every target date has a rate, then
    # sample only the target dates and difference those.
    union = rate.index.union(target_index)
    on_target = rate.reindex(union).ffill().reindex(target_index)

    returns = on_target.pct_change().dropna()
    returns.name = "fx_return"
    return returns


def convert_usd_returns_to_kes(
    usd_returns: pd.DataFrame | pd.Series, fx_return_series: pd.Series
) -> pd.DataFrame | pd.Series:
    """Restate USD-denominated returns in shillings.

    The two returns compound rather than add:

        1 + r_kes = (1 + r_usd)(1 + r_fx)

    The cross term ``r_usd * r_fx`` is small daily but compounds over a decade,
    so it is kept rather than dropped.
    """
    aligned_fx = fx_return_series.reindex(usd_returns.index).dropna()
    aligned_assets = usd_returns.loc[aligned_fx.index]

    if isinstance(aligned_assets, pd.DataFrame):
        return aligned_assets.mul(1 + aligned_fx, axis=0).add(aligned_fx, axis=0)

    return (1 + aligned_assets) * (1 + aligned_fx) - 1


def convert_kes_returns_to_usd(
    kes_returns: pd.DataFrame | pd.Series, fx_return_series: pd.Series
) -> pd.DataFrame | pd.Series:
    """Restate KES-denominated returns in dollars -- the mirror image."""
    aligned_fx = fx_return_series.reindex(kes_returns.index).dropna()
    aligned_assets = kes_returns.loc[aligned_fx.index]

    return (1 + aligned_assets).div(1 + aligned_fx, axis=0) - 1
