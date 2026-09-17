"""Currency risk decomposition.

The question this module answers: **when a Kenyan investor holds US assets, how
much of what happens to them is the assets, and how much is the shilling?**

For a single period the two returns compound:

    1 + r_kes = (1 + r_usd)(1 + r_fx)

so, expanding,

    r_kes = r_usd + r_fx + r_usd*r_fx
            ^^^^^   ^^^^   ^^^^^^^^^^^
            asset   currency  interaction

Risk does not decompose additively like that, because the two can move
together. Taking variances:

    Var(r_kes) ~= Var(r_usd) + Var(r_fx) + 2*Cov(r_usd, r_fx)

That covariance term is the interesting one. If the shilling reliably weakened
when US equities fell, dollar assets would be a natural hedge for a Kenyan
investor and the total risk would be *lower* than either component. If the two
move together, currency risk compounds equity risk instead of offsetting it.

This is the analysis that makes "should a Kenyan hold US assets?" a quantitative
question rather than an opinion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ingestion.fx import fx_returns_on_calendar

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CurrencyDecomposition:
    """How a KES-denominated return on a USD asset breaks down."""

    asset_return_usd: float
    currency_return: float
    interaction: float
    total_return_kes: float

    asset_volatility: float
    currency_volatility: float
    total_volatility_kes: float

    correlation: float
    currency_share_of_variance: float
    diversification_effect: float

    n_periods: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def currency_helped(self) -> bool:
        """Did the currency move add to the KES-denominated return?"""
        return self.currency_return > 0

    @property
    def currency_reduced_risk(self) -> bool:
        """Did holding a foreign asset lower total risk versus the asset alone?

        True when the currency acted as a hedge rather than an amplifier.
        """
        return self.total_volatility_kes < self.asset_volatility


def _compound(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def _annualised_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def decompose_currency_risk(
    asset_returns_usd: pd.Series, fx_rate: pd.Series
) -> CurrencyDecomposition:
    """Split a KES-denominated result into asset, currency, and interaction.

    Parameters
    ----------
    asset_returns_usd:
        Daily returns of the asset in its own currency (USD).
    fx_rate:
        The KES-per-USD exchange rate **as a level series, not returns**.

    Taking the level and differencing it on the asset's own calendar is
    deliberate. Currency markets trade on days equity markets do not, so
    inner-joining pre-computed FX returns onto equity dates throws away real
    currency moves: on 2015-2024 USD/KES that inflated measured depreciation
    from 41.6% to 67.7%. Requiring the rate here makes the correct alignment
    the only thing a caller can do. See
    :func:`src.ingestion.fx.fx_returns_on_calendar`.
    """
    if isinstance(fx_rate, pd.Series) and len(fx_rate) >= 2:
        median = float(fx_rate.median())
        if -1.0 < median < 1.0:
            raise ValueError(
                "fx_rate looks like a return series, not a rate level "
                f"(median {median:.4f}). Pass the KES-per-USD rate itself so "
                "it can be aligned to the asset's trading calendar."
            )

    currency = fx_returns_on_calendar(fx_rate, asset_returns_usd.index)

    aligned = pd.concat(
        [asset_returns_usd.rename("asset"), currency.rename("fx")], axis=1
    ).dropna()

    if len(aligned) < 2:
        raise ValueError(f"Need at least 2 overlapping observations, got {len(aligned)}")

    asset = aligned["asset"]
    currency = aligned["fx"]
    combined = (1.0 + asset) * (1.0 + currency) - 1.0

    asset_total = _compound(asset)
    currency_total = _compound(currency)
    combined_total = _compound(combined)

    asset_volatility = _annualised_volatility(asset)
    currency_volatility = _annualised_volatility(currency)
    combined_volatility = _annualised_volatility(combined)

    correlation = (
        float(asset.corr(currency)) if asset.std() > 0 and currency.std() > 0 else 0.0
    )

    # Share of total variance attributable to the currency, including its half
    # of the covariance term.
    asset_variance = asset_volatility**2
    currency_variance = currency_volatility**2
    covariance_term = 2.0 * correlation * asset_volatility * currency_volatility
    total_variance = asset_variance + currency_variance + covariance_term

    currency_share = (
        (currency_variance + covariance_term / 2.0) / total_variance
        if total_variance > 0
        else 0.0
    )

    return CurrencyDecomposition(
        asset_return_usd=asset_total,
        currency_return=currency_total,
        # Residual of the exact identity, so the three parts sum to the total.
        interaction=combined_total - asset_total - currency_total,
        total_return_kes=combined_total,
        asset_volatility=asset_volatility,
        currency_volatility=currency_volatility,
        total_volatility_kes=combined_volatility,
        correlation=correlation,
        currency_share_of_variance=float(np.clip(currency_share, 0.0, 1.0)),
        diversification_effect=combined_volatility - asset_volatility,
        n_periods=len(aligned),
    )


def summarise_currency_exposure(
    asset_returns_usd: pd.DataFrame, fx_rate: pd.Series
) -> pd.DataFrame:
    """Decompose every column of a return frame. One row per asset.

    ``fx_rate`` is the KES-per-USD level series, for the reason given in
    :func:`decompose_currency_risk`.
    """
    rows = {}
    for ticker in asset_returns_usd.columns:
        try:
            rows[ticker] = decompose_currency_risk(
                asset_returns_usd[ticker], fx_rate
            ).to_dict()
        except ValueError:
            continue

    return pd.DataFrame(rows).T


def kes_risk_free_advantage(
    kes_rate: float, usd_rate: float, expected_depreciation: float
) -> float:
    """Excess of a KES deposit over a USD deposit after expected depreciation.

    Uncovered interest parity says a currency expected to weaken must offer a
    higher rate to compensate. Kenyan T-bills yielding far above US Treasuries
    are not free money -- they are pricing in shilling depreciation. This
    returns what is left over once that is subtracted, which is the number that
    actually decides the allocation.

    A positive result means the KES rate more than compensates for the expected
    currency loss.
    """
    return (1.0 + kes_rate) / (1.0 + expected_depreciation) - 1.0 - usd_rate
