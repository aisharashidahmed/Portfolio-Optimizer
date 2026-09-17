"""Analytics built on top of ingested data."""

from .currency import (
    CurrencyDecomposition,
    decompose_currency_risk,
    summarise_currency_exposure,
)

__all__ = [
    "CurrencyDecomposition",
    "decompose_currency_risk",
    "summarise_currency_exposure",
]
