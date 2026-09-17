"""Typed records for ingested market data.

Validation happens here, at the edge, so nothing malformed reaches the store.
Scraped data is hostile input: a site redesign turns a price column into an
empty string, a suspended stock reports a blank, a thousands separator appears
without warning. Every one of those must fail loudly with a usable message
rather than land in the database as a NaN that poisons a covariance matrix six
modules downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

# NSE tickers are short uppercase alphanumerics: SCOM, EQTY, KCB, BAT, C&G.
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9&.\-]{0,11}$")

# Prices are quoted in KES. A share trading above this is almost certainly a
# parse error (a merged cell, a market cap read as a price).
MAX_PLAUSIBLE_PRICE_KES = 1_000_000.0


class ValidationError(ValueError):
    """A scraped record failed validation."""


def parse_number(raw: str | float | int | None, field_name: str) -> float | None:
    """Parse a scraped numeric cell.

    Handles the shapes these tables actually contain: thousands separators,
    explicit signs, percent suffixes, blanks for suspended counters, and the
    various dashes sites use to mean "no data".
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = raw.strip()
    if text in {"", "-", "--", "—", "–", "N/A", "n/a", "NA", "null"}:
        return None

    cleaned = text.replace(",", "").replace("%", "").replace("+", "").strip()

    # Some sources wrap negatives in parentheses: (1.25) means -1.25.
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    # Volume shorthand: 10.8M, 1.2K, 3.4B.
    multiplier = 1.0
    if cleaned and cleaned[-1] in "KMB":
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[cleaned[-1]]
        cleaned = cleaned[:-1]

    try:
        # Decimal first: it rejects junk that float() would silently accept,
        # such as 'nan' and 'inf'.
        value = float(Decimal(cleaned)) * multiplier
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(
            f"{field_name}: could not parse {raw!r} as a number"
        ) from exc

    if value != value or value in (float("inf"), float("-inf")):
        raise ValidationError(f"{field_name}: {raw!r} is not a finite number")

    return value


def normalise_ticker(raw: str) -> str:
    """Canonicalise a ticker symbol, or reject it."""
    if raw is None:
        raise ValidationError("ticker: missing")

    ticker = raw.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise ValidationError(f"ticker: {raw!r} is not a valid NSE symbol")

    return ticker


@dataclass(frozen=True)
class PriceBar:
    """One ticker's close on one trading day, as published.

    Deliberately close-only. AFX publishes a reliable close and volume; it does
    not publish a consistent OHLC series, and inventing open/high/low from a
    close would be fabrication.
    """

    ticker: str
    trade_date: date
    close_kes: float
    volume: float | None = None
    source: str = "afx"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.close_kes is None:
            raise ValidationError(f"{self.ticker}: close price is required")
        if self.close_kes <= 0:
            raise ValidationError(
                f"{self.ticker}: close price must be positive, got {self.close_kes}"
            )
        if self.close_kes > MAX_PLAUSIBLE_PRICE_KES:
            raise ValidationError(
                f"{self.ticker}: close price {self.close_kes} exceeds the "
                f"plausible maximum of {MAX_PLAUSIBLE_PRICE_KES} KES -- "
                f"probably a parse error"
            )
        if self.volume is not None and self.volume < 0:
            raise ValidationError(
                f"{self.ticker}: volume cannot be negative, got {self.volume}"
            )
        if self.trade_date > datetime.now(UTC).date():
            raise ValidationError(
                f"{self.ticker}: trade date {self.trade_date} is in the future"
            )

    @property
    def key(self) -> tuple[str, date]:
        """Natural key. Upserts dedupe on this."""
        return (self.ticker, self.trade_date)


@dataclass(frozen=True)
class Listing:
    """A company listed on the exchange."""

    ticker: str
    name: str
    source: str = "afx"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError(f"{self.ticker}: company name is required")


@dataclass
class IngestionReport:
    """What one pipeline run did.

    Returned rather than logged so callers -- and tests -- can assert on it.
    A run that silently ingests nothing is a failure mode worth surfacing.
    """

    source: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    rejected: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def rejection_rate(self) -> float:
        total = self.fetched + len(self.rejected)
        return len(self.rejected) / total if total else 0.0

    def summary(self) -> dict:
        return {
            "source": self.source,
            "fetched": self.fetched,
            "inserted": self.inserted,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "rejected": len(self.rejected),
            "rejection_rate": round(self.rejection_rate, 4),
            "duration_seconds": round(self.duration_seconds, 2),
        }

    def __str__(self) -> str:
        parts = [
            f"{self.source}: {self.fetched} fetched",
            f"{self.inserted} new",
            f"{self.updated} updated",
            f"{self.unchanged} unchanged",
        ]
        if self.rejected:
            parts.append(f"{len(self.rejected)} REJECTED")
        return ", ".join(parts)
