"""Nairobi Securities Exchange price ingestion.

Yahoo Finance has no NSE coverage (verified across every ticker-suffix
convention), and the exchange sells its own historical data rather than
publishing it. The free source used here is AFX, which publishes a clean HTML
table of every NSE listing with current price and volume.

Two constraints shape this module:

**The source asks for a 60-second crawl delay** in its robots.txt, and that is
honoured in :class:`RateLimiter`. It is also why the listing page is preferred
over per-ticker pages: one request returns all ~70 prices, so a daily run costs
the source a single hit.

**Only recent history is available.** The listing gives today; per-ticker pages
give roughly ten sessions. There is no free deep archive, so the store
accumulates forward -- each run adds a day. That is a real constraint, stated
plainly, not a limitation to hide.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.models import (
    IngestionReport,
    Listing,
    PriceBar,
    ValidationError,
    normalise_ticker,
    parse_number,
)

logger = logging.getLogger(__name__)

AFX_BASE_URL = "https://afx.kwayisi.org/nse/"
SOURCE_NAME = "afx"

# afx.kwayisi.org/robots.txt specifies Crawl-delay: 60.
CRAWL_DELAY_SECONDS = 60.0

# Identify the client honestly so the operator can block or contact us.
USER_AGENT = (
    "portfolio-optimizer/0.1 (educational project; "
    "https://github.com/aisharashidahmed/Portfolio-Optimizer)"
)

# The listing table has these columns, in this order.
EXPECTED_HEADERS = ["ticker", "name", "volume", "price", "change"]


class IngestionError(RuntimeError):
    """The source could not be fetched or understood."""


class SchemaChangedError(IngestionError):
    """The page no longer has the structure this parser expects.

    Raised loudly and early. A scraper that silently returns zero rows after a
    site redesign is far more dangerous than one that crashes, because the
    pipeline reports success and the data quietly goes stale.
    """


class RateLimiter:
    """Enforces a minimum interval between requests to one host."""

    def __init__(self, min_interval_seconds: float = CRAWL_DELAY_SECONDS):
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at: float | None = None

    def wait(self) -> float:
        """Block until the next request is allowed. Returns seconds slept."""
        if self._last_request_at is None:
            self._last_request_at = time.monotonic()
            return 0.0

        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed

        if remaining > 0:
            logger.debug("Rate limit: sleeping %.1fs", remaining)
            time.sleep(remaining)
            slept = remaining
        else:
            slept = 0.0

        self._last_request_at = time.monotonic()
        return slept


@dataclass
class NSESnapshot:
    """One fetch of the exchange listing."""

    listings: list[Listing]
    bars: list[PriceBar]
    rejected: list[str]
    fetched_at: datetime
    trade_date: date


def parse_listing_page(html: str, trade_date: date | None = None) -> NSESnapshot:
    """Parse the AFX NSE listing table.

    Pure function over HTML so it can be tested against recorded fixtures with
    no network access.
    """
    soup = BeautifulSoup(html, "lxml")
    trade_date = trade_date or datetime.now(UTC).date()

    table = _find_listing_table(soup)
    if table is None:
        raise SchemaChangedError(
            "No table with the expected NSE listing columns "
            f"({EXPECTED_HEADERS}) was found. The source layout has probably "
            "changed and the parser needs updating."
        )

    listings: list[Listing] = []
    bars: list[PriceBar] = []
    rejected: list[str] = []
    fetched_at = datetime.now(UTC)

    rows = table.find_all("tr")
    for row in rows[1:]:  # skip the header
        cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue

        raw_ticker, name, raw_volume, raw_price = cells[0], cells[1], cells[2], cells[3]

        try:
            ticker = normalise_ticker(raw_ticker)
            price = parse_number(raw_price, f"{raw_ticker}.price")

            if price is None:
                # A suspended or never-traded counter. Not an error: record the
                # listing, skip the bar.
                listings.append(Listing(ticker=ticker, name=name, source=SOURCE_NAME))
                continue

            listings.append(Listing(ticker=ticker, name=name, source=SOURCE_NAME))
            bars.append(
                PriceBar(
                    ticker=ticker,
                    trade_date=trade_date,
                    close_kes=price,
                    volume=parse_number(raw_volume, f"{raw_ticker}.volume"),
                    source=SOURCE_NAME,
                    fetched_at=fetched_at,
                )
            )
        except ValidationError as exc:
            rejected.append(str(exc))
            logger.warning("Rejected row %s: %s", raw_ticker, exc)

    if not listings:
        raise SchemaChangedError(
            "The listing table was found but produced no usable rows. "
            "The column layout has probably changed."
        )

    return NSESnapshot(
        listings=listings,
        bars=bars,
        rejected=rejected,
        fetched_at=fetched_at,
        trade_date=trade_date,
    )


def _find_listing_table(soup: BeautifulSoup):
    """Locate the listing table by its headers rather than by position.

    Indexing into ``find_all('table')[3]`` would break the moment the page
    gains a banner. Matching on content is what makes the parser survive
    cosmetic redesigns -- and fail loudly on structural ones.
    """
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]
        if all(expected in headers for expected in ("ticker", "name", "price")):
            return table

    return None


def parse_ticker_history(html: str, ticker: str) -> list[PriceBar]:
    """Parse the recent-history table from a per-ticker page.

    Gives roughly ten sessions -- useful for backfilling a gap after the
    pipeline has not run for a few days.
    """
    soup = BeautifulSoup(html, "lxml")
    symbol = normalise_ticker(ticker)
    fetched_at = datetime.now(UTC)

    bars: list[PriceBar] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]
        if "date" not in headers or "close" not in headers:
            continue

        date_index = headers.index("date")
        close_index = headers.index("close")
        volume_index = headers.index("volume") if "volume" in headers else None

        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) <= max(date_index, close_index):
                continue

            try:
                trade_date = datetime.strptime(cells[date_index], "%Y-%m-%d").date()
                close = parse_number(cells[close_index], f"{symbol}.close")
                if close is None:
                    continue

                volume = (
                    parse_number(cells[volume_index], f"{symbol}.volume")
                    if volume_index is not None and len(cells) > volume_index
                    else None
                )

                bars.append(
                    PriceBar(
                        ticker=symbol,
                        trade_date=trade_date,
                        close_kes=close,
                        volume=volume,
                        source=SOURCE_NAME,
                        fetched_at=fetched_at,
                    )
                )
            except (ValidationError, ValueError) as exc:
                logger.warning("Skipping history row for %s: %s", symbol, exc)

        break

    return bars


class NSEClient:
    """Fetches NSE data, politely."""

    def __init__(
        self,
        base_url: str = AFX_BASE_URL,
        rate_limiter: RateLimiter | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.rate_limiter = rate_limiter or RateLimiter()
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> str:
        self.rate_limiter.wait()

        response = self._client.get(url)
        response.raise_for_status()

        if not response.text.strip():
            raise IngestionError(f"{url} returned an empty body")

        return response.text

    def fetch_snapshot(self, trade_date: date | None = None) -> NSESnapshot:
        """Fetch every listed ticker's current price in a single request."""
        return parse_listing_page(self._get(self.base_url), trade_date=trade_date)

    def fetch_ticker_history(self, ticker: str) -> list[PriceBar]:
        """Fetch the recent sessions for one ticker. Costs one crawl-delay."""
        symbol = normalise_ticker(ticker)
        url = f"{self.base_url}{symbol.lower()}.html"
        return parse_ticker_history(self._get(url), symbol)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NSEClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def report_for(snapshot: NSESnapshot) -> IngestionReport:
    """Build a report describing what a snapshot contained."""
    report = IngestionReport(source=SOURCE_NAME)
    report.fetched = len(snapshot.bars)
    report.rejected = list(snapshot.rejected)
    report.finished_at = datetime.now(UTC)
    return report


__all__ = [
    "AFX_BASE_URL",
    "CRAWL_DELAY_SECONDS",
    "IngestionError",
    "NSEClient",
    "NSESnapshot",
    "RateLimiter",
    "SchemaChangedError",
    "parse_listing_page",
    "parse_ticker_history",
    "report_for",
]
