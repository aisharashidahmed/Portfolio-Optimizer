"""Local market-data warehouse.

Deliberately separate from the application database. ``src/database.py`` holds
users and optimization runs -- state the app owns. This holds ingested market
data, which is a cache of somebody else's facts: it can be dropped and rebuilt,
it has different backup requirements, and it is queried in bulk rather than by
primary key.

The property that matters is **idempotency**. A scheduled job will be re-run
after a failure, run twice by an impatient operator, or overlap with itself.
Running it five times must leave the store exactly as running it once did.
That is achieved with a natural key of (ticker, trade_date) and an upsert.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

from src.ingestion.models import IngestionReport, Listing, PriceBar

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("data/market_data.db")

metadata = MetaData()

price_bars = Table(
    "price_bars",
    metadata,
    Column("ticker", String(16), primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("close_kes", Float, nullable=False),
    Column("volume", Float),
    # Provenance: which source said so, and when we asked. Without these,
    # reconciling a disputed number later is guesswork.
    Column("source", String(32), nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Index("ix_price_bars_date", "trade_date"),
)

listings = Table(
    "listings",
    metadata,
    Column("ticker", String(16), primary_key=True),
    Column("name", String(256), nullable=False),
    Column("source", String(32), nullable=False),
    Column("first_seen", DateTime(timezone=True), nullable=False),
    Column("last_seen", DateTime(timezone=True), nullable=False),
)


class MarketDataStore:
    """Idempotent storage for ingested price data."""

    def __init__(
        self, path: Path | str = DEFAULT_STORE_PATH, engine: Engine | None = None
    ):
        if engine is not None:
            self.engine = engine
            self.path = None
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{self.path}")

        metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert_bars(self, bars: list[PriceBar]) -> IngestionReport:
        """Insert or update bars, reporting exactly what changed.

        Distinguishes updated from unchanged so a re-run is visibly a no-op
        rather than merely appearing to succeed. A run that reports
        ``0 new, 0 updated, 71 unchanged`` is proof of idempotency.
        """
        report = IngestionReport(source=bars[0].source if bars else "unknown")

        if not bars:
            report.finished_at = datetime.now(UTC)
            return report

        with self.engine.begin() as connection:
            existing = {
                (row.ticker, row.trade_date): (row.close_kes, row.volume)
                for row in connection.execute(
                    select(
                        price_bars.c.ticker,
                        price_bars.c.trade_date,
                        price_bars.c.close_kes,
                        price_bars.c.volume,
                    ).where(price_bars.c.trade_date.in_({bar.trade_date for bar in bars}))
                )
            }

            to_insert, to_update = [], []
            for bar in bars:
                row = {
                    "ticker": bar.ticker,
                    "trade_date": bar.trade_date,
                    "close_kes": bar.close_kes,
                    "volume": bar.volume,
                    "source": bar.source,
                    "fetched_at": bar.fetched_at,
                }

                previous = existing.get(bar.key)
                if previous is None:
                    to_insert.append(row)
                elif previous != (bar.close_kes, bar.volume):
                    to_update.append(row)
                    logger.info(
                        "Revised %s %s: %s -> %s",
                        bar.ticker,
                        bar.trade_date,
                        previous[0],
                        bar.close_kes,
                    )
                else:
                    report.unchanged += 1

            if to_insert:
                connection.execute(price_bars.insert(), to_insert)
            for row in to_update:
                connection.execute(
                    price_bars.update()
                    .where(
                        (price_bars.c.ticker == row["ticker"])
                        & (price_bars.c.trade_date == row["trade_date"])
                    )
                    .values(**row)
                )

            report.inserted = len(to_insert)
            report.updated = len(to_update)

        report.fetched = len(bars)
        report.finished_at = datetime.now(UTC)
        return report

    def upsert_listings(self, entries: list[Listing]) -> int:
        """Record the listed universe, tracking when each was first/last seen.

        ``first_seen`` and ``last_seen`` are how delistings become visible: a
        ticker whose last_seen stops advancing has left the exchange.
        """
        if not entries:
            return 0

        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            known = {row.ticker for row in connection.execute(select(listings.c.ticker))}

            new_rows = [
                {
                    "ticker": entry.ticker,
                    "name": entry.name,
                    "source": entry.source,
                    "first_seen": now,
                    "last_seen": now,
                }
                for entry in entries
                if entry.ticker not in known
            ]
            if new_rows:
                connection.execute(listings.insert(), new_rows)

            for entry in entries:
                if entry.ticker in known:
                    connection.execute(
                        listings.update()
                        .where(listings.c.ticker == entry.ticker)
                        .values(name=entry.name, last_seen=now)
                    )

        return len(entries)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def load_prices(
        self,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        min_observations: int = 0,
    ) -> pd.DataFrame:
        """Return a date-indexed, ticker-per-column price frame in KES.

        This is the shape the optimizers consume, so the warehouse hands back
        something usable rather than making every caller pivot it.
        """
        query = select(
            price_bars.c.trade_date, price_bars.c.ticker, price_bars.c.close_kes
        )
        if tickers:
            query = query.where(price_bars.c.ticker.in_([t.upper() for t in tickers]))
        if start_date:
            query = query.where(price_bars.c.trade_date >= start_date)
        if end_date:
            query = query.where(price_bars.c.trade_date <= end_date)

        with self.engine.connect() as connection:
            rows = connection.execute(query).fetchall()

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=["trade_date", "ticker", "close_kes"])
        wide = frame.pivot(index="trade_date", columns="ticker", values="close_kes")
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide.columns.name = None

        if min_observations > 0:
            keep = [c for c in wide.columns if wide[c].notna().sum() >= min_observations]
            wide = wide[keep]

        return wide

    # ------------------------------------------------------------------
    # Portable snapshots
    # ------------------------------------------------------------------
    def export_csv(self, path: Path | str) -> int:
        """Write the store to a long-format CSV.

        The CSV, not the SQLite file, is what gets committed. A binary database
        in version control produces unreadable diffs and merge conflicts on
        every run; a sorted text file diffs cleanly, so a day's ingestion shows
        up in review as the rows it actually added.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    price_bars.c.ticker,
                    price_bars.c.trade_date,
                    price_bars.c.close_kes,
                    price_bars.c.volume,
                    price_bars.c.source,
                ).order_by(price_bars.c.trade_date, price_bars.c.ticker)
            ).fetchall()

        frame = pd.DataFrame(
            rows, columns=["ticker", "trade_date", "close_kes", "volume", "source"]
        )
        frame.to_csv(path, index=False)
        return len(frame)

    def import_csv(self, path: Path | str) -> IngestionReport:
        """Load a snapshot written by :meth:`export_csv`.

        Goes through the same validation and upsert as a live fetch, so a
        seeded store is indistinguishable from an accumulated one.
        """
        frame = pd.read_csv(Path(path), parse_dates=["trade_date"])

        bars = [
            PriceBar(
                ticker=row.ticker,
                trade_date=row.trade_date.date(),
                close_kes=float(row.close_kes),
                volume=None if pd.isna(row.volume) else float(row.volume),
                source=row.source,
            )
            for row in frame.itertuples()
        ]
        return self.upsert_bars(bars)

    def known_tickers(self) -> list[str]:
        with self.engine.connect() as connection:
            return sorted(
                row.ticker for row in connection.execute(select(listings.c.ticker))
            )

    def coverage(self) -> pd.DataFrame:
        """Per-ticker data coverage. The first thing to check before trusting a backtest."""
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(price_bars.c.ticker, price_bars.c.trade_date)
            ).fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "observations", "first_date", "last_date"]
            )

        frame = pd.DataFrame(rows, columns=["ticker", "trade_date"])
        return (
            frame.groupby("ticker")["trade_date"]
            .agg(observations="count", first_date="min", last_date="max")
            .reset_index()
            .sort_values("observations", ascending=False)
        )

    def latest_trade_date(self) -> date | None:
        with self.engine.connect() as connection:
            result = connection.execute(
                select(price_bars.c.trade_date)
                .order_by(price_bars.c.trade_date.desc())
                .limit(1)
            ).fetchone()
        return result[0] if result else None

    def row_count(self) -> int:
        with self.engine.connect() as connection:
            return len(connection.execute(select(price_bars.c.ticker)).fetchall())
