"""Ingestion orchestration.

One entry point that fetches, validates, stores, and reports. Designed to be
run on a schedule and to be safe to re-run: every step is idempotent, and the
report says exactly what changed so a no-op run is visibly a no-op.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from src.ingestion.models import IngestionReport
from src.ingestion.nse import NSEClient, NSESnapshot, SchemaChangedError
from src.ingestion.store import MarketDataStore

logger = logging.getLogger(__name__)

# A run where most rows fail validation means the source changed shape. Better
# to fail the run than to quietly persist a fraction of the universe.
MAX_ACCEPTABLE_REJECTION_RATE = 0.20


class IngestionAborted(RuntimeError):
    """The run was stopped because the data looked wrong."""


def ingest_nse_snapshot(
    store: MarketDataStore,
    client: NSEClient | None = None,
    trade_date: date | None = None,
    snapshot: NSESnapshot | None = None,
) -> IngestionReport:
    """Fetch the current NSE listing and persist it.

    ``snapshot`` can be supplied directly to run the pipeline against recorded
    data, which is how the tests exercise it without network access.
    """
    started = datetime.now(UTC)

    if snapshot is None:
        if client is None:
            raise ValueError("Provide either a client or a snapshot")
        snapshot = client.fetch_snapshot(trade_date=trade_date)

    total = len(snapshot.bars) + len(snapshot.rejected)
    if total == 0:
        raise IngestionAborted(
            "The source returned no rows at all. Treating as a failure rather "
            "than an empty trading day."
        )

    rejection_rate = len(snapshot.rejected) / total
    if rejection_rate > MAX_ACCEPTABLE_REJECTION_RATE:
        raise IngestionAborted(
            f"{rejection_rate:.0%} of rows failed validation "
            f"({len(snapshot.rejected)}/{total}), above the "
            f"{MAX_ACCEPTABLE_REJECTION_RATE:.0%} threshold. The source layout "
            f"has probably changed. First few: {snapshot.rejected[:3]}"
        )

    store.upsert_listings(snapshot.listings)
    report = store.upsert_bars(snapshot.bars)

    report.rejected = list(snapshot.rejected)
    report.started_at = started
    report.finished_at = datetime.now(UTC)

    logger.info("NSE ingestion complete: %s", report)
    return report


def run_daily_ingestion(
    store_path: str = "data/market_data.db",
    trade_date: date | None = None,
) -> IngestionReport:
    """Production entry point: one polite request, persisted idempotently."""
    store = MarketDataStore(store_path)

    with NSEClient() as client:
        try:
            return ingest_nse_snapshot(store, client=client, trade_date=trade_date)
        except SchemaChangedError:
            logger.exception("NSE source layout changed; parser needs updating")
            raise


__all__ = [
    "MAX_ACCEPTABLE_REJECTION_RATE",
    "IngestionAborted",
    "ingest_nse_snapshot",
    "run_daily_ingestion",
]
