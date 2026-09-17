"""Fetch the current NSE listing and store it.

    python run_ingestion.py            # fetch and persist
    python run_ingestion.py --status   # show what is already stored

Safe to re-run: storage is keyed on (ticker, trade_date) and upserts, so a
second run the same day reports everything unchanged rather than duplicating.

The source asks for a 60-second crawl delay in robots.txt, which this honours.
A full run costs it exactly one request, because the listing page carries every
ticker's price.
"""

import argparse
import logging
import sys

from src.ingestion import MarketDataStore, run_daily_ingestion
from src.ingestion.pipeline import IngestionAborted

STORE_PATH = "data/market_data.db"


def show_status() -> None:
    store = MarketDataStore(STORE_PATH)
    coverage = store.coverage()

    if coverage.empty:
        print("Store is empty. Run `python run_ingestion.py` to populate it.")
        return

    print(f"Store:          {STORE_PATH}")
    print(f"Tickers known:  {len(store.known_tickers())}")
    print(f"Price rows:     {store.row_count():,}")
    print(f"Latest date:    {store.latest_trade_date()}")
    print(f"\nCoverage (top 10 of {len(coverage)}):")
    print(coverage.head(10).to_string(index=False))

    thin = coverage[coverage["observations"] < 30]
    if not thin.empty:
        print(
            f"\n{len(thin)} tickers have under 30 observations and are not yet "
            f"usable for covariance estimation."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show stored coverage")
    parser.add_argument("--export", metavar="PATH", help="write the store to CSV")
    parser.add_argument("--load", metavar="PATH", help="load a CSV snapshot")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.status:
        show_status()
        return 0

    if args.export:
        count = MarketDataStore(STORE_PATH).export_csv(args.export)
        print(f"Exported {count:,} rows to {args.export}")
        return 0

    if args.load:
        report = MarketDataStore(STORE_PATH).import_csv(args.load)
        print(f"Loaded {args.load}: {report}")
        return 0

    print("Fetching NSE listing (one request, honouring the 60s crawl delay)...")
    try:
        report = run_daily_ingestion(store_path=STORE_PATH)
    except IngestionAborted as exc:
        print(f"\nAborted without writing: {exc}", file=sys.stderr)
        return 1

    print(f"\n{report}")
    print(f"\nStore now holds {MarketDataStore(STORE_PATH).row_count():,} price rows.")

    if report.rejected:
        print(f"\n{len(report.rejected)} rows rejected:")
        for message in report.rejected[:5]:
            print(f"  - {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
