"""Tests for the NSE ingestion pipeline.

Every test runs against recorded HTML fixtures. Nothing here touches the
network, so the suite is deterministic and does not hammer a free service that
asks for a 60-second crawl delay.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.models import (
    IngestionReport,
    Listing,
    PriceBar,
    ValidationError,
    normalise_ticker,
    parse_number,
)
from src.ingestion.nse import (
    RateLimiter,
    SchemaChangedError,
    parse_listing_page,
    parse_ticker_history,
)
from src.ingestion.pipeline import IngestionAborted, ingest_nse_snapshot
from src.ingestion.store import MarketDataStore

FIXTURES = Path(__file__).parent / "fixtures"
TRADE_DATE = date(2026, 9, 16)


@pytest.fixture
def listing_html() -> str:
    return (FIXTURES / "afx_nse_listing.html").read_text()


@pytest.fixture
def ticker_html() -> str:
    return (FIXTURES / "afx_nse_ticker_scom.html").read_text()


@pytest.fixture
def store(tmp_path) -> MarketDataStore:
    return MarketDataStore(tmp_path / "test_market.db")


# ---------------------------------------------------------------------------
# Parsing scraped numbers
# ---------------------------------------------------------------------------
class TestParseNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("35.10", 35.10),
            ("1,234.50", 1234.50),  # thousands separator
            ("+0.65", 0.65),  # explicit sign
            ("-1.40", -1.40),
            ("(1.25)", -1.25),  # accounting negative
            ("-3.84%", -3.84),  # percent suffix
            ("10.8M", 10_800_000.0),  # volume shorthand
            ("1.2K", 1_200.0),
            ("3.4B", 3_400_000_000.0),
            ("  42.00  ", 42.0),  # whitespace
        ],
    )
    def test_parses_the_shapes_these_tables_contain(self, raw, expected):
        assert parse_number(raw, "field") == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "-", "--", "—", "N/A", "n/a", None])
    def test_treats_blanks_as_missing_rather_than_zero(self, raw):
        """A suspended counter has no price. Zero would be a lie."""
        assert parse_number(raw, "field") is None

    @pytest.mark.parametrize("raw", ["abc", "12.3.4", "$", "1.2.3", "12-34"])
    def test_rejects_junk_loudly(self, raw):
        with pytest.raises(ValidationError, match="could not parse"):
            parse_number(raw, "field")

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "Infinity"])
    def test_rejects_non_finite_values(self, raw):
        """float('nan') succeeds silently and poisons every downstream number."""
        with pytest.raises(ValidationError):
            parse_number(raw, "field")


class TestNormaliseTicker:
    def test_uppercases_and_strips(self):
        assert normalise_ticker("  scom  ") == "SCOM"

    @pytest.mark.parametrize("raw", ["C&G", "BAT", "SCOM", "E.A.C"])
    def test_accepts_real_nse_symbols(self, raw):
        assert normalise_ticker(raw) == raw.upper()

    @pytest.mark.parametrize("raw", ["", "  ", "1ABC", "way-too-long-symbol", "<b>"])
    def test_rejects_malformed_symbols(self, raw):
        with pytest.raises(ValidationError):
            normalise_ticker(raw)


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------
class TestPriceBar:
    def test_accepts_a_sane_bar(self):
        bar = PriceBar("SCOM", TRADE_DATE, 35.10, volume=10_775_518)
        assert bar.key == ("SCOM", TRADE_DATE)

    @pytest.mark.parametrize("price", [0.0, -1.0, -0.01])
    def test_rejects_non_positive_prices(self, price):
        with pytest.raises(ValidationError, match="must be positive"):
            PriceBar("SCOM", TRADE_DATE, price)

    def test_rejects_an_implausibly_large_price(self):
        """A market cap parsed into the price column looks like this."""
        with pytest.raises(ValidationError, match="plausible maximum"):
            PriceBar("SCOM", TRADE_DATE, 5_000_000.0)

    def test_rejects_negative_volume(self):
        with pytest.raises(ValidationError, match="volume cannot be negative"):
            PriceBar("SCOM", TRADE_DATE, 35.0, volume=-1)

    def test_rejects_a_future_trade_date(self):
        """A clock skew or a misparsed date must not enter the store."""
        tomorrow = datetime.now(UTC).date() + timedelta(days=2)
        with pytest.raises(ValidationError, match="in the future"):
            PriceBar("SCOM", tomorrow, 35.0)

    def test_missing_name_is_rejected_for_listings(self):
        with pytest.raises(ValidationError, match="name is required"):
            Listing(ticker="SCOM", name="   ")


# ---------------------------------------------------------------------------
# Parsing the real pages
# ---------------------------------------------------------------------------
class TestParseListingPage:
    def test_extracts_the_whole_exchange(self, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        assert len(snapshot.listings) == 71
        assert snapshot.rejected == []

    def test_extracts_known_companies(self, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        by_ticker = {listing.ticker: listing.name for listing in snapshot.listings}

        assert "SCOM" in by_ticker
        assert "Safaricom" in by_ticker["SCOM"]
        assert by_ticker["ABSA"] == "Absa Bank Kenya Plc"

    def test_prices_are_plausible_kenyan_shilling_values(self, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        for bar in snapshot.bars:
            assert 0 < bar.close_kes < 100_000, f"{bar.ticker} at {bar.close_kes}"

    def test_records_a_listing_even_when_it_has_no_price(self, listing_html):
        """Suspended counters still exist; they just have no bar that day."""
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        assert len(snapshot.listings) >= len(snapshot.bars)

    def test_applies_the_given_trade_date(self, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        assert all(bar.trade_date == TRADE_DATE for bar in snapshot.bars)

    def test_finds_the_table_by_headers_not_position(self, listing_html):
        """Injecting a decorative table must not shift the parser onto it."""
        decoy = "<table><tr><th>Ad</th></tr><tr><td>buy now</td></tr></table>"
        modified = listing_html.replace("<body>", "<body>" + decoy, 1)

        snapshot = parse_listing_page(modified, trade_date=TRADE_DATE)
        assert len(snapshot.listings) == 71

    def test_raises_when_the_layout_changes(self):
        """Silence after a redesign is the dangerous failure. Fail loudly."""
        with pytest.raises(SchemaChangedError, match="expected NSE listing columns"):
            parse_listing_page("<html><body><p>Site under maintenance</p></body></html>")

    def test_raises_on_an_empty_document(self):
        with pytest.raises(SchemaChangedError):
            parse_listing_page("")

    def test_one_bad_row_does_not_sink_the_batch(self, listing_html):
        # The source writes unclosed <td> tags, so match the opening tag.
        corrupted = listing_html.replace("<td>31.25<", "<td>NOT_A_PRICE<", 1)
        snapshot = parse_listing_page(corrupted, trade_date=TRADE_DATE)

        assert len(snapshot.rejected) == 1
        assert len(snapshot.listings) == 70


class TestParseTickerHistory:
    def test_extracts_recent_sessions(self, ticker_html):
        bars = parse_ticker_history(ticker_html, "SCOM")
        assert len(bars) == 10
        assert all(bar.ticker == "SCOM" for bar in bars)

    def test_dates_are_parsed_and_distinct(self, ticker_html):
        bars = parse_ticker_history(ticker_html, "SCOM")
        dates = [bar.trade_date for bar in bars]
        assert len(set(dates)) == len(dates)
        assert all(isinstance(d, date) for d in dates)

    def test_volume_is_captured(self, ticker_html):
        bars = parse_ticker_history(ticker_html, "SCOM")
        assert bars[0].volume == pytest.approx(10_775_518)

    def test_returns_empty_when_there_is_no_history_table(self):
        assert parse_ticker_history("<html><body></body></html>", "SCOM") == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class TestRateLimiter:
    def test_first_request_is_not_delayed(self):
        assert RateLimiter(min_interval_seconds=10).wait() == 0.0

    def test_second_request_waits_for_the_interval(self, monkeypatch):
        slept = []
        monkeypatch.setattr("src.ingestion.nse.time.sleep", slept.append)

        # wait() calls monotonic() once on the first call (to set the mark)
        # and twice on each later call (measure elapsed, then re-mark).
        clock = iter([0.0, 0.5, 0.5])
        monkeypatch.setattr("src.ingestion.nse.time.monotonic", lambda: next(clock))

        limiter = RateLimiter(min_interval_seconds=60.0)
        limiter.wait()
        limiter.wait()

        assert slept == [pytest.approx(59.5)]

    def test_no_wait_once_the_interval_has_already_passed(self, monkeypatch):
        slept = []
        monkeypatch.setattr("src.ingestion.nse.time.sleep", slept.append)

        clock = iter([0.0, 120.0, 120.0])
        monkeypatch.setattr("src.ingestion.nse.time.monotonic", lambda: next(clock))

        limiter = RateLimiter(min_interval_seconds=60.0)
        limiter.wait()
        assert limiter.wait() == 0.0
        assert slept == []


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class TestMarketDataStore:
    def test_inserts_new_bars(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        report = store.upsert_bars(snapshot.bars)

        assert report.inserted == len(snapshot.bars)
        assert report.updated == 0
        assert store.row_count() == len(snapshot.bars)

    def test_reingesting_identical_data_changes_nothing(self, store, listing_html):
        """Idempotency. A scheduled job gets re-run; that must be a no-op."""
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)

        store.upsert_bars(snapshot.bars)
        before = store.row_count()
        second = store.upsert_bars(snapshot.bars)

        assert second.inserted == 0
        assert second.updated == 0
        assert second.unchanged == len(snapshot.bars)
        assert store.row_count() == before

    def test_running_five_times_equals_running_once(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        for _ in range(5):
            store.upsert_bars(snapshot.bars)
        assert store.row_count() == len(snapshot.bars)

    def test_a_revised_price_updates_rather_than_duplicates(self, store):
        """Exchanges do restate prices. The row must change, not multiply."""
        original = PriceBar("SCOM", TRADE_DATE, 35.10, volume=1000)
        store.upsert_bars([original])

        revised = PriceBar("SCOM", TRADE_DATE, 35.50, volume=1000)
        report = store.upsert_bars([revised])

        assert report.updated == 1
        assert report.inserted == 0
        assert store.row_count() == 1
        assert store.load_prices()["SCOM"].iloc[0] == pytest.approx(35.50)

    def test_the_same_ticker_on_different_days_is_two_rows(self, store):
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
            ]
        )
        assert store.row_count() == 2

    def test_load_prices_returns_a_date_indexed_wide_frame(self, store):
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("EQTY", date(2026, 9, 15), 50.00),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
                PriceBar("EQTY", date(2026, 9, 16), 51.00),
            ]
        )
        frame = store.load_prices()

        assert list(frame.columns) == ["EQTY", "SCOM"]
        assert len(frame) == 2
        assert frame.index.is_monotonic_increasing

    def test_load_prices_filters_by_ticker_and_date(self, store):
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("EQTY", date(2026, 9, 15), 50.00),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
            ]
        )
        assert list(store.load_prices(tickers=["SCOM"]).columns) == ["SCOM"]
        assert len(store.load_prices(start_date=date(2026, 9, 16))) == 1

    def test_load_prices_can_drop_thinly_covered_tickers(self, store):
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
                PriceBar("THIN", date(2026, 9, 16), 10.00),
            ]
        )
        frame = store.load_prices(min_observations=2)
        assert list(frame.columns) == ["SCOM"]

    def test_load_prices_on_an_empty_store_returns_an_empty_frame(self, store):
        assert store.load_prices().empty

    def test_listings_track_first_and_last_seen(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        store.upsert_listings(snapshot.listings)
        store.upsert_listings(snapshot.listings)

        assert len(store.known_tickers()) == 71
        assert "SCOM" in store.known_tickers()

    def test_coverage_reports_observations_per_ticker(self, store):
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
                PriceBar("EQTY", date(2026, 9, 16), 51.00),
            ]
        )
        coverage = store.coverage().set_index("ticker")

        assert coverage.loc["SCOM", "observations"] == 2
        assert coverage.loc["EQTY", "observations"] == 1

    def test_latest_trade_date(self, store):
        assert store.latest_trade_date() is None
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
            ]
        )
        assert store.latest_trade_date() == date(2026, 9, 16)

    def test_empty_upsert_is_harmless(self, store):
        report = store.upsert_bars([])
        assert report.inserted == 0
        assert store.row_count() == 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_end_to_end_against_a_recorded_page(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        report = ingest_nse_snapshot(store, snapshot=snapshot)

        assert report.inserted == 71
        assert store.row_count() == 71
        assert len(store.known_tickers()) == 71

    def test_the_pipeline_is_idempotent(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        ingest_nse_snapshot(store, snapshot=snapshot)
        second = ingest_nse_snapshot(store, snapshot=snapshot)

        assert second.inserted == 0
        assert second.unchanged == 71

    def test_aborts_when_too_many_rows_fail_validation(self, store, listing_html):
        """A mass validation failure means the layout changed. Do not persist."""
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        snapshot.rejected.extend(f"synthetic failure {i}" for i in range(60))

        with pytest.raises(IngestionAborted, match="failed validation"):
            ingest_nse_snapshot(store, snapshot=snapshot)

        assert store.row_count() == 0, "nothing should have been written"

    def test_aborts_on_a_completely_empty_snapshot(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        snapshot.bars.clear()
        snapshot.rejected.clear()

        with pytest.raises(IngestionAborted, match="no rows at all"):
            ingest_nse_snapshot(store, snapshot=snapshot)

    def test_tolerates_a_rejection_rate_below_the_threshold(self, store, listing_html):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        snapshot.rejected.append("one bad row")

        report = ingest_nse_snapshot(store, snapshot=snapshot)
        assert report.inserted == 71
        assert len(report.rejected) == 1

    def test_requires_a_client_or_a_snapshot(self, store):
        with pytest.raises(ValueError, match="client or a snapshot"):
            ingest_nse_snapshot(store)


class TestIngestionReport:
    def test_summarises_a_run(self):
        report = IngestionReport(source="afx", fetched=71, inserted=70)
        report.rejected = ["bad row"]
        report.finished_at = report.started_at

        summary = report.summary()
        assert summary["source"] == "afx"
        assert summary["inserted"] == 70
        assert summary["rejection_rate"] == pytest.approx(1 / 72, abs=1e-4)

    def test_reads_clearly_when_printed(self):
        report = IngestionReport(source="afx", fetched=71, inserted=71)
        assert "71 fetched" in str(report)


class TestCsvRoundTrip:
    def test_export_then_import_preserves_the_data(self, store, listing_html, tmp_path):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        store.upsert_bars(snapshot.bars)

        csv_path = tmp_path / "snapshot.csv"
        exported = store.export_csv(csv_path)

        restored = MarketDataStore(tmp_path / "restored.db")
        report = restored.import_csv(csv_path)

        assert exported == len(snapshot.bars)
        assert report.inserted == len(snapshot.bars)

        pd.testing.assert_frame_equal(store.load_prices(), restored.load_prices())

    def test_import_is_idempotent(self, store, listing_html, tmp_path):
        snapshot = parse_listing_page(listing_html, trade_date=TRADE_DATE)
        store.upsert_bars(snapshot.bars)
        csv_path = tmp_path / "snapshot.csv"
        store.export_csv(csv_path)

        restored = MarketDataStore(tmp_path / "restored.db")
        restored.import_csv(csv_path)
        second = restored.import_csv(csv_path)

        assert second.inserted == 0
        assert second.unchanged == len(snapshot.bars)

    def test_export_is_sorted_for_clean_diffs(self, store, tmp_path):
        """A committed snapshot must diff as the rows it added, not a reshuffle."""
        store.upsert_bars(
            [
                PriceBar("SCOM", date(2026, 9, 16), 35.10),
                PriceBar("ABSA", date(2026, 9, 15), 31.25),
                PriceBar("SCOM", date(2026, 9, 15), 36.50),
            ]
        )
        csv_path = tmp_path / "snapshot.csv"
        store.export_csv(csv_path)

        frame = pd.read_csv(csv_path)
        assert list(frame["trade_date"]) == sorted(frame["trade_date"])

    def test_missing_volume_survives_the_round_trip(self, store, tmp_path):
        store.upsert_bars([PriceBar("ALP", TRADE_DATE, 1.02, volume=None)])
        csv_path = tmp_path / "snapshot.csv"
        store.export_csv(csv_path)

        restored = MarketDataStore(tmp_path / "restored.db")
        restored.import_csv(csv_path)

        assert restored.row_count() == 1
