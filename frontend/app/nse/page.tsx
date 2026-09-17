"use client";

import { useEffect, useState } from "react";
import BarChart from "@/components/charts/BarChart";
import StatTile from "@/components/StatTile";
import TableView from "@/components/TableView";
import { api, type NseResponse } from "@/lib/api";
import { money } from "@/lib/format";

/**
 * Public page: market data, not user data, so it reads without signing in.
 */
export default function NsePage() {
  const [data, setData] = useState<NseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .nse()
      .then(setData)
      .catch((exc) =>
        setError(exc instanceof Error ? exc.message : "Could not load NSE data"),
      );
  }, []);

  return (
    <>
      <div className="page-head">
        <h1>Nairobi Securities Exchange</h1>
        <p>
          Yahoo Finance has no NSE coverage under any ticker convention, and the
          exchange sells its historical data rather than publishing it. This is
          scraped from a free aggregator, one request per day, honouring the
          60-second crawl delay its robots.txt asks for.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="tiles" style={{ marginBottom: 20 }}>
            <StatTile label="Companies listed" value={String(data.tickers_known)} />
            <StatTile label="Price rows stored" value={String(data.price_rows)} />
            <StatTile
              label="Latest session"
              value={data.latest_trade_date ?? "—"}
            />
            <StatTile
              label="Usable for covariance"
              value={String(data.usable_for_covariance)}
              sub="needs 30+ observations"
            />
          </div>

          <div className="notice" style={{ marginBottom: 20 }}>
            <strong>The store accumulates forward.</strong> No free archive of
            deep NSE history exists, so each run adds a session rather than
            backfilling years. Until coverage builds, NSE-based backtests are
            not meaningful &mdash; and the page says so rather than showing a
            chart built on one day.
          </div>

          {data.listings.length > 0 && (
            <div className="card">
              <div className="card-head">
                <h2>Latest close, by price</h2>
                <span className="small muted">{data.latest_trade_date}</span>
              </div>
              <p className="card-note">
                Share price in shillings. Price level says nothing about size
                &mdash; it is simply what the exchange last printed.
              </p>
              <BarChart
                bars={data.listings
                  .slice(0, 15)
                  .map((l) => ({ label: l.ticker, value: l.close_kes }))}
                format={(v) => money(v, 2)}
              />
              <TableView
                caption={`All ${data.listings.length} counters with a close on ${data.latest_trade_date}`}
                columns={["Ticker", "Close (KES)"]}
                rows={data.listings.map((l) => [l.ticker, money(l.close_kes, 2)])}
              />
            </div>
          )}
        </>
      )}

      {!data && !error && <div className="notice">Loading NSE coverage…</div>}
    </>
  );
}
