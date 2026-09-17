"use client";

import { useState } from "react";
import AuthGate from "@/components/AuthGate";
import BarChart from "@/components/charts/BarChart";
import LineChart from "@/components/charts/LineChart";
import StatTile from "@/components/StatTile";
import TableView from "@/components/TableView";
import { api, type CurrencyResponse } from "@/lib/api";
import { percent, ratio, signedPercent } from "@/lib/format";

const DEFAULT_TICKERS = "SPY, AAPL, MSFT, JNJ, PG";

export default function CurrencyPage() {
  return (
    <>
      <div className="page-head">
        <h1>What a Kenyan investor actually earns on US assets</h1>
        <p>
          A Kenyan holding US assets does not earn the asset&apos;s return. They
          earn it compounded with the KES/USD move &mdash; and over the last
          decade that currency term was larger than the asset return for several
          holdings. This page separates the two.
        </p>
      </div>
      <AuthGate>
        <CurrencyAnalysis />
      </AuthGate>
    </>
  );
}

function CurrencyAnalysis() {
  const [tickers, setTickers] = useState(DEFAULT_TICKERS);
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2025-01-01");
  const [data, setData] = useState<CurrencyResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(
        await api.currency({
          tickers: tickers.split(",").map((t) => t.trim()).filter(Boolean),
          start_date: startDate,
          end_date: endDate,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {/* One filter row above everything it scopes. */}
      <div className="filters">
        <div className="field field-grow">
          <label htmlFor="tickers">US tickers</label>
          <input
            id="tickers"
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            placeholder="SPY, AAPL"
          />
        </div>
        <div className="field">
          <label htmlFor="start">From</label>
          <input
            id="start"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="end">To</label>
          <input
            id="end"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? "Running…" : "Analyse"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {!data && !error && (
        <div className="notice">
          Choose a window and run the analysis. Ten years of daily data takes a
          few seconds to fetch.
        </div>
      )}

      {data && <CurrencyResults data={data} busy={busy} />}
    </>
  );
}

function CurrencyResults({
  data,
  busy,
}: {
  data: CurrencyResponse;
  busy: boolean;
}) {
  const headline = data.assets[0];

  return (
    <div className={busy ? "loading" : undefined}>
      <div className="tiles" style={{ marginBottom: 20 }}>
        <StatTile
          label="KES per USD"
          value={ratio(data.fx.end_rate, 1)}
          sub={`from ${ratio(data.fx.start_rate, 1)}`}
        />
        <StatTile
          label="Shilling move"
          value={signedPercent(data.fx.total_depreciation)}
          sub="a weaker shilling lifts USD assets"
          tone={data.fx.total_depreciation > 0 ? "up" : "down"}
        />
        {headline && (
          <>
            <StatTile
              label={`${headline.ticker} in USD`}
              value={signedPercent(headline.asset_return_usd, 0)}
              sub="asset return alone"
            />
            <StatTile
              label={`${headline.ticker} in KES`}
              value={signedPercent(headline.total_return_kes, 0)}
              sub="what the investor actually got"
              tone="up"
            />
          </>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <h2>KES per USD</h2>
        </div>
        <p className="card-note">
          A rising line is a weakening shilling. Holding dollars alone returned{" "}
          {signedPercent(data.fx.total_depreciation)} over this window, before
          any asset appreciation.
        </p>
        <LineChart
          series={[
            {
              key: "fx",
              label: "KES/USD",
              colour: "var(--series-1)",
              points: data.fx.curve,
            },
          ]}
          format={(v) => v.toFixed(0)}
          yLabel="KES per USD"
        />
        <TableView
          caption="USD/KES exchange rate, sampled"
          columns={["Date", "KES per USD"]}
          rows={data.fx.curve
            .filter((_, i) => i % 20 === 0)
            .map((p) => [p.date, ratio(p.value, 2)])}
        />
      </div>

      <div className="card">
        <div className="card-head">
          <h2>How much of the risk is the currency?</h2>
        </div>
        <p className="card-note">
          Share of each asset&apos;s total variance attributable to the exchange
          rate, for an investor who spends shillings. The pattern is the
          interesting part: the safer the asset, the more the currency dominates.
        </p>
        <BarChart
          bars={data.assets
            .slice()
            .sort((a, b) => b.currency_share_of_variance - a.currency_share_of_variance)
            .map((a) => ({
              label: a.ticker,
              value: a.currency_share_of_variance,
            }))}
          format={(v) => percent(v, 1)}
          colour="var(--series-2)"
        />
        <TableView
          caption="Return and risk decomposition, asset currency versus shillings"
          columns={[
            "Ticker",
            "Return (USD)",
            "Return (KES)",
            "Vol (USD)",
            "Vol (KES)",
            "corr(asset, FX)",
            "FX share of variance",
          ]}
          rows={data.assets.map((a) => [
            a.ticker,
            signedPercent(a.asset_return_usd, 1),
            signedPercent(a.total_return_kes, 1),
            percent(a.asset_volatility),
            percent(a.total_volatility_kes),
            a.correlation.toFixed(3),
            percent(a.currency_share_of_variance),
          ])}
        />
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Return, split three ways</h2>
        </div>
        <p className="card-note">
          The asset return and the currency return compound rather than add, so
          the total is asset + currency + their interaction.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Ticker</th>
                <th scope="col">Asset (USD)</th>
                <th scope="col">Currency</th>
                <th scope="col">Interaction</th>
                <th scope="col">Total (KES)</th>
              </tr>
            </thead>
            <tbody>
              {data.assets.map((a) => (
                <tr key={a.ticker}>
                  <td>{a.ticker}</td>
                  <td>{signedPercent(a.asset_return_usd, 1)}</td>
                  <td>{signedPercent(a.currency_return, 1)}</td>
                  <td>{signedPercent(a.interaction, 1)}</td>
                  <td style={{ fontWeight: 600 }}>
                    {signedPercent(a.total_return_kes, 1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
