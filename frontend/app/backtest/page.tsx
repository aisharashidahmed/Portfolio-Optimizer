"use client";

import { useState } from "react";
import AuthGate from "@/components/AuthGate";
import LineChart from "@/components/charts/LineChart";
import StatTile from "@/components/StatTile";
import TableView from "@/components/TableView";
import { api, type BacktestResponse } from "@/lib/api";
import { percent, ratio, seriesColour, strategyLabel } from "@/lib/format";

export default function BacktestPage() {
  return (
    <>
      <div className="page-head">
        <h1>Walk-forward backtest</h1>
        <p>
          Weights are re-fitted on a rolling window and applied forward, so the
          result reflects many independent decisions rather than one lucky
          period. Weights drift between rebalances and every trade is charged.
        </p>
      </div>
      <AuthGate>
        <Backtest />
      </AuthGate>
    </>
  );
}

function Backtest() {
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, PG");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2025-01-01");
  const [costBps, setCostBps] = useState(10);
  const [rebalance, setRebalance] = useState(21);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(
        await api.backtest({
          tickers: tickers.split(",").map((t) => t.trim()).filter(Boolean),
          start_date: startDate,
          end_date: endDate,
          algorithm: "hrp",
          max_weight: 0.35,
          training_days: 252,
          rebalance_every: rebalance,
          transaction_cost_bps: costBps,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Backtest failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="filters">
        <div className="field field-grow">
          <label htmlFor="tickers">Tickers</label>
          <input
            id="tickers"
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
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
        <div className="field">
          <label htmlFor="cost">Cost (bp)</label>
          <input
            id="cost"
            type="number"
            min={0}
            max={500}
            step={5}
            value={costBps}
            onChange={(e) => setCostBps(Number(e.target.value))}
            style={{ width: 78 }}
          />
        </div>
        <div className="field">
          <label htmlFor="rebalance">Rebalance (days)</label>
          <input
            id="rebalance"
            type="number"
            min={1}
            max={252}
            value={rebalance}
            onChange={(e) => setRebalance(Number(e.target.value))}
            style={{ width: 78 }}
          />
        </div>
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? "Running…" : "Run backtest"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {!data && !error && (
        <div className="notice">
          A decade of daily data across three strategies takes a few seconds.
          Try raising the cost to see which strategy degrades fastest.
        </div>
      )}

      {data && <BacktestResults data={data} busy={busy} />}
    </>
  );
}

function BacktestResults({
  data,
  busy,
}: {
  data: BacktestResponse;
  busy: boolean;
}) {
  const ok = data.strategies.filter((s) => !s.error && s.curve.length > 0);
  const failed = data.strategies.filter((s) => s.error);

  const series = ok.map((s) => ({
    key: s.strategy,
    label: strategyLabel(s.strategy),
    colour: seriesColour(s.strategy),
    points: s.curve,
  }));

  const best = ok.reduce<(typeof ok)[number] | null>(
    (a, b) =>
      (b.performance.sharpe_ratio ?? -Infinity) >
      (a?.performance.sharpe_ratio ?? -Infinity)
        ? b
        : a,
    null,
  );

  return (
    <div className={busy ? "loading" : undefined}>
      {best && (
        <div className="tiles" style={{ marginBottom: 20 }}>
          <StatTile
            label="Best Sharpe"
            value={ratio(best.performance.sharpe_ratio ?? 0)}
            sub={strategyLabel(best.strategy)}
          />
          <StatTile
            label="Its annual return"
            value={percent(best.performance.annualised_return ?? 0)}
            tone="up"
          />
          <StatTile
            label="Its worst drawdown"
            value={percent(best.performance.max_drawdown ?? 0)}
            tone="down"
          />
          <StatTile
            label="Trading cost"
            value={`${data.transaction_cost_bps} bp`}
            sub="charged per unit traded"
          />
        </div>
      )}

      {failed.length > 0 && (
        <div className="notice" style={{ marginBottom: 20 }}>
          {failed.map((s) => (
            <div key={s.strategy}>
              <strong>{strategyLabel(s.strategy)}</strong> could not run:{" "}
              {s.error}
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <h2>Growth of 1.00, net of costs</h2>
        </div>
        <p className="card-note">
          Out-of-sample only. Each strategy starts fully invested at the end of
          the first training window.
        </p>

        {/* Legend is always present for two or more series. */}
        <div className="legend">
          {series.map((s) => (
            <span key={s.key} className="legend-item">
              <span className="swatch" style={{ background: s.colour }} />
              {s.label}
            </span>
          ))}
        </div>

        <LineChart
          series={series}
          height={300}
          baseline={1}
          format={(v) => v.toFixed(2)}
          yLabel="growth of 1.00"
        />

        <TableView
          caption="Out-of-sample performance, net of transaction costs"
          columns={[
            "Strategy",
            "Ann. return",
            "Ann. vol",
            "Sharpe",
            "Sortino",
            "Max drawdown",
            "Avg turnover",
            "Cost drag",
          ]}
          swatches={ok.map((s) => seriesColour(s.strategy))}
          rows={ok.map((s) => [
            strategyLabel(s.strategy),
            percent(s.performance.annualised_return ?? 0),
            percent(s.performance.annualised_volatility ?? 0),
            ratio(s.performance.sharpe_ratio ?? 0),
            ratio(s.performance.sortino_ratio ?? 0),
            percent(s.performance.max_drawdown ?? 0),
            percent(s.average_turnover),
            percent(s.total_cost_drag),
          ])}
        />
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Turnover is where the money goes</h2>
        </div>
        <p className="card-note">
          Mean-variance weights swing with small changes in estimated returns,
          so it trades hard. That turnover is the usual reason a strategy looks
          excellent gross and mediocre net.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Strategy</th>
                <th scope="col">Average turnover per rebalance</th>
                <th scope="col">Cumulative cost drag</th>
              </tr>
            </thead>
            <tbody>
              {ok.map((s) => (
                <tr key={s.strategy}>
                  <td>
                    <span
                      className="swatch"
                      style={{ background: seriesColour(s.strategy) }}
                    />
                    {strategyLabel(s.strategy)}
                  </td>
                  <td>{percent(s.average_turnover)}</td>
                  <td>{percent(s.total_cost_drag)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
