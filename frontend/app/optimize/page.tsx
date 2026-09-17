"use client";

import { useState } from "react";
import AuthGate from "@/components/AuthGate";
import BarChart from "@/components/charts/BarChart";
import StatTile from "@/components/StatTile";
import TableView from "@/components/TableView";
import { api, type OptimizeResponse } from "@/lib/api";
import { percent, ratio } from "@/lib/format";

export default function OptimizePage() {
  return (
    <>
      <div className="page-head">
        <h1>Build a portfolio</h1>
        <p>
          Three allocation strategies behind one interface. Equal weight is the
          benchmark, not a placeholder &mdash; it is hard to beat out of sample
          once estimation error is counted.
        </p>
      </div>
      <AuthGate>
        <Optimizer />
      </AuthGate>
    </>
  );
}

function Optimizer() {
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL, JPM, JNJ");
  const [algorithm, setAlgorithm] = useState("hrp");
  const [estimator, setEstimator] = useState("ledoit_wolf");
  const [maxWeight, setMaxWeight] = useState(0.35);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2025-01-01");
  const [data, setData] = useState<OptimizeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setData(
        await api.optimize({
          tickers: tickers.split(",").map((t) => t.trim()).filter(Boolean),
          start_date: startDate,
          end_date: endDate,
          algorithm,
          max_weight: maxWeight,
          covariance_estimator: estimator,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Optimization failed");
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
          <label htmlFor="algorithm">Strategy</label>
          <select
            id="algorithm"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            <option value="hrp">Hierarchical Risk Parity</option>
            <option value="mvo">Mean-variance</option>
            <option value="equal_weight">Equal weight</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="estimator">Covariance</label>
          <select
            id="estimator"
            value={estimator}
            onChange={(e) => setEstimator(e.target.value)}
          >
            <option value="ledoit_wolf">Ledoit-Wolf shrinkage</option>
            <option value="sample">Sample</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="cap">Max weight</label>
          <input
            id="cap"
            type="number"
            min={0.05}
            max={1}
            step={0.05}
            value={maxWeight}
            onChange={(e) => setMaxWeight(Number(e.target.value))}
            style={{ width: 84 }}
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
          {busy ? "Running…" : "Optimize"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {!data && !error && (
        <div className="notice">
          Pick a universe and run it. Weights come back with the covariance
          diagnostics used to produce them.
        </div>
      )}

      {data && <OptimizeResults data={data} busy={busy} />}
    </>
  );
}

function OptimizeResults({
  data,
  busy,
}: {
  data: OptimizeResponse;
  busy: boolean;
}) {
  const diagnostics = data.covariance_diagnostics;
  const bars = data.tickers
    .map((ticker, i) => ({ label: ticker, value: data.weights[i] }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className={busy ? "loading" : undefined}>
      <div className="tiles" style={{ marginBottom: 20 }}>
        <StatTile
          label="Expected annual return"
          value={percent(data.metrics.expected_annual_return)}
          sub="from historical means"
        />
        <StatTile
          label="Annual volatility"
          value={percent(data.metrics.annual_volatility)}
        />
        <StatTile
          label="Sharpe ratio"
          value={ratio(data.metrics.sharpe_ratio)}
          sub="in sample"
        />
        <StatTile
          label="Shrinkage applied"
          value={percent(diagnostics.shrinkage_intensity, 0)}
          sub={`${diagnostics.observations_per_asset.toFixed(0)} obs per asset`}
        />
      </div>

      {data.missing_tickers.length > 0 && (
        <div className="notice" style={{ marginBottom: 20 }}>
          No usable price history for: {data.missing_tickers.join(", ")}. They
          were dropped rather than failing the whole request.
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <h2>Allocation</h2>
          <span className="small muted">{data.algorithm.toUpperCase()}</span>
        </div>
        <p className="card-note">
          Long-only and fully invested. Every weight is labelled directly rather
          than left to the colour scale.
        </p>
        <BarChart bars={bars} format={(v) => percent(v, 1)} maxValue={1} />
        <TableView
          caption="Portfolio weights by ticker"
          columns={["Ticker", "Weight"]}
          rows={bars.map((b) => [b.label, percent(b.value, 2)])}
        />
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Covariance diagnostics</h2>
        </div>
        <p className="card-note">
          Estimation quality decides whether the allocation above means anything.
          A condition number above roughly 1,000 means the matrix is
          near-singular and mean-variance output is mostly noise.
        </p>
        <div className="table-wrap">
          <table>
            <tbody>
              <tr>
                <td>Estimator</td>
                <td>
                  {diagnostics.estimator === "ledoit_wolf"
                    ? "Ledoit-Wolf shrinkage"
                    : "Sample covariance"}
                </td>
              </tr>
              <tr>
                <td>Shrinkage intensity (δ)</td>
                <td>{diagnostics.shrinkage_intensity.toFixed(3)}</td>
              </tr>
              <tr>
                <td>Condition number</td>
                <td>{diagnostics.condition_number.toFixed(1)}</td>
              </tr>
              <tr>
                <td>Observations</td>
                <td>{diagnostics.n_observations}</td>
              </tr>
              <tr>
                <td>Observations per asset</td>
                <td>{diagnostics.observations_per_asset.toFixed(1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
