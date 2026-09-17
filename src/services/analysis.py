"""Service layer for backtesting and currency analysis.

Same pattern as ``src.services.optimization``: the API and any worker call in
here, so there is one code path per capability and it is testable without a
web server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.algorithms.base import BaseOptimizer
from src.algorithms.equal_weight import EqualWeightOptimizer
from src.algorithms.hrp import HRPOptimizer
from src.algorithms.mvo import MVOOptimizer
from src.analytics.currency import summarise_currency_exposure
from src.backtesting.engine import run_walk_forward
from src.data_fetcher import prepare_market_data
from src.ingestion.fx import fetch_usd_kes

# Fixed identity -> colour mapping lives in the frontend; the order here just
# keeps responses stable so a chart's series never swap places between calls.
STRATEGY_ORDER = ["equal_weight", "hrp", "mvo"]

# Downsample long equity curves before sending them over the wire. A decade of
# daily data is ~2500 points per series; no screen resolves that, and the JSON
# gets large enough to be felt.
MAX_CURVE_POINTS = 400


class AnalysisError(ValueError):
    """The requested analysis could not be produced."""


def _build(name: str, risk_free_rate: float, max_weight: float) -> BaseOptimizer:
    if name == "equal_weight":
        return EqualWeightOptimizer()
    if name == "hrp":
        return HRPOptimizer()
    if name == "mvo":
        return MVOOptimizer(risk_free_rate=risk_free_rate, max_weight=max_weight)
    raise AnalysisError(f"Unknown strategy {name!r}")


def _downsample(series: pd.Series, max_points: int = MAX_CURVE_POINTS) -> pd.Series:
    """Thin a series for transport, always keeping the final point.

    The last point is the cumulative return the reader will quote, so it must
    survive exactly rather than being averaged away.
    """
    if len(series) <= max_points:
        return series

    # ceil, not floor: with 648 points and a 400 cap, floor division gives a
    # step of 1 and thins nothing.
    step = math.ceil(len(series) / max_points)
    thinned = series.iloc[::step]

    if thinned.index[-1] != series.index[-1]:
        # At the cap, replace the last strided point rather than appending,
        # so the result never exceeds max_points by one.
        if len(thinned) >= max_points:
            thinned = thinned.iloc[:-1]
        thinned = pd.concat([thinned, series.iloc[[-1]]])

    return thinned


@dataclass
class StrategyResult:
    strategy: str
    curve: list[dict[str, Any]]
    performance: dict[str, float]
    final_weights: dict[str, float]
    average_turnover: float
    total_cost_drag: float
    error: str | None = None


@dataclass
class BacktestComparison:
    tickers: list[str]
    start_date: str
    end_date: str
    transaction_cost_bps: float
    strategies: list[StrategyResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "transaction_cost_bps": self.transaction_cost_bps,
            "strategies": [
                {
                    "strategy": s.strategy,
                    "curve": s.curve,
                    "performance": s.performance,
                    "final_weights": s.final_weights,
                    "average_turnover": s.average_turnover,
                    "total_cost_drag": s.total_cost_drag,
                    "error": s.error,
                }
                for s in self.strategies
            ],
        }


def run_backtest_comparison(
    tickers: list[str],
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    training_days: int = 252,
    rebalance_every: int = 21,
    transaction_cost_bps: float = 10.0,
    max_weight: float = 0.35,
    strategies: list[str] | None = None,
) -> BacktestComparison:
    """Walk-forward every strategy over identical data."""
    requested = strategies or STRATEGY_ORDER

    market_data = prepare_market_data(tickers, start_date, end_date)
    daily_returns = market_data["daily_returns"]

    if len(daily_returns) <= training_days:
        raise AnalysisError(
            f"Need more than {training_days} observations to leave an "
            f"out-of-sample period; the requested window has {len(daily_returns)}. "
            f"Widen the date range or lower the training window."
        )

    comparison = BacktestComparison(
        tickers=list(daily_returns.columns),
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
    )

    for name in requested:
        try:
            result = run_walk_forward(
                daily_returns,
                _build(name, risk_free_rate, max_weight),
                training_days=training_days,
                rebalance_every=rebalance_every,
                transaction_cost_bps=transaction_cost_bps,
                risk_free_rate=risk_free_rate,
            )
        except Exception as exc:
            # One strategy failing (MVO can refuse a degenerate matrix) must not
            # deny the reader the others.
            comparison.strategies.append(
                StrategyResult(
                    strategy=name,
                    curve=[],
                    performance={},
                    final_weights={},
                    average_turnover=0.0,
                    total_cost_drag=0.0,
                    error=str(exc),
                )
            )
            continue

        equity = _downsample(result.equity_curve())
        final = result.weights_history.iloc[-1]

        comparison.strategies.append(
            StrategyResult(
                strategy=name,
                curve=[
                    {"date": index.strftime("%Y-%m-%d"), "value": float(value)}
                    for index, value in equity.items()
                ],
                performance={
                    k: float(v) for k, v in result.performance.to_dict().items()
                },
                final_weights={t: float(w) for t, w in final.items()},
                average_turnover=result.average_turnover,
                total_cost_drag=result.total_cost_drag,
            )
        )

    return comparison


def run_currency_analysis(
    tickers: list[str], start_date: str, end_date: str
) -> dict[str, Any]:
    """Split each asset's KES-denominated result into asset and currency."""
    market_data = prepare_market_data(tickers, start_date, end_date)
    usd_returns = market_data["daily_returns"]

    rate = fetch_usd_kes(start_date, end_date)

    table = summarise_currency_exposure(usd_returns, rate)
    if table.empty:
        raise AnalysisError("No overlapping data between the assets and USD/KES")

    rate_curve = _downsample(rate)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "fx": {
            "start_rate": float(rate.iloc[0]),
            "end_rate": float(rate.iloc[-1]),
            "total_depreciation": float(rate.iloc[-1] / rate.iloc[0] - 1.0),
            "curve": [
                {"date": index.strftime("%Y-%m-%d"), "value": float(value)}
                for index, value in rate_curve.items()
            ],
        },
        "assets": [
            {
                "ticker": ticker,
                **{k: float(v) for k, v in row.items()},
            }
            for ticker, row in table.iterrows()
        ],
    }
