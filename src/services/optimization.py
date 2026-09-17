"""Optimization orchestration.

The single place where "fetch data, pick an algorithm, run it, measure it"
happens. The API and the Celery worker both call into here, so there is exactly
one data path and it can be tested without Redis or Postgres running.
"""

from dataclasses import dataclass, field
from typing import Any

from src.algorithms.base import BaseOptimizer
from src.algorithms.equal_weight import EqualWeightOptimizer
from src.algorithms.hrp import HRPOptimizer
from src.algorithms.mvo import MVOOptimizer
from src.data_fetcher import prepare_market_data
from src.metrics import calculate_portfolio_metrics

ALGORITHMS = {"hrp", "mvo", "equal_weight"}


class UnknownAlgorithmError(ValueError):
    pass


class InsufficientDataError(ValueError):
    pass


@dataclass
class OptimizationResult:
    tickers: list[str]
    weights: list[float]
    algorithm: str
    metrics: dict[str, float]
    missing_tickers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "missing_tickers": self.missing_tickers,
        }


def build_optimizer(
    algorithm: str, risk_free_rate: float, max_weight: float
) -> BaseOptimizer:
    if algorithm not in ALGORITHMS:
        raise UnknownAlgorithmError(
            f"Unknown algorithm {algorithm!r}. Expected one of {sorted(ALGORITHMS)}."
        )

    if algorithm == "hrp":
        return HRPOptimizer()
    if algorithm == "equal_weight":
        return EqualWeightOptimizer()
    return MVOOptimizer(risk_free_rate=risk_free_rate, max_weight=max_weight)


def run_optimization(
    tickers: list[str],
    start_date: str,
    end_date: str,
    algorithm: str,
    risk_free_rate: float,
    max_weight: float = 0.20,
) -> OptimizationResult:
    """Fetch market data, allocate capital, and measure the resulting portfolio."""
    optimizer = build_optimizer(algorithm, risk_free_rate, max_weight)

    market_data = prepare_market_data(tickers, start_date, end_date)

    daily_returns = market_data["daily_returns"]
    covariance = market_data["annual_covariance"]
    surviving_tickers = list(daily_returns.columns)

    if len(surviving_tickers) < 2:
        raise InsufficientDataError(
            f"Need at least 2 tickers with usable price history; "
            f"got {len(surviving_tickers)}. Missing: {market_data['missing_tickers']}"
        )

    # max_weight must admit a feasible solution: n assets capped at w can only
    # sum to n*w, so n*w must be at least 1.0.
    if algorithm == "mvo" and len(surviving_tickers) * max_weight < 1.0:
        raise InsufficientDataError(
            f"max_weight={max_weight} is infeasible for {len(surviving_tickers)} "
            f"assets; weights could sum to at most "
            f"{len(surviving_tickers) * max_weight:.2f}, not 1.0."
        )

    weights = optimizer.optimize(daily_returns, covariance)

    metrics = calculate_portfolio_metrics(
        weights=list(weights),
        expected_returns=market_data["expected_annual_returns"]
        .loc[surviving_tickers]
        .to_numpy(),
        covariance_matrix=covariance.loc[surviving_tickers, surviving_tickers].to_numpy(),
        risk_free_rate=risk_free_rate,
    )

    return OptimizationResult(
        tickers=surviving_tickers,
        weights=[float(w) for w in weights],
        algorithm=algorithm,
        metrics={k: float(v) for k, v in metrics.items()},
        missing_tickers=list(market_data["missing_tickers"]),
    )
