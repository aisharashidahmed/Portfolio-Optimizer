"""Walk-forward backtesting.

A single train/test split tells you how one strategy did across one regime.
Walk-forward re-fits the optimizer on a rolling window and steps forward, so
the result reflects many independent decisions instead of one lucky one.

Two details that a naive backtester gets wrong, and that this one models:

**Weights drift.** Buy a 50/50 portfolio and after a month it is not 50/50 any
more -- the winner grew. Assuming the weights stay fixed silently credits the
strategy with rebalancing continuously and for free.

**Rebalancing costs money.** Turnover incurs spread and commission. A strategy
that trades hard can look excellent gross and lose money net, so costs are
charged on every rebalance rather than bolted on at the end.

No-lookahead is enforced structurally: weights used from day ``t`` onward are
fitted on returns strictly before ``t``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.algorithms.base import BaseOptimizer
from src.backtesting.performance import PerformanceStats, summarise
from src.risk.covariance import estimate_covariance

# 10bp per unit traded: a retail-realistic round-trip cost on liquid equities.
DEFAULT_TRANSACTION_COST_BPS = 10.0


class BacktestConfigError(ValueError):
    pass


@dataclass
class BacktestResult:
    """Everything a backtest produced, gross and net."""

    net_returns: pd.Series
    gross_returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    performance: PerformanceStats
    gross_performance: PerformanceStats
    diagnostics: dict = field(default_factory=dict)

    @property
    def total_cost_drag(self) -> float:
        """How much return the trading cost consumed, in return terms."""
        return self.gross_performance.total_return - self.performance.total_return

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.mean()) if not self.turnover.empty else 0.0

    def equity_curve(self, net: bool = True) -> pd.Series:
        returns = self.net_returns if net else self.gross_returns
        return (1.0 + returns).cumprod()

    def summary(self) -> dict:
        return {
            **self.performance.to_dict(),
            "average_turnover": self.average_turnover,
            "total_cost_drag": self.total_cost_drag,
            "n_rebalances": int(len(self.turnover)),
            **self.diagnostics,
        }


def _drift_weights(weights: np.ndarray, period_returns: np.ndarray) -> np.ndarray:
    """Update weights for one period of price movement, before any trading.

    A position's value grows by (1 + r); its share of the portfolio is its new
    value over the new total.
    """
    grown = weights * (1.0 + period_returns)
    total = grown.sum()

    if total <= 0:
        # Total loss; nothing left to allocate.
        return np.zeros_like(weights)

    return grown / total


def run_walk_forward(
    daily_returns: pd.DataFrame,
    optimizer: BaseOptimizer,
    training_days: int = 252,
    rebalance_every: int = 21,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    covariance_estimator: str = "ledoit_wolf",
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Re-fit on a rolling window, step forward, and charge for trading.

    Parameters
    ----------
    training_days:
        Length of the lookback window used to fit weights. 252 is one year.
    rebalance_every:
        Trading days between refits. 21 is roughly monthly.
    transaction_cost_bps:
        Basis points charged per unit of weight traded. 10bp means moving 100%
        of the portfolio costs 0.10%.
    """
    if daily_returns.empty:
        raise BacktestConfigError("Cannot backtest an empty return series")
    if daily_returns.shape[1] < 2:
        raise BacktestConfigError("Backtesting requires at least 2 assets")
    if training_days < 2:
        raise BacktestConfigError("training_days must be at least 2")
    if rebalance_every < 1:
        raise BacktestConfigError("rebalance_every must be at least 1")
    if transaction_cost_bps < 0:
        raise BacktestConfigError("transaction_cost_bps cannot be negative")
    if len(daily_returns) <= training_days:
        raise BacktestConfigError(
            f"Need more than {training_days} observations to leave an "
            f"out-of-sample period; got {len(daily_returns)}"
        )

    returns_array = daily_returns.to_numpy()
    n_periods, n_assets = returns_array.shape
    cost_rate = transaction_cost_bps / 10_000.0

    current_weights = np.zeros(n_assets)

    gross: list[float] = []
    net: list[float] = []
    dates: list = []

    weight_rows: list[np.ndarray] = []
    weight_dates: list = []
    turnover_values: list[float] = []
    turnover_dates: list = []
    cost_values: list[float] = []

    shrinkage_intensities: list[float] = []
    condition_numbers: list[float] = []

    for day in range(training_days, n_periods):
        periods_since_start = day - training_days

        if periods_since_start % rebalance_every == 0:
            # Fit on data strictly before `day`. This slice is the no-lookahead
            # guarantee -- it can never include the return we are about to earn.
            window = daily_returns.iloc[day - training_days : day]

            estimate = estimate_covariance(window, estimator=covariance_estimator)
            shrinkage_intensities.append(estimate.shrinkage_intensity)
            condition_numbers.append(estimate.condition_number)

            target_weights = np.asarray(
                optimizer.optimize(window, estimate.matrix), dtype=float
            )

            # Trade from wherever drift left us to the new target.
            traded = np.abs(target_weights - current_weights).sum()
            cost = traded * cost_rate

            turnover_values.append(float(traded))
            turnover_dates.append(daily_returns.index[day])
            cost_values.append(float(cost))

            current_weights = target_weights
        else:
            cost = 0.0

        period_returns = returns_array[day]
        gross_return = float(current_weights @ period_returns)

        # The cost is paid at the start of the period, before returns accrue.
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0

        gross.append(gross_return)
        net.append(net_return)
        dates.append(daily_returns.index[day])

        weight_rows.append(current_weights.copy())
        weight_dates.append(daily_returns.index[day])

        current_weights = _drift_weights(current_weights, period_returns)

    net_series = pd.Series(net, index=dates, name="net")
    gross_series = pd.Series(gross, index=dates, name="gross")

    return BacktestResult(
        net_returns=net_series,
        gross_returns=gross_series,
        weights_history=pd.DataFrame(
            weight_rows, index=weight_dates, columns=daily_returns.columns
        ),
        turnover=pd.Series(turnover_values, index=turnover_dates, name="turnover"),
        costs=pd.Series(cost_values, index=turnover_dates, name="cost"),
        performance=summarise(net_series, risk_free_rate),
        gross_performance=summarise(gross_series, risk_free_rate),
        diagnostics={
            "covariance_estimator": covariance_estimator,
            "mean_shrinkage_intensity": float(np.mean(shrinkage_intensities))
            if shrinkage_intensities
            else 0.0,
            "mean_condition_number": float(np.mean(condition_numbers))
            if condition_numbers
            else 0.0,
            "transaction_cost_bps": transaction_cost_bps,
            "training_days": training_days,
            "rebalance_every": rebalance_every,
        },
    )


def compare_strategies(
    daily_returns: pd.DataFrame,
    optimizers: dict[str, BaseOptimizer],
    **kwargs,
) -> pd.DataFrame:
    """Run several strategies over identical data and tabulate the results."""
    rows = {
        name: run_walk_forward(daily_returns, optimizer, **kwargs).summary()
        for name, optimizer in optimizers.items()
    }
    return pd.DataFrame(rows).T
