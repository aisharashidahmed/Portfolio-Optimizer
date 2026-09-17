"""Walk-forward backtesting and performance measurement."""

from .engine import (
    BacktestConfigError,
    BacktestResult,
    compare_strategies,
    run_walk_forward,
)
from .performance import PerformanceStats, drawdown_series, summarise

__all__ = [
    "BacktestConfigError",
    "BacktestResult",
    "PerformanceStats",
    "compare_strategies",
    "drawdown_series",
    "run_walk_forward",
    "summarise",
]
