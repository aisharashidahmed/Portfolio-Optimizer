"""Performance statistics for a return series.

Deliberately separate from the backtest engine: these functions take a plain
series of periodic returns and know nothing about how it was produced, so the
same code measures a backtest, a benchmark, or a live track record.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# A constant return series has a standard deviation of ~1e-18 rather than
# exactly 0, so an `== 0` guard misses it and the ratio explodes to ~1e17.
# Anything below this floor is treated as riskless.
VOLATILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class PerformanceStats:
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    value_at_risk_95: float
    conditional_value_at_risk_95: float
    win_rate: float
    n_periods: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period."""
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annualised_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Geometric (compound) annual growth rate.

    Geometric, not arithmetic: +50% then -50% is a 25% loss, not break-even.
    """
    if returns.empty:
        return 0.0

    growth = float((1.0 + returns).prod())
    years = len(returns) / periods_per_year

    if years <= 0:
        return 0.0
    if growth <= 0:
        return -1.0  # the portfolio was wiped out

    return float(growth ** (1.0 / years) - 1.0)


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Excess return per unit of total volatility."""
    volatility = annualised_volatility(returns, periods_per_year)
    if volatility < VOLATILITY_FLOOR:
        return 0.0
    return (annualised_return(returns, periods_per_year) - risk_free_rate) / volatility


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Like Sharpe, but only counts downside volatility as risk.

    Upside surprises are not something an investor needs compensating for.
    """
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0

    downside_deviation = float(downside.std(ddof=1) * np.sqrt(periods_per_year))
    if downside_deviation < VOLATILITY_FLOOR:
        return 0.0

    return (
        annualised_return(returns, periods_per_year) - risk_free_rate
    ) / downside_deviation


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak, at every point in time."""
    if returns.empty:
        return pd.Series(dtype=float)

    equity_curve = (1.0 + returns).cumprod()
    running_peak = equity_curve.cummax()
    return equity_curve / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative number.

    Often the number that actually matters: it is the loss an investor had to
    sit through without capitulating, and it is invisible in an average return.
    """
    drawdowns = drawdown_series(returns)
    return float(drawdowns.min()) if not drawdowns.empty else 0.0


def calmar_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised return divided by the worst drawdown."""
    worst = abs(max_drawdown(returns))
    if worst < VOLATILITY_FLOOR:
        return 0.0
    return annualised_return(returns, periods_per_year) / worst


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the loss exceeded only (1 - confidence) of the time."""
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1.0 - confidence) * 100.0))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected loss on the days when VaR is breached -- how bad the tail is."""
    if returns.empty:
        return 0.0

    threshold = value_at_risk(returns, confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if not tail.empty else threshold


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return."""
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def summarise(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceStats:
    """Compute every statistic for a return series."""
    returns = returns.dropna()

    return PerformanceStats(
        total_return=total_return(returns),
        annualised_return=annualised_return(returns, periods_per_year),
        annualised_volatility=annualised_volatility(returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(returns),
        calmar_ratio=calmar_ratio(returns, periods_per_year),
        value_at_risk_95=value_at_risk(returns, 0.95),
        conditional_value_at_risk_95=conditional_value_at_risk(returns, 0.95),
        win_rate=win_rate(returns),
        n_periods=len(returns),
    )
