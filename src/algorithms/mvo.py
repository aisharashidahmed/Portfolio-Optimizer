"""Mean-Variance Optimization (Markowitz, 1952).

Finds the weights maximising the Sharpe ratio subject to being long-only,
fully invested, and capped per asset.

The weight cap is not decoration. Unconstrained MVO is notorious for producing
extreme, unstable allocations: it treats the estimated expected returns as
exact and piles into whichever asset's mean was most overestimated. Capping
per-asset weight is the crudest but most reliable defence, and pairing it with
a shrunk covariance matrix (see :mod:`src.risk.covariance`) addresses the other
half of the estimation-error problem.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import BaseOptimizer

TRADING_DAYS_PER_YEAR = 252

# Below this, a portfolio is riskless and the Sharpe ratio is undefined.
VOLATILITY_FLOOR = 1e-12


class OptimizationFailedError(ValueError):
    """The solver could not find a valid allocation."""


class MVOOptimizer(BaseOptimizer):
    """Maximise the Sharpe ratio under long-only, capped-weight constraints."""

    def __init__(self, risk_free_rate: float = 0.045, max_weight: float = 0.20):
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")

        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.n_assets: int | None = None

    def _neg_sharpe(
        self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> float:
        """Negative Sharpe ratio -- minimising this maximises Sharpe."""
        portfolio_return = float(weights @ expected_returns)

        # SLSQP's line search probes points slightly outside the feasible set,
        # where floating-point error can make this quadratic form a tiny
        # negative number. sqrt() of that is NaN, which poisons the objective
        # and makes the solver fail with a spurious line-search error.
        variance = max(float(weights @ cov_matrix @ weights), 0.0)
        volatility = np.sqrt(variance)

        if volatility < VOLATILITY_FLOOR:
            # Riskless portfolio: return a large finite penalty rather than
            # inf, which SLSQP cannot differentiate through.
            return 0.0 if portfolio_return <= self.risk_free_rate else -1e6

        return -(portfolio_return - self.risk_free_rate) / volatility

    def _constraints(self) -> dict:
        """Fully invested: weights sum to exactly 1."""
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def _bounds(self) -> tuple[tuple[float, float], ...]:
        """Long-only, and no position above the cap."""
        return tuple((0.0, self.max_weight) for _ in range(self.n_assets))

    def _starting_points(self) -> list[np.ndarray]:
        """Several feasible starts.

        SLSQP is a local method: on a badly conditioned problem it can stall at
        the first point it tries. Retrying from a different corner of the
        feasible set usually clears it, and costs microseconds.
        """
        n = self.n_assets
        equal = np.full(n, 1.0 / n)

        starts = [equal]

        # Concentrate on the first assets, respecting the cap.
        if self.max_weight < 1.0:
            greedy = np.zeros(n)
            remaining = 1.0
            for i in range(n):
                allocation = min(self.max_weight, remaining)
                greedy[i] = allocation
                remaining -= allocation
                if remaining <= 0:
                    break
            starts.append(greedy)

        rng = np.random.default_rng(0)  # fixed seed: results stay reproducible
        for _ in range(3):
            candidate = rng.dirichlet(np.ones(n))
            starts.append(np.clip(candidate, 0.0, self.max_weight))

        return starts

    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        self.n_assets = len(cov_matrix)

        if self.n_assets == 0:
            raise OptimizationFailedError("Cannot optimize an empty universe")
        if self.n_assets == 1:
            return np.array([1.0])

        if self.n_assets * self.max_weight < 1.0:
            raise OptimizationFailedError(
                f"max_weight={self.max_weight} is infeasible for {self.n_assets} "
                f"assets: weights could sum to at most "
                f"{self.n_assets * self.max_weight:.4f}, never 1.0"
            )

        expected_returns = returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
        covariance = cov_matrix.to_numpy()

        failures: list[str] = []
        for start in self._starting_points():
            # Renormalise: clipping to the cap can break the sum-to-one start.
            start = start / start.sum()

            result = minimize(
                self._neg_sharpe,
                start,
                args=(expected_returns, covariance),
                method="SLSQP",
                bounds=self._bounds(),
                constraints=self._constraints(),
                options={"maxiter": 500, "ftol": 1e-10},
            )

            if result.success and np.all(np.isfinite(result.x)):
                # Clean up the small constraint violations SLSQP tolerates.
                weights = np.clip(result.x, 0.0, self.max_weight)
                total = weights.sum()
                if total > 0:
                    return weights / total

            failures.append(result.message)

        raise OptimizationFailedError(
            f"SLSQP failed from all {len(failures)} starting points. "
            f"This usually means the covariance matrix is degenerate -- try the "
            f"'hrp' algorithm, which does not invert it. Last error: {failures[-1]}"
        )
