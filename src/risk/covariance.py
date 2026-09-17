"""Covariance estimation.

The sample covariance matrix is the obvious estimator and a poor one. With N
assets you are estimating N(N+1)/2 parameters from T observations; at N=50 that
is 1,275 numbers from maybe 500 trading days. The estimate is noisy, and
mean-variance optimization actively seeks out that noise -- it loads up on
whichever asset pair *appears* most negatively correlated, which is usually the
pair whose correlation was estimated worst.

Ledoit-Wolf shrinkage pulls the sample estimate toward a structured target that
has far fewer parameters and therefore far less estimation error. The blend is

    Sigma = delta * F + (1 - delta) * S

and the point of the method is that delta is *derived*, not tuned: it is the
value minimising expected squared error from the true covariance matrix.

References
----------
Ledoit & Wolf (2003), "Honey, I Shrunk the Sample Covariance Matrix",
Journal of Portfolio Management. The constant-correlation target below is
theirs; it is the standard choice for equity portfolios.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

ESTIMATORS = ("sample", "ledoit_wolf")


@dataclass(frozen=True)
class CovarianceEstimate:
    """An annualised covariance matrix plus diagnostics about how it was made."""

    matrix: pd.DataFrame
    estimator: str
    shrinkage_intensity: float
    condition_number: float
    n_observations: int
    n_assets: int

    @property
    def observations_per_asset(self) -> float:
        """Below ~10 the sample estimator should not be trusted."""
        return self.n_observations / self.n_assets

    def summary(self) -> dict[str, float | str]:
        return {
            "estimator": self.estimator,
            "shrinkage_intensity": self.shrinkage_intensity,
            "condition_number": self.condition_number,
            "n_observations": self.n_observations,
            "n_assets": self.n_assets,
            "observations_per_asset": self.observations_per_asset,
        }


def condition_number(matrix: pd.DataFrame | np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue.

    A high condition number means the matrix is near-singular: inverting it
    amplifies estimation noise enormously, which is exactly what mean-variance
    optimization does. Above ~1000 the optimizer's output is mostly noise.
    """
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    eigenvalues = np.linalg.eigvalsh(values)

    smallest = eigenvalues.min()
    if smallest <= 0:
        return float("inf")

    return float(eigenvalues.max() / smallest)


def sample_covariance(
    daily_returns: pd.DataFrame, annualise: bool = True
) -> pd.DataFrame:
    """Plain sample covariance. The baseline every other estimator must beat."""
    covariance = daily_returns.cov()
    return covariance * TRADING_DAYS_PER_YEAR if annualise else covariance


def constant_correlation_target(sample: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf's shrinkage target F.

    Keeps each asset's own variance but replaces every pairwise correlation
    with the average correlation across the whole universe. That is N
    parameters plus one, instead of N(N+1)/2 -- heavily biased, but almost
    noise-free, which is the trade shrinkage exists to make.
    """
    values = sample.to_numpy()
    variances = np.diag(values)
    std_devs = np.sqrt(variances)

    outer_std = np.outer(std_devs, std_devs)
    correlations = values / outer_std

    n_assets = len(sample)
    off_diagonal = ~np.eye(n_assets, dtype=bool)
    mean_correlation = float(correlations[off_diagonal].mean())

    target = mean_correlation * outer_std
    np.fill_diagonal(target, variances)

    return pd.DataFrame(target, index=sample.index, columns=sample.columns)


def _optimal_shrinkage_intensity(
    demeaned: np.ndarray, sample: np.ndarray, target: np.ndarray
) -> float:
    """Ledoit-Wolf's analytically optimal delta.

    delta* = (pi - rho) / gamma / T, clipped to [0, 1], where

      pi    -- total estimation variance of the sample covariance entries
      rho   -- covariance between the estimation errors of S and F
      gamma -- how wrong the target is (its squared distance from S)

    The intuition is the whole method: shrink hard when the sample estimate is
    noisy (large pi) and the target is not badly misspecified (small gamma).
    """
    n_obs, n_assets = demeaned.shape

    # pi: asymptotic variance of each entry of the sample covariance.
    products = np.einsum("ti,tj->tij", demeaned, demeaned)
    pi_matrix = ((products - sample) ** 2).mean(axis=0)
    pi = pi_matrix.sum()

    # gamma: squared Frobenius distance between target and sample.
    gamma = float(((target - sample) ** 2).sum())
    if gamma <= 0:
        # The target already equals the sample estimate; shrinking is a no-op.
        return 0.0

    variances = np.diag(sample)
    std_devs = np.sqrt(variances)
    correlations = sample / np.outer(std_devs, std_devs)
    off_diagonal = ~np.eye(n_assets, dtype=bool)
    mean_correlation = correlations[off_diagonal].mean()

    # theta_ii,ij: covariance between the error in s_ii and the error in s_ij.
    squared_demeaned = demeaned**2
    term = np.einsum("ti,tij->ij", squared_demeaned, products) / n_obs
    theta = term - variances[:, None] * sample

    ratio = np.outer(std_devs, 1.0 / std_devs)  # sqrt(s_jj/s_ii) transposed below
    rho_off = (mean_correlation / 2.0) * (ratio.T * theta + ratio * theta.T)
    np.fill_diagonal(rho_off, 0.0)

    rho = float(np.diag(pi_matrix).sum() + rho_off.sum())

    intensity = (pi - rho) / gamma / n_obs
    return float(np.clip(intensity, 0.0, 1.0))


def ledoit_wolf_shrinkage(
    daily_returns: pd.DataFrame, annualise: bool = True
) -> tuple[pd.DataFrame, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Returns the estimate and the shrinkage intensity actually used, so callers
    can report how much the sample estimate was distrusted. The result uses the
    same unbiased 1/(T-1) convention as :func:`sample_covariance`, so the two
    are interchangeable.
    """
    if daily_returns.shape[1] < 2:
        raise ValueError("Shrinkage requires at least 2 assets")
    if len(daily_returns) < 2:
        raise ValueError("Shrinkage requires at least 2 observations")

    # Use the 1/T (biased) estimator here: Ledoit-Wolf's derivation assumes it.
    values = daily_returns.to_numpy()
    demeaned = values - values.mean(axis=0)
    n_obs = len(demeaned)

    sample = demeaned.T @ demeaned / n_obs

    sample_frame = pd.DataFrame(
        sample, index=daily_returns.columns, columns=daily_returns.columns
    )
    target_frame = constant_correlation_target(sample_frame)
    target = target_frame.to_numpy()

    intensity = _optimal_shrinkage_intensity(demeaned, sample, target)

    shrunk = intensity * target + (1.0 - intensity) * sample
    shrunk = (shrunk + shrunk.T) / 2.0  # kill floating-point asymmetry

    # Ledoit-Wolf's derivation is stated for the 1/T (maximum-likelihood)
    # covariance, but pandas' .cov() -- and therefore sample_covariance() --
    # uses the unbiased 1/(T-1) divisor. Without this correction the two
    # estimators would return matrices on scales differing by T/(T-1), and
    # swapping one for the other would silently shift every volatility number.
    # The factor is ~1.002 at T=400, but correctness here is free.
    shrunk = shrunk * n_obs / (n_obs - 1)

    estimate = pd.DataFrame(
        shrunk, index=daily_returns.columns, columns=daily_returns.columns
    )
    if annualise:
        estimate = estimate * TRADING_DAYS_PER_YEAR

    return estimate, intensity


def estimate_covariance(
    daily_returns: pd.DataFrame, estimator: str = "ledoit_wolf"
) -> CovarianceEstimate:
    """Estimate an annualised covariance matrix and report diagnostics."""
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"Unknown estimator {estimator!r}. Expected one of {list(ESTIMATORS)}."
        )

    if estimator == "sample":
        matrix = sample_covariance(daily_returns)
        intensity = 0.0
    else:
        matrix, intensity = ledoit_wolf_shrinkage(daily_returns)

    return CovarianceEstimate(
        matrix=matrix,
        estimator=estimator,
        shrinkage_intensity=intensity,
        condition_number=condition_number(matrix),
        n_observations=len(daily_returns),
        n_assets=daily_returns.shape[1],
    )
