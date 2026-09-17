"""Tests for covariance estimation and Ledoit-Wolf shrinkage."""

import numpy as np
import pandas as pd
import pytest

from src.risk.covariance import (
    condition_number,
    constant_correlation_target,
    estimate_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)


def factor_returns(n_assets: int, n_obs: int, seed: int = 0) -> pd.DataFrame:
    """A market factor plus idiosyncratic noise -- realistic correlation structure."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_obs)
    betas = rng.uniform(0.5, 1.5, n_assets)
    data = np.outer(market, betas) + rng.normal(0, 0.008, (n_obs, n_assets))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


def naive_shrinkage_intensity(returns: pd.DataFrame) -> float:
    """Independent reference implementation, transcribed straight from
    Ledoit & Wolf (2003) with explicit loops.

    Deliberately slow and unvectorised. Its only job is to disagree with the
    fast implementation if the einsum rewrite got an index wrong.
    """
    values = returns.to_numpy()
    n_obs, n_assets = values.shape
    demeaned = values - values.mean(axis=0)

    sample = demeaned.T @ demeaned / n_obs
    std_devs = np.sqrt(np.diag(sample))

    correlations = [
        sample[i, j] / (std_devs[i] * std_devs[j])
        for i in range(n_assets)
        for j in range(n_assets)
        if i != j
    ]
    mean_correlation = float(np.mean(correlations))

    target = np.empty_like(sample)
    for i in range(n_assets):
        for j in range(n_assets):
            target[i, j] = (
                sample[i, i] if i == j else mean_correlation * std_devs[i] * std_devs[j]
            )

    # pi
    pi_matrix = np.zeros_like(sample)
    for i in range(n_assets):
        for j in range(n_assets):
            errors = demeaned[:, i] * demeaned[:, j] - sample[i, j]
            pi_matrix[i, j] = np.mean(errors**2)
    pi = pi_matrix.sum()

    # theta_ii,ij
    def theta(i: int, j: int) -> float:
        return float(
            np.mean(
                (demeaned[:, i] ** 2 - sample[i, i])
                * (demeaned[:, i] * demeaned[:, j] - sample[i, j])
            )
        )

    rho = float(np.trace(pi_matrix))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                continue
            rho += (mean_correlation / 2.0) * (
                (std_devs[j] / std_devs[i]) * theta(i, j)
                + (std_devs[i] / std_devs[j]) * theta(j, i)
            )

    gamma = float(((target - sample) ** 2).sum())

    return float(np.clip((pi - rho) / gamma / n_obs, 0.0, 1.0))


class TestAgainstReferenceImplementation:
    """The vectorised einsum code must match the literal transcription."""

    @pytest.mark.parametrize(
        ("n_assets", "n_obs", "seed"),
        [(4, 300, 0), (8, 250, 1), (15, 200, 2), (6, 80, 3), (20, 60, 4)],
    )
    def test_shrinkage_intensity_matches_naive_loops(self, n_assets, n_obs, seed):
        returns = factor_returns(n_assets, n_obs, seed)

        _, fast = ledoit_wolf_shrinkage(returns)
        slow = naive_shrinkage_intensity(returns)

        assert fast == pytest.approx(slow, abs=1e-12)


class TestConstantCorrelationTarget:
    def test_preserves_each_asset_variance(self):
        sample = sample_covariance(factor_returns(8, 300))
        target = constant_correlation_target(sample)
        np.testing.assert_allclose(np.diag(target), np.diag(sample), rtol=1e-12)

    def test_every_off_diagonal_correlation_is_identical(self):
        sample = sample_covariance(factor_returns(8, 300))
        target = constant_correlation_target(sample).to_numpy()

        std_devs = np.sqrt(np.diag(target))
        correlations = target / np.outer(std_devs, std_devs)
        off_diagonal = correlations[~np.eye(len(target), dtype=bool)]

        assert off_diagonal.std() == pytest.approx(0.0, abs=1e-12)

    def test_target_is_symmetric(self):
        target = constant_correlation_target(sample_covariance(factor_returns(6, 200)))
        np.testing.assert_allclose(target.to_numpy(), target.to_numpy().T, atol=1e-15)


class TestShrinkageProperties:
    def test_intensity_is_a_valid_proportion(self):
        _, intensity = ledoit_wolf_shrinkage(factor_returns(10, 250))
        assert 0.0 <= intensity <= 1.0

    def test_diagonal_matches_the_sample_estimator(self):
        """Both targets keep asset variances, so shrinkage must not move them.

        This also pins the 1/T vs 1/(T-1) correction: without it the diagonals
        differ by a factor of T/(T-1).
        """
        returns = factor_returns(12, 300)
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        np.testing.assert_allclose(
            np.diag(shrunk), np.diag(sample_covariance(returns)), rtol=1e-12
        )

    def test_result_is_symmetric(self):
        shrunk, _ = ledoit_wolf_shrinkage(factor_returns(10, 200))
        np.testing.assert_allclose(shrunk.to_numpy(), shrunk.to_numpy().T, atol=1e-18)

    def test_result_is_positive_definite_when_sample_is_not(self):
        """T < N makes the sample covariance singular. Shrinkage repairs it."""
        returns = factor_returns(n_assets=40, n_obs=25)

        sample = sample_covariance(returns).to_numpy()
        shrunk, intensity = ledoit_wolf_shrinkage(returns)

        assert np.linalg.eigvalsh(sample).min() < 1e-12  # singular
        assert np.linalg.eigvalsh(shrunk.to_numpy()).min() > 0  # repaired
        assert intensity > 0

    def test_preserves_labels_and_shape(self):
        returns = factor_returns(5, 200)
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        assert list(shrunk.columns) == list(returns.columns)
        assert list(shrunk.index) == list(returns.columns)

    def test_shrinks_harder_as_data_gets_scarcer(self):
        """Less data per parameter means the sample estimate deserves less trust."""
        plentiful = ledoit_wolf_shrinkage(factor_returns(20, 2000, seed=5))[1]
        scarce = ledoit_wolf_shrinkage(factor_returns(20, 60, seed=5))[1]
        assert scarce > plentiful

    def test_improves_conditioning_when_it_matters(self):
        returns = factor_returns(n_assets=60, n_obs=90)
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        assert condition_number(shrunk) < condition_number(sample_covariance(returns))

    def test_rejects_a_single_asset(self):
        with pytest.raises(ValueError, match="at least 2 assets"):
            ledoit_wolf_shrinkage(pd.DataFrame({"ONLY": [0.01, 0.02, -0.01]}))

    def test_rejects_a_single_observation(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            ledoit_wolf_shrinkage(pd.DataFrame({"A": [0.01], "B": [0.02]}))


class TestShrinkageAccuracy:
    """The point of shrinkage is a better estimate, not just a prettier one."""

    def test_is_closer_to_the_true_covariance_than_the_sample_estimate(self):
        """Simulate from a KNOWN covariance, then measure who gets nearer to it.

        This is the claim that justifies the whole module. Averaged over many
        draws in the hard regime (few observations per asset), Ledoit-Wolf
        should beat the sample estimator on Frobenius distance to truth.
        """
        n_assets, n_obs, n_trials = 30, 45, 40

        base = np.random.default_rng(99).normal(0, 1, (n_assets, n_assets))
        true_cov = (base @ base.T) / n_assets * 1e-4

        sample_errors, shrunk_errors = [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(1000 + trial)
            draws = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_obs)
            returns = pd.DataFrame(draws, columns=[f"A{i}" for i in range(n_assets)])

            annual_truth = true_cov * 252
            sample_errors.append(
                np.linalg.norm(sample_covariance(returns).to_numpy() - annual_truth)
            )
            shrunk_errors.append(
                np.linalg.norm(
                    ledoit_wolf_shrinkage(returns)[0].to_numpy() - annual_truth
                )
            )

        assert np.mean(shrunk_errors) < np.mean(sample_errors)

    def test_wins_on_the_large_majority_of_individual_draws(self):
        n_assets, n_obs, n_trials = 25, 40, 30

        base = np.random.default_rng(7).normal(0, 1, (n_assets, n_assets))
        true_cov = (base @ base.T) / n_assets * 1e-4
        annual_truth = true_cov * 252

        wins = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(2000 + trial)
            draws = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_obs)
            returns = pd.DataFrame(draws, columns=[f"A{i}" for i in range(n_assets)])

            sample_error = np.linalg.norm(
                sample_covariance(returns).to_numpy() - annual_truth
            )
            shrunk_error = np.linalg.norm(
                ledoit_wolf_shrinkage(returns)[0].to_numpy() - annual_truth
            )
            wins += shrunk_error < sample_error

        assert wins >= int(0.8 * n_trials), f"only won {wins}/{n_trials}"


class TestConditionNumber:
    def test_identity_is_perfectly_conditioned(self):
        assert condition_number(np.eye(5)) == pytest.approx(1.0)

    def test_singular_matrix_reports_infinity(self):
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert condition_number(singular) == float("inf")

    def test_grows_as_a_matrix_approaches_singularity(self):
        mild = np.array([[1.0, 0.5], [0.5, 1.0]])
        severe = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        assert condition_number(severe) > condition_number(mild)


class TestEstimateCovariance:
    def test_sample_estimator_reports_no_shrinkage(self):
        estimate = estimate_covariance(factor_returns(6, 300), estimator="sample")
        assert estimate.shrinkage_intensity == 0.0
        assert estimate.estimator == "sample"

    def test_ledoit_wolf_is_the_default(self):
        assert estimate_covariance(factor_returns(6, 300)).estimator == "ledoit_wolf"

    def test_reports_diagnostics(self):
        estimate = estimate_covariance(factor_returns(10, 250))
        assert estimate.n_assets == 10
        assert estimate.n_observations == 250
        assert estimate.observations_per_asset == pytest.approx(25.0)
        assert estimate.condition_number > 0
        assert set(estimate.summary()) == {
            "estimator",
            "shrinkage_intensity",
            "condition_number",
            "n_observations",
            "n_assets",
            "observations_per_asset",
        }

    def test_rejects_an_unknown_estimator(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            estimate_covariance(factor_returns(4, 100), estimator="magic")
