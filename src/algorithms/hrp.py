"""Hierarchical Risk Parity (Lopez de Prado, 2016).

HRP avoids inverting the covariance matrix. Instead it clusters assets by
correlation distance, then splits capital down the resulting tree using
inverse-variance allocation. This makes it far more stable than mean-variance
optimization when the covariance matrix is ill-conditioned -- which it usually
is when you have more assets than you have clean observations.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .base import BaseOptimizer


def correlation_from_covariance(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Derive the correlation matrix implied by a covariance matrix.

    corr = D^-1 @ cov @ D^-1, where D = diag(sqrt(diag(cov))).

    Note this is *not* the same as ``cov_matrix.corr()``, which would treat the
    covariance matrix's columns as if they were observations of a variable and
    return a meaningless result.
    """
    variances = np.diag(cov_matrix.to_numpy())

    if np.any(variances <= 0):
        raise ValueError(
            "Covariance matrix has a non-positive variance on its diagonal; "
            "an asset with zero variance cannot be clustered."
        )

    inverse_std = 1.0 / np.sqrt(variances)
    correlation_array = cov_matrix.to_numpy() * np.outer(inverse_std, inverse_std)

    # Floating point drift can push entries a hair outside [-1, 1], which makes
    # the distance formula below produce NaNs.
    correlation_array = np.clip(correlation_array, -1.0, 1.0)
    np.fill_diagonal(correlation_array, 1.0)

    return pd.DataFrame(
        correlation_array,
        index=cov_matrix.index,
        columns=cov_matrix.columns,
    )


def inverse_variance_weights(cov_matrix: pd.DataFrame) -> np.ndarray:
    """Allocate inversely to each asset's variance.

    This is what HRP reduces to when the correlation structure carries no
    clustering information, and it depends only on the diagonal, so it is
    independent of asset ordering.
    """
    inverse_variance = 1.0 / np.diag(cov_matrix.to_numpy())
    return inverse_variance / inverse_variance.sum()


class HRPOptimizer(BaseOptimizer):
    """Allocate capital by recursively bisecting a correlation dendrogram."""

    def __init__(self, linkage_method: str = "single"):
        # Lopez de Prado uses single linkage. 'ward' assumes Euclidean geometry
        # that correlation distances do not satisfy, so it is not the default.
        self.linkage_method = linkage_method

    def _get_distance_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Convert correlations into a proper distance metric: d = sqrt(0.5*(1-rho))."""
        correlation = correlation_from_covariance(cov_matrix)

        distance_array = np.sqrt(0.5 * (1.0 - correlation.to_numpy()))
        np.fill_diagonal(distance_array, 0.0)

        # squareform() rejects matrices that are not exactly symmetric.
        distance_array = (distance_array + distance_array.T) / 2.0

        return pd.DataFrame(
            distance_array,
            index=correlation.index,
            columns=correlation.columns,
        )

    @staticmethod
    def _distances_are_degenerate(
        distance_matrix: pd.DataFrame, tolerance: float = 1e-12
    ) -> bool:
        """True when every off-diagonal distance is effectively identical."""
        values = distance_matrix.to_numpy()
        off_diagonal = values[~np.eye(len(values), dtype=bool)]

        if off_diagonal.size == 0:
            return True

        return bool(np.ptp(off_diagonal) <= tolerance)

    def _cluster(self, distance_matrix: pd.DataFrame) -> np.ndarray:
        condensed_distances = squareform(distance_matrix.to_numpy(), checks=False)
        return linkage(condensed_distances, method=self.linkage_method)

    def _get_items(
        self, linkage_matrix: np.ndarray, node_id: int, n_assets: int
    ) -> list[int]:
        """Return the original asset indices sitting under ``node_id``."""
        if node_id < n_assets:
            return [int(node_id)]

        stack = [node_id]
        leaves: list[int] = []

        # Iterative walk: a recursive one blows the stack on large universes.
        while stack:
            current = stack.pop()
            if current < n_assets:
                leaves.append(int(current))
                continue
            row = linkage_matrix[current - n_assets]
            stack.append(int(row[1]))
            stack.append(int(row[0]))

        return leaves

    def _get_cluster_variance(
        self, cov_matrix: pd.DataFrame, cluster_indices: list[int]
    ) -> float:
        """Variance of a cluster held at inverse-variance weights.

        Lopez de Prado weights assets *within* a cluster by inverse variance
        before measuring the cluster's risk -- not equally.
        """
        sub_cov = cov_matrix.to_numpy()[np.ix_(cluster_indices, cluster_indices)]

        inverse_variance = 1.0 / np.diag(sub_cov)
        weights = inverse_variance / inverse_variance.sum()

        return float(weights @ sub_cov @ weights)

    def _bisect(
        self,
        cov_matrix: pd.DataFrame,
        linkage_matrix: np.ndarray,
        node_id: int,
        n_assets: int,
    ) -> dict[int, float]:
        """Split capital down the tree, favouring the lower-variance branch."""
        if node_id < n_assets:
            return {int(node_id): 1.0}

        row = linkage_matrix[node_id - n_assets]
        left_child, right_child = int(row[0]), int(row[1])

        left_items = self._get_items(linkage_matrix, left_child, n_assets)
        right_items = self._get_items(linkage_matrix, right_child, n_assets)

        left_variance = self._get_cluster_variance(cov_matrix, left_items)
        right_variance = self._get_cluster_variance(cov_matrix, right_items)

        total_variance = left_variance + right_variance
        if total_variance <= 0:
            left_weight = right_weight = 0.5
        else:
            # The calmer branch receives the larger share.
            left_weight = 1.0 - left_variance / total_variance
            right_weight = 1.0 - right_variance / total_variance

        allocation: dict[int, float] = {}
        for index, weight in self._bisect(
            cov_matrix, linkage_matrix, left_child, n_assets
        ).items():
            allocation[index] = weight * left_weight
        for index, weight in self._bisect(
            cov_matrix, linkage_matrix, right_child, n_assets
        ).items():
            allocation[index] = weight * right_weight

        return allocation

    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Return weights aligned to the column order of ``cov_matrix``."""
        n_assets = len(cov_matrix)

        if n_assets == 0:
            raise ValueError("Cannot optimize an empty universe")
        if n_assets == 1:
            return np.array([1.0])

        distance_matrix = self._get_distance_matrix(cov_matrix)

        # When every pair of assets is equidistant there is no canonical
        # dendrogram: scipy breaks the tie by index order, so permuting the
        # inputs would silently change the allocation. Hypothesis found exactly
        # this (three assets, all pairwise distances 0.713176). With no
        # clustering signal to exploit, fall back to the order-independent
        # limit of HRP.
        #
        # Note this only covers *total* degeneracy. Partial ties -- some pairs
        # equidistant, others not -- remain order-dependent, which is a
        # property of hierarchical clustering itself rather than of this code.
        if self._distances_are_degenerate(distance_matrix):
            return inverse_variance_weights(cov_matrix)

        linkage_matrix = self._cluster(distance_matrix)

        root_node = 2 * n_assets - 2
        allocation = self._bisect(cov_matrix, linkage_matrix, root_node, n_assets)

        weights = np.array([allocation[i] for i in range(n_assets)])

        # Guard against drift from the repeated multiplications above.
        return weights / weights.sum()
