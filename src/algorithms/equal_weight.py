"""Equal-weight allocation: the baseline every optimizer must beat."""

import numpy as np
import pandas as pd

from .base import BaseOptimizer


def create_equal_weights(asset_count: int) -> list[float]:
    if asset_count <= 0:
        raise ValueError("Asset count must be greater than zero")

    equal_weight = 1.0 / asset_count
    return [equal_weight] * asset_count


class EqualWeightOptimizer(BaseOptimizer):
    """1/N allocation.

    Worth taking seriously: DeMiguel, Garlappi & Uppal (2009) found 1/N beats
    most optimizers out-of-sample once estimation error is accounted for. This
    is the benchmark, not a placeholder.
    """

    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        return np.array(create_equal_weights(len(cov_matrix)))
