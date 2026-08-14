from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseOptimizer(ABC):
    """Abstract base class for all portfolio optimization algorithms."""
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate the optimal portfolio weights.
        
        Args:
            returns: DataFrame of historical returns (assets as columns).
            cov_matrix: Covariance matrix (annualized).
            
        Returns:
            np.ndarray: Array of weights that sum to 1.0.
        """
        pass