import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .base import BaseOptimizer

class MVOOptimizer(BaseOptimizer):
    """
    The Swiss Watch. 
    Uses Mean-Variance Optimization to mathematically find the peak of the mountain.
    """
    
    def __init__(self, risk_free_rate: float = 0.045, max_weight: float = 0.20):
        """
        Setting up the Watch's default dials.
        - risk_free_rate: The return you get from cash (e.g., 4.5%).
        - max_weight: No single stock can exceed this percentage.
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.n_assets = None

    def _neg_sharpe(self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Measures the negative slope of the mountain.
        The lower this number, the higher the peak.
        """
        # 1. Calculate the Return of this portfolio
        portfolio_return = np.dot(weights, expected_returns)
        
        # 2. Calculate the Risk (Volatility) of this portfolio
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 3. Calculate the Sharpe Ratio (Return per unit of Risk)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # 4. Return the negative version (so the compass finds the peak)
        return -sharpe
    
    def _constraints(self):
        """Rule 1: The percentages must add up to exactly 100% (1.0)."""
        return {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    def _bounds(self):
        """Rule 2: No single stock gets more than max_weight (e.g., 20%)."""
        return tuple((0.0, self.max_weight) for _ in range(self.n_assets))
    
    def optimize(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        The main door to the Swiss Watch.
        Calculates the optimal weights using Math (Quadratic Programming).
        """
        # 1. Set up the number of assets (stocks) we are dealing with
        self.n_assets = len(cov_matrix)
        
        # 2. Calculate the Average Annualized Return for each stock (the 'Taste')
        expected_returns = returns.mean().values * 252
        
        # 3. Convert the covariance matrix to a numpy array for the Calculator
        cov_array = cov_matrix.values
        
        # 4. Initial Guess: Just give every stock the same amount (e.g., 20% each)
        initial_guess = np.ones(self.n_assets) / self.n_assets
        
        # 5. The Swiss Watch spins its gears (the minimization)
        result = minimize(
            self._neg_sharpe,           # The compass (what to minimize)
            initial_guess,              # Where to start looking
            args=(expected_returns, cov_array),  # The data
            method='SLSQP',             # The specific math engine (works well with rules)
            bounds=self._bounds(),      # Rule 2
            constraints=self._constraints()  # Rule 1
        )
        
        # 6. Check if the Watch succeeded
        if not result.success:
            raise ValueError(f"Swiss Watch failed to find the peak: {result.message}")
        
        # 7. Return the optimal weights
        return result.x