# src/portfolio_optimization/pyportfolio_optimizer.py

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import objective_functions
import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PyPortfolioOptimizer:
    """
    A class to perform portfolio optimization using the PyPortfolioOpt library,
    integrated with the project's data flow.
    """
    def __init__(self):
        """
        Initializes the PyPortfolioOptimizer. Returns data needs to be passed
        to the optimization methods.
        """
        self.returns: Optional[pd.DataFrame] = None
        self.mu: Optional[pd.Series] = None
        self.S: Optional[pd.DataFrame] = None
        self.ef: Optional[EfficientFrontier] = None
        self.weights: Optional[dict] = None

    def load_returns(self, returns: pd.DataFrame):
        """
        Loads the asset returns DataFrame.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns with dates as index and symbols as columns.
        """
        self.returns = returns
        self.mu = self.returns.mean()
        self.S = risk_models.sample_cov(self.returns)
        logging.info("Asset returns loaded for PyPortfolioOpt optimization.")

    def efficient_frontier(self, method: str = 'max_sharpe', **kwargs) -> Optional[dict]:
        """
        Calculates the efficient frontier or a specific portfolio on it.

        Args:
            method (str): The optimization method ('max_sharpe', 'min_volatility', 'efficient_return', 'efficient_risk').
            **kwargs: Additional keyword arguments for the chosen method (e.g., target_return, target_volatility).

        Returns:
            Optional[dict]: Dictionary of optimized weights, or None if optimization fails.
        """
        if self.mu is None or self.S is None:
            logging.error("Asset returns not loaded. Call load_returns() first.")
            return None
        try:
            self.ef = EfficientFrontier(self.mu, self.S, **kwargs)
            if method == 'max_sharpe':
                self.weights = self.ef.max_sharpe()
            elif method == 'min_volatility':
                self.weights = self.ef.min_volatility()
            elif method == 'efficient_return' and 'target_return' in kwargs:
                self.weights = self.ef.efficient_return(kwargs['target_return'])
            elif method == 'efficient_risk' and 'target_volatility' in kwargs:
                self.weights = self.ef.efficient_risk(kwargs['target_volatility'])
            else:
                logging.error(f"Invalid optimization method: '{method}' or missing required arguments.")
                return None
            cleaned_weights = self.ef.clean_weights()
            self.weights = cleaned_weights
            logging.info(f"Performed portfolio optimization using '{method}'.")
            return self.weights
        except Exception as e:
            logging.error(f"Error during PyPortfolioOpt optimization: {e}")
            return None

    def performance(self, risk_free_rate: float = 0.02) -> Optional[tuple]:
        """
        Returns the performance of the optimized portfolio (if weights are calculated).

        Args:
            risk_free_rate (float): The risk-free rate.

        Returns:
            Optional[tuple]: Tuple containing expected annual return, annual volatility, and Sharpe ratio.
        """
        if self.ef is None or self.weights is None:
            logging.warning("Portfolio weights have not been calculated yet. Call efficient_frontier() first.")
            return None
        try:
            performance = self.ef.portfolio_performance(risk_free_rate=risk_free_rate)
            logging.info("Calculated portfolio performance.")
            return performance
        except Exception as e:
            logging.error(f"Error calculating portfolio performance: {e}")
            return None

    def get_optimized_weights(self) -> Optional[dict]:
        """
        Returns the optimized portfolio weights.

        Returns:
            Optional[dict]: Dictionary of optimized portfolio weights.
        """
        return self.weights

# Example Usage (requires asset returns DataFrame loaded elsewhere)
if __name__ == "__main__":
    # Sample returns data (in a real scenario, this would come from arcticdb_loader)
    data = {'AAPL': np.random.rand(100),
            'MSFT': np.random.rand(100) + 0.01,
            'GOOG': np.random.rand(100) - 0.01}
    index = pd.to_datetime(pd.date_range('2025-01-01', periods=100, freq='B'))
    returns_df = pd.DataFrame(data, index=index)

    optimizer = PyPortfolioOptimizer()
    optimizer.load_returns(returns_df)
    weights = optimizer.efficient_frontier(method='max_sharpe')

    if weights is not None:
        print("\nOptimized Portfolio Weights (PyPortfolioOpt):\n", weights)
        performance = optimizer.performance()
        if performance is not None:
            expected_return, volatility, sharpe_ratio = performance
            print(f"Expected Annual Return: {expected_return:.4f}")
            print(f"Annual Volatility: {volatility:.4f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    min_vol_weights = optimizer.efficient_frontier(method='min_volatility')
    if min_vol_weights is not None:
        print("\nMinimum Volatility Portfolio Weights (PyPortfolioOpt):\n", min_vol_weights)