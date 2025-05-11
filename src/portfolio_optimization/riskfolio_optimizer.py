# src/portfolio_optimization/riskfolio_optimizer.py

import pandas as pd
import riskfolio as rf
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskfolioOptimizer:
    """
    A class to perform portfolio optimization using the riskfolio library,
    integrated with the project's data flow.
    """
    def __init__(self):
        """
        Initializes the RiskfolioOptimizer. Returns data needs to be passed
        to the optimization methods.
        """
        self.portfolio: Optional[rf.Portfolio] = None
        self.weights: Optional[pd.DataFrame] = None

    def load_returns(self, returns: pd.DataFrame):
        """
        Loads the asset returns DataFrame.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns with dates as index and symbols as columns.
        """
        self.portfolio = rf.Portfolio(returns=returns)
        logging.info("Asset returns loaded for Riskfolio optimization.")

    def estimate_covariance(self, method: str = 'hist', **kwargs):
        """
        Estimates the covariance matrix of asset returns.

        Args:
            method (str): The method to use for covariance estimation (e.g., 'hist', 'ledoit').
            **kwargs: Additional keyword arguments to pass to the estimation method.
        """
        if self.portfolio is None:
            logging.error("Asset returns not loaded. Call load_returns() first.")
            return
        try:
            self.portfolio.estimation(method=method, **kwargs)
            logging.info(f"Estimated covariance matrix using '{method}' method.")
        except Exception as e:
            logging.error(f"Error estimating covariance matrix: {e}")

    def optimize_portfolio(self, model: str = 'Classic', rm: str = 'MV', obj: str = 'Sharpe',
                           rf: float = 0, **kwargs) -> Optional[pd.DataFrame]:
        """
        Performs portfolio optimization.

        Args:
            model (str): The optimization model (e.g., 'Classic', 'Factor').
            rm (str): The risk measure to use (e.g., 'MV' for variance, 'MAD' for mean absolute deviation).
            obj (str): The objective function to optimize (e.g., 'Sharpe', 'MinRisk').
            rf (float): The risk-free rate (for Sharpe ratio).
            **kwargs: Additional keyword arguments to pass to the optimization method (e.g., constraints).

        Returns:
            Optional[pd.DataFrame]: DataFrame of optimized portfolio weights, or None if optimization fails.
        """
        if self.portfolio is None:
            logging.error("Asset returns not loaded. Call load_returns() first.")
            return None
        try:
            self.portfolio.optimization(model=model, rm=rm, obj=obj, rf=rf, **kwargs)
            self.weights = self.portfolio.clean_weights()
            logging.info(f"Performed portfolio optimization with model='{model}', risk='{rm}', objective='{obj}'.")
            return self.weights
        except Exception as e:
            logging.error(f"Error during portfolio optimization: {e}")
            return None

    def get_efficient_frontier(self, points: int = 30, rf: float = 0) -> Optional[pd.DataFrame]:
        """
        Calculates the efficient frontier.

        Args:
            points (int): The number of points to calculate on the frontier.
            rf (float): The risk-free rate.

        Returns:
            Optional[pd.DataFrame]: DataFrame representing the efficient frontier.
        """
        if self.portfolio is None:
            logging.error("Asset returns not loaded. Call load_returns() first.")
            return None
        try:
            frontier = self.portfolio.efficient_frontier(points=points, rf=rf)
            logging.info(f"Calculated efficient frontier with {points} points.")
            return frontier
        except Exception as e:
            logging.error(f"Error calculating efficient frontier: {e}")
            return None

    def get_risk_contribution(self, rm: str = 'MV', **kwargs) -> Optional[pd.DataFrame]:
        """
        Calculates the risk contribution of each asset in the optimized portfolio.

        Args:
            rm (str): The risk measure to use.
            **kwargs: Additional keyword arguments for the risk contribution method.

        Returns:
            Optional[pd.DataFrame]: DataFrame of risk contributions.
        """
        if self.weights is None:
            logging.warning("Portfolio weights have not been calculated yet. Call optimize_portfolio() first.")
            return None
        try:
            risk_contribution = self.portfolio.risk_contribution(rm=rm, **kwargs)
            logging.info(f"Calculated risk contribution using '{rm}'.")
            return risk_contribution
        except Exception as e:
            logging.error(f"Error calculating risk contribution: {e}")
            return None

    def get_optimized_weights(self) -> Optional[pd.DataFrame]:
        """
        Returns the optimized portfolio weights.

        Returns:
            Optional[pd.DataFrame]: DataFrame of optimized portfolio weights.
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

    optimizer = RiskfolioOptimizer()
    optimizer.load_returns(returns_df)
    optimizer.estimate_covariance()
    weights = optimizer.optimize_portfolio(obj='Sharpe', rf=0.02)

    if weights is not None:
        print("\nOptimized Portfolio Weights (Riskfolio):\n", weights)
        risk_contribution = optimizer.get_risk_contribution()
        if risk_contribution is not None:
            print("\nRisk Contribution (Riskfolio):\n", risk_contribution)

        efficient_frontier = optimizer.get_efficient_frontier()
        if efficient_frontier is not None:
            print("\nEfficient Frontier (Riskfolio):\n", efficient_frontier.head())