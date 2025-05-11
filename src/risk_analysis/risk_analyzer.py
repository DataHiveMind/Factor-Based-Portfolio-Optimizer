# src/risk_analysis/risk_analyzer.py

import pandas as pd
import numpy as np
import logging
import riskfolio as rf
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskAnalyzer:
    """
    A class to perform risk analysis on investment portfolios, designed for
    seamless integration with the project's data and optimization modules.
    """
    def __init__(self, returns: Optional[pd.DataFrame] = None):
        """
        Initializes the RiskAnalyzer with asset returns.

        Args:
            returns (Optional[pd.DataFrame]): DataFrame of asset returns (index: dates, columns: symbols).
        """
        self.returns = returns
        self.portfolio = rf.Portfolio(returns=self.returns) if returns is not None else None
        self.weights: Optional[pd.DataFrame] = None

    def load_returns(self, returns: pd.DataFrame):
        """
        Loads the asset returns DataFrame.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns.
        """
        self.returns = returns
        self.portfolio = rf.Portfolio(returns=self.returns)
        logging.info("Asset returns loaded for risk analysis.")

    def load_portfolio_weights(self, weights: pd.Series):
        """
        Loads portfolio weights from an optimization result (e.g., from Riskfolio or PyPortfolioOpt).

        Args:
            weights (pd.Series): Series of portfolio weights with symbols as index.
        """
        self.weights = weights.to_frame('weights').T
        logging.info("Portfolio weights loaded for risk analysis.")

    def calculate_portfolio_volatility(self, covariance_matrix: Optional[pd.DataFrame] = None,
                                        method: str = 'MV') -> Optional[float]:
        """
        Calculates the portfolio volatility.

        Args:
            covariance_matrix (Optional[pd.DataFrame]): Pre-calculated covariance matrix (optional).
            method (str): The risk measure to use ('MV' for standard deviation).

        Returns:
            Optional[float]: The portfolio volatility.
        """
        if self.weights is None or self.returns is None:
            logging.error("Weights and/or returns not loaded.")
            return None

        if self.portfolio is None:
            self.portfolio = rf.Portfolio(returns=self.returns)

        if covariance_matrix is None:
            self.portfolio.estimation(method='hist')  # Use historical by default
            covariance_matrix = self.portfolio.cov

        try:
            portfolio_risk = self.portfolio.risk_portfolio(weights=self.weights.values.flatten(), rm=method, cov=covariance_matrix)
            logging.info(f"Calculated portfolio volatility using '{method}'.")
            return portfolio_risk if np.isscalar(portfolio_risk) else portfolio_risk.iloc[0, 0]
        except Exception as e:
            logging.error(f"Error calculating portfolio volatility: {e}")
            return None

    def calculate_value_at_risk(self, alpha: float = 0.05, method: str = 'hist') -> Optional[pd.DataFrame]:
        """
        Calculates the Value at Risk (VaR) of the portfolio.

        Args:
            alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
            method (str): The method to use ('hist', 'gaussian', etc.).

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the VaR.
        """
        if self.returns is None or self.weights is None:
            logging.error("Returns and/or weights not loaded.")
            return None

        if self.portfolio is None:
            self.portfolio = rf.Portfolio(returns=self.returns)

        try:
            portfolio_var = self.portfolio.VaR(weights=self.weights.values.flatten(), alpha=alpha, method=method)
            logging.info(f"Calculated Value at Risk (VaR) with alpha={alpha} using '{method}'.")
            return portfolio_var
        except Exception as e:
            logging.error(f"Error calculating Value at Risk (VaR): {e}")
            return None

    def calculate_expected_shortfall(self, alpha: float = 0.05, method: str = 'hist') -> Optional[pd.DataFrame]:
        """
        Calculates the Expected Shortfall (ES) or Conditional Value at Risk (CVaR) of the portfolio.

        Args:
            alpha (float): The confidence level.
            method (str): The method to use.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the Expected Shortfall.
        """
        if self.returns is None or self.weights is None:
            logging.error("Returns and/or weights not loaded.")
            return None

        if self.portfolio is None:
            self.portfolio = rf.Portfolio(returns=self.returns)

        try:
            portfolio_es = self.portfolio.ES(weights=self.weights.values.flatten(), alpha=alpha, method=method)
            logging.info(f"Calculated Expected Shortfall (ES) with alpha={alpha} using '{method}'.")
            return portfolio_es
        except Exception as e:
            logging.error(f"Error calculating Expected Shortfall (ES): {e}")
            return None

    def calculate_risk_contribution(self, covariance_matrix: Optional[pd.DataFrame] = None,
                                      rm: str = 'MV') -> Optional[pd.DataFrame]:
        """
        Calculates the risk contribution of each asset to the portfolio risk.

        Args:
            covariance_matrix (Optional[pd.DataFrame]): Pre-calculated covariance matrix.
            rm (str): The risk measure to use.

        Returns:
            Optional[pd.DataFrame]: DataFrame of risk contributions.
        """
        if self.weights is None or self.returns is None:
            logging.error("Weights and/or returns not loaded.")
            return None

        if self.portfolio is None:
            self.portfolio = rf.Portfolio(returns=self.returns)

        if covariance_matrix is None:
            self.portfolio.estimation(method='hist')
            covariance_matrix = self.portfolio.cov

        try:
            risk_contribution = self.portfolio.risk_contribution(weights=self.weights.values.flatten(), rm=rm, cov=covariance_matrix)
            logging.info(f"Calculated risk contribution using '{rm}'.")
            return risk_contribution
        except Exception as e:
            logging.error(f"Error calculating risk contribution: {e}")
            return None

# Example Usage (requires returns and weights from optimization)
if __name__ == "__main__":
    # Sample returns data (loaded using ArcticDBLoader elsewhere)
    data = {'AAPL': np.random.rand(100),
            'MSFT': np.random.rand(100) + 0.01,
            'GOOG': np.random.rand(100) - 0.01}
    index = pd.to_datetime(pd.date_range('2025-01-01', periods=100, freq='B'))
    returns_df = pd.DataFrame(data, index=index)

    # Sample weights obtained from an optimizer (e.g., RiskfolioOptimizer)
    weights_data_rf = pd.Series({'AAPL': 0.4, 'MSFT': 0.3, 'GOOG': 0.3})

    analyzer_rf = RiskAnalyzer(returns_df)
    analyzer_rf.load_portfolio_weights(weights_data_rf)

    volatility_rf = analyzer_rf.calculate_portfolio_volatility()
    if volatility_rf is not None:
        print(f"\nPortfolio Volatility (Riskfolio Weights): {volatility_rf:.4f}")

    # Sample weights obtained from another optimizer (e.g., PyPortfolioOptimizer)
    weights_data_pypfopt = pd.Series({'AAPL': 0.35, 'MSFT': 0.35, 'GOOG': 0.3})

    analyzer_pypfopt = RiskAnalyzer(returns_df)
    analyzer_pypfopt.load_portfolio_weights(weights_data_pypfopt)

    var_pypfopt = analyzer_pypfopt.calculate_value_at_risk()
    if var_pypfopt is not None:
        print("\nPortfolio VaR (PyPortfolioOpt Weights):\n", var_pypfopt)