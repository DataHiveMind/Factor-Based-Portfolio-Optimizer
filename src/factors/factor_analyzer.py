# src/factors/factor_analyzer.py

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from typing import Optional
from src.data_ingestion.arcticdb_loader import ArcticDBLoader  # Import the data loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FactorAnalyzer:
    """
    A class to analyze the properties of calculated factors and their relationship with asset returns.
    """
    def __init__(self, data_loader: ArcticDBLoader):
        """
        Initializes the FactorAnalyzer with an instance of ArcticDBLoader.

        Args:
            data_loader (ArcticDBLoader): An instance of the ArcticDBLoader class.
        """
        self.data_loader = data_loader

    def analyze_factor_correlation(self, factor_df: pd.DataFrame, library_name: str, collection_name: str,
                                   item_names: List[str], return_field: str = 'close') -> Optional[pd.DataFrame]:
        """
        Calculates the correlation between provided factor values and asset returns loaded from arcticdb.

        Args:
            factor_df (pd.DataFrame): DataFrame of factor values (index should align with returns).
            library_name (str): The name of the library containing price data for returns calculation.
            collection_name (str): The name of the collection containing price data.
            item_names (List[str]): A list of asset symbols/identifiers.
            return_field (str): The name of the price column to use for return calculation.

        Returns:
            Optional[pd.DataFrame]: DataFrame of correlations between the factor and each asset return.
        """
        all_prices = self.data_loader.load_multiple_data(library_name, collection_name, item_names)
        if not all_prices:
            logging.warning("No price data loaded for return calculation in factor analysis.")
            return None

        all_returns = {}
        for symbol, prices in all_prices.items():
            if return_field in prices.columns:
                returns = prices[return_field].pct_change().dropna()
                all_returns[symbol] = returns
            else:
                logging.warning(f"Return field '{return_field}' not found for {symbol}.")

        if not all_returns:
            logging.warning("No returns calculated for factor correlation analysis.")
            return None

        aligned_factor, aligned_returns = factor_df.align(pd.DataFrame(all_returns), join='inner')
        if aligned_factor.empty or aligned_returns.empty:
            logging.warning("No overlapping data between factor and returns for correlation analysis.")
            return None

        correlation = aligned_factor.corrwith(aligned_returns)
        logging.info("Calculated factor correlations with returns.")
        return pd.DataFrame(correlation, columns=['Correlation'])

    def perform_univariate_factor_sort(self, factor_df: pd.DataFrame, library_name: str, collection_name: str,
                                       item_names: List[str], return_field: str = 'close', n_portfolios: int = 5,
                                       lookahead_days: int = 1) -> Optional[pd.DataFrame]:
        """
        Performs a univariate factor sort using factor values and future returns loaded from arcticdb.

        Args:
            factor_df (pd.DataFrame): DataFrame of factor values.
            library_name (str): The name of the library containing price data for returns.
            collection_name (str): The name of the collection containing price data.
            item_names (List[str]): A list of asset symbols/identifiers.
            return_field (str): The name of the price column for return calculation.
            n_portfolios (int): The number of portfolios to create.
            lookahead_days (int): The number of days to look ahead for returns.

        Returns:
            Optional[pd.DataFrame]: DataFrame of average forward returns for each factor portfolio.
        """
        all_prices = self.data_loader.load_multiple_data(library_name, collection_name, item_names)
        if not all_prices:
            logging.warning("No price data loaded for factor sort.")
            return None

        forward_returns = {}
        for symbol, prices in all_prices.items():
            if return_field in prices.columns:
                forward_ret = prices[return_field].pct_change(periods=lookahead_days).shift(-lookahead_days).dropna()
                forward_returns[symbol] = forward_ret
            else:
                logging.warning(f"Return field '{return_field}' not found for {symbol}.")

        if not forward_returns:
            logging.warning("No forward returns calculated for factor sort.")
            return None

        aligned_factor, aligned_forward_returns = factor_df.align(pd.DataFrame(forward_returns), join='inner')
        if aligned_factor.empty or aligned_forward_returns.empty:
            logging.warning("No overlapping data between factor and forward returns for factor sort.")
            return None

        portfolio_returns = pd.DataFrame(index=range(1, n_portfolios + 1), columns=aligned_forward_returns.columns)
        for asset in aligned_forward_returns.columns:
            factor_asset = aligned_factor[asset].dropna()
            returns_asset = aligned_forward_returns[asset][factor_asset.index]
            if factor_asset.empty:
                continue
            bins = pd.qcut(factor_asset, q=n_portfolios, labels=False, duplicates='drop')
            average_returns = returns_asset.groupby(bins).mean()
            for i, ret in average_returns.items():
                if i + 1 in portfolio_returns.index:
                    portfolio_returns.loc[i + 1, asset] = ret

        logging.info(f"Performed univariate factor sort into {n_portfolios} portfolios with lookahead {lookahead_days} days.")
        return portfolio_returns.mean(axis=1).to_frame(name='Average Forward Return')

    # Add more factor analysis methods that can directly load data using the ArcticDBLoader instance.

if __name__ == "__main__":
    # Example Usage (requires a running arcticdb instance with data)
    arctic_uri = "localhost"
    data_loader = ArcticDBLoader(arctic_uri)
    factor_calculator = FactorCalculator(data_loader)
    factor_analyzer = FactorAnalyzer(data_loader)

    asset_symbols = ["AAPL", "MSFT"]
    momentum_df = factor_calculator.calculate_momentum("equity_data", "prices", asset_symbols)

    if momentum_df is not None:
        print("\nMomentum Factors for Analysis:\n", momentum_df.head())

        correlation_results = factor_analyzer.analyze_factor_correlation(momentum_df, "equity_data", "prices", asset_symbols)
        if correlation_results is not None:
            print("\nCorrelation of Momentum with Returns:\n", correlation_results)

        factor_sort_results = factor_analyzer.perform_univariate_factor_sort(momentum_df, "equity_data", "prices", asset_symbols)
        if factor_sort_results is not None:
            print("\nUnivariate Momentum Factor Sort Results (Forward Returns):\n", factor_sort_results)