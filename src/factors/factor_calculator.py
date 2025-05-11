# src/factors/factor_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Dict
from src.data_ingestion.arcticdb_loader import ArcticDBLoader  # Import the data loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FactorCalculator:
    """
    A class to calculate various alpha factors using data loaded by ArcticDBLoader.
    """
    def __init__(self, data_loader: ArcticDBLoader):
        """
        Initializes the FactorCalculator with an instance of ArcticDBLoader.

        Args:
            data_loader (ArcticDBLoader): An instance of the ArcticDBLoader class.
        """
        self.data_loader = data_loader

    def calculate_momentum(self, library_name: str, collection_name: str, item_names: List[str],
                           price_field: str = 'close', window: int = 252) -> Optional[pd.DataFrame]:
        """
        Calculates the momentum factor for specified assets.

        Args:
            library_name (str): The name of the library containing price data.
            collection_name (str): The name of the collection containing price data.
            item_names (List[str]): A list of asset symbols/identifiers.
            price_field (str): The name of the price column in the DataFrame.
            window (int): The lookback window in days for calculating momentum.

        Returns:
            Optional[pd.DataFrame]: DataFrame of momentum scores with dates as index and symbols as columns.
        """
        all_prices = self.data_loader.load_multiple_data(library_name, collection_name, item_names)
        if not all_prices:
            logging.warning("No price data loaded for momentum calculation.")
            return None

        momentum_factors = {}
        for symbol, prices in all_prices.items():
            if price_field in prices.columns:
                shifted_prices = prices[price_field].shift(window)
                momentum = prices[price_field] / shifted_prices - 1
                momentum_factors[symbol] = momentum
            else:
                logging.warning(f"Price field '{price_field}' not found for {symbol}.")

        if momentum_factors:
            momentum_df = pd.DataFrame(momentum_factors)
            logging.info(f"Calculated momentum for {len(momentum_df.columns)} assets with window {window}.")
            return momentum_df
        else:
            return None

    def calculate_simple_moving_average_spread(self, library_name: str, collection_name: str, item_names: List[str],
                                               price_field: str = 'close', short_window: int = 20, long_window: int = 50) -> Optional[pd.DataFrame]:
        """
        Calculates the spread between short-term and long-term simple moving averages.

        Args:
            library_name (str): The name of the library containing price data.
            collection_name (str): The name of the collection containing price data.
            item_names (List[str]): A list of asset symbols/identifiers.
            price_field (str): The name of the price column.
            short_window (int): The window for the short-term SMA.
            long_window (int): The window for the long-term SMA.

        Returns:
            Optional[pd.DataFrame]: DataFrame of SMA spreads with dates as index and symbols as columns.
        """
        all_prices = self.data_loader.load_multiple_data(library_name, collection_name, item_names)
        if not all_prices:
            logging.warning("No price data loaded for SMA calculation.")
            return None

        sma_spreads = {}
        for symbol, prices in all_prices.items():
            if price_field in prices.columns:
                short_sma = prices[price_field].rolling(window=short_window).mean()
                long_sma = prices[price_field].rolling(window=long_window).mean()
                sma_spread = short_sma - long_sma
                sma_spreads[symbol] = sma_spread
            else:
                logging.warning(f"Price field '{price_field}' not found for {symbol}.")

        if sma_spreads:
            sma_spread_df = pd.DataFrame(sma_spreads)
            logging.info(f"Calculated SMA spread (short={short_window}, long={long_window}) for {len(sma_spread_df.columns)} assets.")
            return sma_spread_df
        else:
            return None

    # Add more factor calculation methods here, potentially taking library and collection names as arguments
    # to directly load the necessary data.

if __name__ == "__main__":
    # Example Usage (requires a running arcticdb instance with data)
    arctic_uri = "localhost"
    data_loader = ArcticDBLoader(arctic_uri)
    factor_calculator = FactorCalculator(data_loader)

    asset_symbols = ["AAPL", "MSFT"]
    momentum_df = factor_calculator.calculate_momentum("equity_data", "prices", asset_symbols)
    if momentum_df is not None:
        print("\nMomentum Factors:\n", momentum_df.head())

    sma_spread_df = factor_calculator.calculate_simple_moving_average_spread("equity_data", "prices", asset_symbols)
    if sma_spread_df is not None:
        print("\nSMA Spread Factors:\n", sma_spread_df.head())