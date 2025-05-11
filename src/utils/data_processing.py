# src/utils/data_processing.py

import pandas as pd
import numpy as np
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def calculate_returns(price_data: pd.DataFrame, method: str = 'pct_change', fill_method: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Calculates asset returns from price data, with optional handling of missing values
    before calculation.

    Args:
        price_data (pd.DataFrame): DataFrame of asset prices (index: dates, columns: symbols).
        method (str): The method for calculating returns ('pct_change' or 'log').
        fill_method (Optional[str]): Method to fill missing prices before return calculation
                                     ('ffill', 'bfill', 'linear', None).

    Returns:
        Optional[pd.DataFrame]: DataFrame of asset returns.
    """
    if price_data is None or price_data.empty:
        logging.warning("Price data is empty or None, cannot calculate returns.")
        return None

    processed_prices = price_data.copy()
    if fill_method:
        processed_prices = handle_missing_data(processed_prices, method='fillna', fill_value=None, fill_method=fill_method)

    try:
        if method == 'pct_change':
            returns = processed_prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(processed_prices / processed_prices.shift(1)).dropna()
        else:
            logging.error(f"Invalid return calculation method: '{method}'. Choose 'pct_change' or 'log'.")
            return None
        logging.info(f"Calculated returns using '{method}' method (missing filled with '{fill_method}').")
        return returns
    except Exception as e:
        logging.error(f"Error calculating returns: {e}")
        return None

def handle_missing_data(df: pd.DataFrame, method: str = 'dropna', fill_value=None, fill_method: str = 'ffill', **kwargs) -> Optional[pd.DataFrame]:
    """
    Handles missing data in a DataFrame with more flexible filling options.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method for handling missing data ('dropna', 'fillna', 'interpolate').
        fill_value: The value to use for 'fillna' (if applicable).
        fill_method (str): The method to use for 'fillna' ('ffill', 'bfill', 'linear').
        **kwargs: Additional keyword arguments for the chosen method.

    Returns:
        Optional[pd.DataFrame]: DataFrame with missing data handled.
    """
    if df is None:
        logging.warning("Input DataFrame is None.")
        return None
    try:
        if method == 'dropna':
            cleaned_df = df.dropna(**kwargs)
            logging.info("Dropped rows with missing data.")
        elif method == 'fillna':
            if fill_value is not None:
                cleaned_df = df.fillna(value=fill_value, **kwargs)
                logging.info(f"Filled missing data with value: {fill_value}.")
            elif fill_method in ['ffill', 'bfill', 'linear']:
                cleaned_df = df.fillna(method=fill_method, **kwargs)
                logging.info(f"Filled missing data using '{fill_method}' method.")
            else:
                logging.error("Value or valid fill method ('ffill', 'bfill', 'linear') must be provided for 'fillna' method.")
                return None
        elif method == 'interpolate':
            cleaned_df = df.interpolate(**kwargs)
            logging.info("Interpolated missing data.")
        else:
            logging.error(f"Invalid missing data handling method: '{method}'. Choose 'dropna', 'fillna', or 'interpolate'.")
            return None
        return cleaned_df
    except Exception as e:
        logging.error(f"Error handling missing data: {e}")
        return None

def align_dataframes(df_list: List[pd.DataFrame], join: str = 'inner') -> Optional[List[pd.DataFrame]]:
    """
    Aligns a list of DataFrames based on their index.

    Args:
        df_list (List[pd.DataFrame]): A list of DataFrames to align.
        join (str): The type of join to perform ('inner', 'outer', 'left', 'right').

    Returns:
        Optional[List[pd.DataFrame]]: A list containing the aligned DataFrames,
                                       or None if alignment fails or the list is empty.
    """
    if not df_list:
        logging.warning("Input DataFrame list is empty.")
        return None
    try:
        aligned_dfs = []
        first_df = df_list[0]
        aligned_dfs.append(first_df)
        for i in range(1, len(df_list)):
            aligned_df1, aligned_df2 = first_df.align(df_list[i], join=join, axis=0)
            aligned_dfs[0] = aligned_df1
            aligned_dfs.append(aligned_df2)
        logging.info(f"Aligned {len(df_list)} DataFrames using '{join}' join.")
        return aligned_dfs
    except Exception as e:
        logging.error(f"Error aligning DataFrames: {e}")
        return None

def standardize_data(series: pd.Series) -> Optional[pd.Series]:
    """
    Standardizes a pandas Series (removes mean and scales to unit variance).
    """
    if series is None or series.empty:
        logging.warning("Input Series is empty or None, cannot standardize.")
        return None
    try:
        standardized_series = (series - series.mean()) / series.std()
        logging.info("Standardized the input Series.")
        return standardized_series
    except Exception as e:
        logging.error(f"Error standardizing Series: {e}")
        return None

def normalize_data(series: pd.Series, min_val: float = 0, max_val: float = 1) -> Optional[pd.Series]:
    """
    Normalizes a pandas Series to a specified range (default [0, 1]).
    """
    if series is None or series.empty:
        logging.warning("Input Series is empty or None, cannot normalize.")
        return None
    try:
        numerator = series - series.min()
        denominator = series.max() - series.min()
        if denominator == 0:
            logging.warning("Range of Series is zero, cannot normalize.")
            return series  # Return original to avoid division by zero
        normalized_series = min_val + (max_val - min_val) * numerator / denominator
        logging.info(f"Normalized the input Series to the range [{min_val}, {max_val}].")
        return normalized_series
    except Exception as e:
        logging.error(f"Error normalizing Series: {e}")
        return None

# Future Enhancements:
# - Function to resample data to different frequencies.
# - Function for rolling window calculations (mean, std, etc.).
# - More advanced outlier detection and handling methods.