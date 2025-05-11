# src/utils/time_utils.py

import pandas as pd
import logging
from typing import Optional, Union
import pytz
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

def convert_timezone(timestamp: Union[pd.Timestamp, pd.Series, pd.DataFrame], to_tz: str) -> Optional[Union[pd.Timestamp, pd.Series, pd.DataFrame]]:
    """
    Converts a pandas Timestamp, Series, or DataFrame index to a different timezone.

    Args:
        timestamp (Union[pd.Timestamp, pd.Series, pd.DataFrame]): The timestamp(s) to convert.
        to_tz (str): The target timezone (e.g., 'US/Eastern', 'UTC').

    Returns:
        Optional[Union[pd.Timestamp, pd.Series, pd.DataFrame]]: The converted timestamp(s), or None if conversion fails.
    """
    try:
        if isinstance(timestamp, pd.Timestamp):
            converted = timestamp.tz_convert(to_tz)
            logging.info(f"Converted timestamp to timezone: {to_tz}")
            return converted
        elif isinstance(timestamp, pd.Series):
            converted = timestamp.tz_convert(to_tz)
            logging.info(f"Converted Series timezone to: {to_tz}")
            return converted
        elif isinstance(timestamp, pd.DataFrame):
            converted = timestamp.tz_convert(to_tz)
            logging.info(f"Converted DataFrame timezone to: {to_tz}")
            return converted
        else:
            logging.error("Input is not a pandas Timestamp, Series, or DataFrame.")
            return None
    except pytz.exceptions.UnknownTimeZoneError as e:
        logging.error(f"Unknown timezone: {to_tz}")
        return None
    except Exception as e:
        logging.error(f"Error converting timezone: {e}")
        return None

def generate_date_range(start_date: str, end_date: str, freq: str = 'D', timezone: Optional[str] = None) -> Optional[pd.DatetimeIndex]:
    """
    Generates a pandas DatetimeIndex with optional timezone specification.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        freq (str): The frequency of the date range (e.g., 'D' for daily, 'B' for business days).
        timezone (Optional[str]): The timezone for the DatetimeIndex (e.g., 'UTC').

    Returns:
        Optional[pd.DatetimeIndex]: The generated DatetimeIndex, or None if generation fails.
    """
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq, tz=timezone)
        logging.info(f"Generated date range from {start_date} to {end_date} with frequency '{freq}' and timezone '{timezone}'.")
        return date_range
    except Exception as e:
        logging.error(f"Error generating date range: {e}")
        return None

def get_current_timestamp(timezone: Optional[str] = None) -> pd.Timestamp:
    """
    Returns the current pandas Timestamp, optionally in a specified timezone.

    Args:
        timezone (Optional[str]): The desired timezone (e.g., 'UTC', 'US/Eastern').
                                  If None, returns the local timezone.

    Returns:
        pd.Timestamp: The current timestamp.
    """
    now = pd.Timestamp(datetime.now())
    if timezone:
        try:
            now = now.tz_localize(pytz.timezone(str(datetime.now().astimezone().tzinfo))).tz_convert(timezone)
            logging.info(f"Got current timestamp in timezone: {timezone}")
        except pytz.exceptions.UnknownTimeZoneError as e:
            logging.error(f"Unknown timezone: {timezone}. Returning local time.")
    else:
        logging.info("Got current timestamp in local timezone.")
    return now

def format_timestamp(timestamp: pd.Timestamp, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Formats a pandas Timestamp into a specified string format.

    Args:
        timestamp (pd.Timestamp): The timestamp to format.
        format_str (str): The desired format string (e.g., '%Y-%m-%d', '%H:%M').

    Returns:
        str: The formatted timestamp string.
    """
    try:
        formatted = timestamp.strftime(format_str)
        logging.info(f"Formatted timestamp to: '{formatted}'")
        return formatted
    except Exception as e:
        logging.error(f"Error formatting timestamp: {e}")
        return str(timestamp) # Return default string representation on error

# Future Enhancements:
# - Functions for calculating time differences.
# - Utilities for working with trading calendars (e.g., checking if a date is a trading day).
# - Functions for converting between different time frequencies.