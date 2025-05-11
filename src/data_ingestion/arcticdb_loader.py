# src/data_ingestion/arcticdb_loader.py

import arcticdb
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArcticDBLoader:
    """
    A class to handle loading financial data from an arcticdb instance.
    """
    def __init__(self, uri: str):
        """
        Initializes the ArcticDBLoader with the connection URI.

        Args:
            uri (str): The connection URI for the arcticdb instance.
        """
        self.uri = uri
        self.conn: Optional[arcticdb.Arctic] = None
        try:
            self.conn = arcticdb.connect(self.uri)
            logging.info(f"Successfully connected to arcticdb at {self.uri}")
        except arcticdb.exceptions.ArcticException as e:
            logging.error(f"Error connecting to arcticdb at {self.uri}: {e}")
            self.conn = None

    def load_collection(self, library_name: str, collection_name: str) -> Optional[arcticdb.Collection]:
        """
        Loads a specific collection from the given library.

        Args:
            library_name (str): The name of the library.
            collection_name (str): The name of the collection.

        Returns:
            Optional[arcticdb.Collection]: The loaded arcticdb Collection object,
                                           or None if the library or collection is not found
                                           or if the connection failed.
        """
        if not self.conn:
            logging.error("ArcticDB connection is not established.")
            return None

        try:
            library = self.conn.get_library(library_name)
            collection = library.get_collection(collection_name)
            logging.info(f"Successfully loaded collection '{collection_name}' from library '{library_name}'.")
            return collection
        except arcticdb.exceptions.LibraryNotFound:
            logging.error(f"Library '{library_name}' not found.")
            return None
        except arcticdb.exceptions.CollectionNotFound:
            logging.error(f"Collection '{collection_name}' not found in library '{library_name}'.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading collection '{collection_name}': {e}")
            return None

    def load_data(self, library_name: str, collection_name: str, item_name: str,
                  start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Loads data for a specific item from a collection into a pandas DataFrame.

        Args:
            library_name (str): The name of the library.
            collection_name (str): The name of the collection.
            item_name (str): The name of the item (symbol or identifier).
            start_date (Optional[datetime]): The start date for the data (inclusive). Defaults to None (all data).
            end_date (Optional[datetime]): The end date for the data (inclusive). Defaults to None (all data).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                     or None if there was an error.
        """
        collection = self.load_collection(library_name, collection_name)
        if not collection:
            return None

        try:
            item = collection.read(item_name, from_date=start_date, to_date=end_date)
            if item and item.data is not None:
                logging.info(f"Successfully loaded data for '{item_name}' from '{collection_name}'.")
                return item.data
            else:
                logging.warning(f"No data found for '{item_name}' in '{collection_name}' within the specified date range.")
                return None
        except arcticdb.exceptions.ItemNotFound:
            logging.error(f"Item '{item_name}' not found in collection '{collection_name}'.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading data for '{item_name}': {e}")
            return None

    def load_multiple_data(self, library_name: str, collection_name: str, item_names: List[str],
                           start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[dict[str, pd.DataFrame]]:
        """
        Loads data for multiple items from a collection into a dictionary of pandas DataFrames.

        Args:
            library_name (str): The name of the library.
            collection_name (str): The name of the collection.
            item_names (List[str]): A list of item names (symbols or identifiers).
            start_date (Optional[datetime]): The start date for the data (inclusive). Defaults to None (all data).
            end_date (Optional[datetime]): The end date for the data (inclusive). Defaults to None (all data).

        Returns:
            Optional[dict[str, pd.DataFrame]]: A dictionary where keys are item names and values are
                                               pandas DataFrames containing the loaded data, or None if there was an error.
        """
        collection = self.load_collection(library_name, collection_name)
        if not collection:
            return None

        all_data = {}
        for item_name in item_names:
            data = self.load_data(library_name, collection_name, item_name, start_date, end_date)
            if data is not None:
                all_data[item_name] = data
            else:
                logging.warning(f"Could not load data for '{item_name}'. Skipping.")

        if all_data:
            logging.info(f"Successfully loaded data for {len(all_data)} items from '{collection_name}'.")
            return all_data
        else:
            logging.warning(f"No data could be loaded for the specified items from '{collection_name}'.")
            return None

# Example usage (to be used in your notebooks or main script):
if __name__ == "__main__":
    # Replace with your actual arcticdb URI
    arcticdb_uri = "localhost"
    loader = ArcticDBLoader(arcticdb_uri)

    # Example loading a single item
    macro_data = loader.load_data("macro_data", "economic_indicators", "macro_indicators")
    if macro_data is not None:
        print("\nMacro Data:")
        print(macro_data.head())

    # Example loading multiple asset prices
    asset_symbols = ["AAPL", "MSFT"]
    price_data = loader.load_multiple_data("equity_data", "prices", asset_symbols,
                                            start_date=datetime(2023, 1, 1), end_date=datetime(2024, 1, 1))
    if price_data:
        print("\nAsset Price Data:")
        for symbol, df in price_data.items():
            print(f"\n{symbol}:\n{df.head()}")