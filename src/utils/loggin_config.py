# src/utils/logging_config.py

import logging
import sys
import os

def setup_logging(level=logging.INFO, log_to_console=True, log_to_file=False, filename='app.log', log_format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'):
    """
    Sets up the logging configuration for the application with more flexibility.

    Args:
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_console (bool): Whether to log to the console (stdout).
        log_to_file (bool): Whether to log to a file.
        filename (str): The name of the log file.
        log_format (str): The format string for log messages.
    """
    handlers = []
    formatter = logging.Formatter(log_format)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_to_file:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # Overrides any existing logging configuration
    )

# Future Enhancements:
# - Support for logging to other destinations (e.g., network, cloud services).
# - Configuration via a file (e.g., JSON, YAML).
# - More sophisticated log rotation and retention policies.