"""
logger.py

This module provides a centralized logging configuration for the entire
EUR-LEX Information Retrieval system. It ensures all terminal outputs
adhere strictly to the format defined in the global configuration.
"""

import logging
import sys
import config


def get_logger(module_name: str) -> logging.Logger:
    """
    Instantiates and configures a logger for a specific module.

    This function reads the centralized format and level from config.py
    and attaches a StreamHandler to ensure outputs are directed to the standard output.
    It prevents duplicate logging if the logger is requested multiple times.

    Args:
        module_name (str): The name of the module requesting the logger (typically __name__).

    Returns:
        logging.Logger: A configured standard library logger instance.
    """
    logger: logging.Logger = logging.getLogger(module_name)

    # Set the threshold level (e.g., INFO, DEBUG, ERROR) from the central config
    logger.setLevel(config.LOG_LEVEL)

    # Step 1: Prevent adding multiple handlers if the logger already exists
    if not logger.handlers:
        # Step 2: Create a console handler directing output to standard output (terminal)
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

        # Step 3: Create a formatter using the strict string templates from config.py
        formatter: logging.Formatter = logging.Formatter(
            fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT
        )

        # Step 4: Attach the formatter to the handler, and the handler to the logger
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Step 5: Prevent the logger from propagating messages to the root logger
    # This stops duplicate prints in certain complex module hierarchies
    logger.propagate = False

    return logger


if __name__ == "__main__":
    # Test execution block to verify the formatting works as intended
    test_logger = get_logger(__name__)
    test_logger.info(
        "The centralized logging utility has been successfully initialized."
    )
