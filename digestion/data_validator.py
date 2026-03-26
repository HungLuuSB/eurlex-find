"""
data_validator.py

This module provides an isolated, memory-safe mechanism to validate the dataset.
It is used to verify column structures, data types, and textual contents of the
EUR-LEX dataset directly through the terminal output.
"""

from numpy import log
import pandas as pd
from pathlib import Path
from typing import List
import config

# Initialize the logger adhering strictly to the centralized configuration
from utils.logger import get_logger

logger = get_logger(__name__)


COLUMN_DTYPES = {
    "date_publication": "str",
    "temporal_status": "str",
    "oeil_link": "str",
}


def validate_dataset(file_path: Path = config.RAW_DATA_PATH) -> None:
    """
    Reads a small chunk of the dataset and logs its metadata and sample content.

    Steps performed:
    1. Checks if the file exists.
    2. Reads only the first N rows using the 'nrows' parameter.
    3. Logs the columns, data types, and non-null counts.
    4. Logs a truncated sample of the text and EUROVOC columns for visual inspection.

    Args:
        file_path (Path): The absolute path to the raw CSV file.
        sample_size (int): The number of rows to read into memory.
    """
    logger.info(
        "Initiating Exploratory Data Analysis (EDA) viewer for the whole dataset."
    )

    if not file_path.exists():
        logger.error(f"Cannot perform EDA. File missing at {file_path}")
        return

    try:
        df: pd.DataFrame = pd.read_csv(
            file_path,
            low_memory=False,
            dtype={
                "date_publication": "str",
                "temporal_status": "str",
                "oeil_link": "str",
            },
        )

        logger.info("--- Dataset Metadata ---")

        logger.info(df.info())

        print(df.describe())

        print(df.head(5))

        logger.info("Dataset Validation completed.")

    except Exception as e:
        logger.error(f"An error occurred during EDA viewing: {str(e)}")


if __name__ == "__main__":
    validate_dataset()
