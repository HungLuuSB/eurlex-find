"""
eda_viewer.py

This module provides an isolated, memory-safe mechanism to preview the dataset.
It is used to verify column structures, data types, and textual contents of the
EUR-LEX dataset directly through the terminal output.
"""

import pandas as pd
from pathlib import Path
from typing import List
import config

# Initialize the logger adhering strictly to the centralized configuration
from utils.logger import get_logger

logger = get_logger(__name__)


def view_dataset_sample(
    file_path: Path = config.RAW_DATA_PATH, sample_size: int = 5
) -> None:
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
        f"Initiating Exploratory Data Analysis (EDA) viewer for top {sample_size} rows."
    )

    if not file_path.exists():
        logger.error(f"Cannot perform EDA. File missing at {file_path}")
        return

    try:
        # Step 2: Read only the first N rows to save memory
        df_sample: pd.DataFrame = pd.read_csv(file_path, nrows=sample_size)

        # Step 3: Log metadata
        logger.info("--- Dataset Metadata ---")
        logger.info(f"Total Columns Detected: {len(df_sample.columns)}")

        columns_info: List[str] = [
            f"{col} ({dtype})"
            for col, dtype in zip(df_sample.columns, df_sample.dtypes)
        ]
        logger.info(f"Columns and Types: {', '.join(columns_info)}")

        # Step 4: Log specific samples
        logger.info("--- Content Samples ---")
        for index, row in df_sample.iterrows():
            logger.info(f"Row {index} | CELEX: {row.get('CELEX', 'N/A')}")

            # Truncate raw text for the log output to maintain terminal readability
            raw_text: str = str(row.get("act_raw_text", "N/A"))
            truncated_text: str = (
                raw_text[:150] + "..." if len(raw_text) > 150 else raw_text
            )
            logger.info(f"Row {index} | Text Sample: {truncated_text}")

            logger.info(f"Row {index} | EUROVOC: {row.get('EUROVOC', 'N/A')}")
            logger.info("-" * 40)

        logger.info("EDA viewing sequence completed.")

    except Exception as e:
        logger.error(f"An error occurred during EDA viewing: {str(e)}")


if __name__ == "__main__":
    # Test execution block
    view_dataset_sample()
