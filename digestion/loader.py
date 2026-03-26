"""
loader.py

This module handles the ingestion and initial cleaning of the EUR-LEX dataset.
It enforces memory constraints by loading only essential columns and standardizes
the text data for downstream Information Retrieval tasks.
"""

import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path
import config

# Initialize the logger adhering strictly to the centralized configuration
from utils.logger import get_logger

logger = get_logger(__name__)


def load_and_clean_dataset(
    file_path: Path = config.RAW_DATA_PATH, columns_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Validates, loads, and cleans the EUR-LEX dataset from a CSV file.

    Steps performed:
    1. Validates the existence of the file.
    2. Loads only the specified columns to minimize RAM consumption.
    3. Drops any rows missing the core text data ('act_raw_text').
    4. Safely fills missing metadata with empty strings and converts to lowercase.
    5. Parses the EUROVOC string into a Python list of tags safely.

    Args:
        file_path (Path): The absolute path to the raw CSV file.
        columns_to_keep (Optional[List[str]]): The specific columns to extract.

    Returns:
        pd.DataFrame: The cleaned and normalized pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file is not found at the configured path.
    """
    if columns_to_keep is None:
        columns_to_keep = ["Act_type", "Act_name", "CELEX", "act_raw_text", "EUROVOC"]

    logger.info(f"Initiating data loading sequence from: {file_path}")

    # Step 1: Validate file existence
    if not file_path.exists():
        error_msg: str = (
            f"Dataset not found at {file_path}. "
            f"Please ensure 'EurLex_all.csv' is in the 'data/raw/' directory."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("File located. Loading specified columns into memory...")

    # Step 2: Load only specific columns
    try:
        df: pd.DataFrame = pd.read_csv(
            file_path, usecols=columns_to_keep, low_memory=False
        )
        logger.info(f"Successfully loaded {len(df)} rows from the dataset.")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {str(e)}")
        raise

    # Step 3: Drop rows without raw text
    initial_row_count: int = len(df)
    df = df.dropna(subset=["act_raw_text"])
    dropped_rows: int = initial_row_count - len(df)
    logger.info(
        f"Dropped {dropped_rows} rows due to missing 'act_raw_text'. Remaining: {len(df)}."
    )

    # Step 4: Safely handle NaN values and lowercase all string fields
    logger.info(
        "Applying NaN filling and lowercase normalization to all selected fields."
    )
    for col in columns_to_keep:
        if col in df.columns:
            # fillna("") guarantees no floats remain before converting to string
            df[col] = df[col].fillna("").astype(str).str.lower()

    # Step 5: Parse EUROVOC labels into lists using a safe parser function
    if "EUROVOC" in df.columns:
        logger.info("Parsing EUROVOC string field into Python lists.")

        def safe_parse_tags(val: str) -> List[str]:
            """Safely splits the EUROVOC string, ignoring empty or 'nan' strings."""
            if not val or val == "nan":
                return []
            return [tag.strip() for tag in val.split(";") if tag.strip()]

        # Initialize tqdm for pandas operations
        tqdm.pandas(desc="Parsing EUROVOC Labels")

        # Utilize progress_apply instead of apply to track the row-by-row execution
        df["EUROVOC"] = df["EUROVOC"].progress_apply(safe_parse_tags)
    else:
        logger.info("'EUROVOC' column not in loaded data; skipping parsing.")

    logger.info("Data loading and initial cleaning completed successfully.")

    return df


if __name__ == "__main__":
    # Test execution block
    df_cleaned = load_and_clean_dataset()
