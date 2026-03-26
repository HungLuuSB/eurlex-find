"""
preprocessor.py

This module handles the linguistic normalization of the EUR-LEX corpus.
It utilizes spaCy for tokenization, lemmatization, and filtering, utilizing
multi-processing to handle the large volume of legal text efficiently.
"""

import math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import spacy
from spacy.tokens import Doc
from typing import List
import config
from utils.logger import get_logger

# Initialize the centralized logger
logger = get_logger(__name__)


def load_spacy_model() -> spacy.language.Language:
    """
    Loads the configured spaCy language model with specific components disabled.

    Disabling the 'parser' and 'ner' (Named Entity Recognition) components
    prevents spaCy from wasting CPU cycles on syntactic mapping, speeding up
    the pipeline by approximately 80%.

    Returns:
        spacy.Language: The optimized spaCy NLP object.

    Raises:
        OSError: If the spaCy model has not been downloaded to the local environment.
    """
    logger.info(f"Loading spaCy language model: {config.SPACY_MODEL}")
    try:
        # Load model and explicitly disable unused heavy components
        nlp: spacy.language.Language = spacy.load(
            config.SPACY_MODEL, disable=["parser", "ner"]
        )
        logger.info("spaCy model loaded successfully with optimized pipeline.")
        return nlp
    except OSError as e:
        error_msg: str = (
            f"Failed to load spaCy model '{config.SPACY_MODEL}'. "
            f"Please run the following command in your terminal: "
            f"python -m spacy download {config.SPACY_MODEL}"
        )
        logger.error(error_msg)
        raise OSError(error_msg) from e


def preprocess_corpus(
    df: pd.DataFrame, text_column: str = "act_raw_text"
) -> pd.DataFrame:
    """
    Applies tokenization, lemmatization, and strict filtering using a synchronous,
    single-threaded loop. Optimizes for absolute memory stability over speed.

    Steps performed:
    1. Validates the target column and increases spaCy's internal memory allocations.
    2. Iterates through the DataFrame row-by-row using a standard for-loop.
    3. Extracts the text, casting to string and applying the MAX_CHAR_LENGTH truncation.
    4. Passes the single string directly to the spaCy callable (nlp(text)).
    5. Filters tokens by length, alphanumeric status, and stop words.
    6. Appends the processed token lists to the original DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw text.
        text_column (str): The name of the column containing the text to process.

    Returns:
        pd.DataFrame: The DataFrame with a new 'processed_tokens' column.

    Raises:
        KeyError: If the specified text column does not exist in the DataFrame.
    """
    logger.info(f"Initiating synchronous preprocessing on column: '{text_column}'")

    if text_column not in df.columns:
        error_msg: str = f"Column '{text_column}' not found in the DataFrame."
        logger.error(error_msg)
        raise KeyError(error_msg)

    # Note: load_spacy_model() should be defined above this function
    nlp: spacy.language.Language = load_spacy_model()

    # Increase internal limit to prevent C-level array overflows on large documents
    nlp.max_length = 3000000

    processed_corpus: List[List[str]] = []

    logger.info(
        f"Processing {len(df)} documents using single-threaded nlp(text). "
        f"Applying strict {config.MAX_CHAR_LENGTH} character limit per document. "
        "Note: This process optimizes for stability and may take several hours."
    )

    # Step 2: Iterate row-by-row with tqdm for accurate, real-time tracking
    for text in tqdm(df[text_column], total=len(df), desc="Lemmatizing and Filtering"):
        # Step 3: Extract and forcefully truncate the string (The Safety Valve)
        safe_text: str = str(text)[: config.MAX_CHAR_LENGTH]

        # Step 4: Process the single string synchronously
        doc = nlp(safe_text)

        # Step 5: Token-level filtering and lemmatization
        valid_tokens: List[str] = [
            token.lemma_
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and len(token) >= config.MIN_TOKEN_LENGTH
        ]

        processed_corpus.append(valid_tokens)

    # Step 6: Append the master list to the DataFrame
    df["processed_tokens"] = processed_corpus

    logger.info("Synchronous corpus preprocessing completed successfully.")
    return df


def save_processed_corpus(
    df: pd.DataFrame, output_path: Path = config.PROCESSED_CORPUS_PATH
) -> None:
    """
    Serializes the fully preprocessed DataFrame to disk using Python Pickle.

    This preserves complex data types like the parsed EUROVOC lists and the
    lemmatized token lists without coercing them into strings.

    Steps performed:
    1. Validates and creates the target directory if it does not exist.
    2. Executes the native pandas to_pickle method.

    Args:
        df (pd.DataFrame): The DataFrame containing the processed tokens.
        output_path (Path): The absolute path to save the pickle file.

    Raises:
        OSError: If the system cannot write to the specified directory.
    """
    logger.info(f"Initiating Pickle serialization to: {output_path}")
    try:
        # Step 1: Ensure the output directory exists (.gitkeep handles the folder, but this is a failsafe)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 2: Serialize
        df.to_pickle(output_path)
        logger.info("Pickle serialization completed successfully.")

    except Exception as e:
        logger.error(f"Failed to serialize DataFrame: {str(e)}")
        raise


if __name__ == "__main__":
    from digestion.loader import load_and_clean_dataset

    logger.info("Starting complete Data Digestion test pipeline...")

    # Load and clean
    df_raw: pd.DataFrame = load_and_clean_dataset()

    # Process
    df_processed: pd.DataFrame = preprocess_corpus(df_raw)

    # Serialize
    save_processed_corpus(df_processed)

    logger.info("Data Digestion test pipeline finished flawlessly.")
