"""
indexer.py

This module constructs the Inverted Index and Document Metadata structures required
for Boolean retrieval and BM25 ranking. It reads the preprocessed corpus and serializes
the mathematical mappings to disk.
"""

import pandas as pd
import pickle
from collections import Counter
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
import config
from utils.logger import get_logger

# Initialize the centralized logger
logger = get_logger(__name__)


def build_and_save_index(
    corpus_path: Path = config.PROCESSED_CORPUS_PATH,
    index_path: Path = config.INVERTED_INDEX_PATH,
    metadata_path: Path = config.DOCUMENT_METADATA_PATH,
) -> None:
    """
    Constructs the Inverted Index and BM25 Document Metadata from the processed corpus.

    Steps performed:
    1. Validates the existence of the preprocessed corpus file.
    2. Loads the serialized pandas DataFrame into memory.
    3. Iterates through the corpus to calculate Term Frequencies (TF) and Document Lengths.
    4. Populates the Inverted Index mapping: {term -> {doc_id -> frequency}}.
    5. Calculates the Average Document Length (avgdl) for BM25 normalization.
    6. Serializes both data structures to disk using Python Pickle for rapid loading.

    Args:
        corpus_path (Path): Path to the input preprocessed corpus.
        index_path (Path): Destination path for the Inverted Index pickle file.
        metadata_path (Path): Destination path for the Document Metadata pickle file.

    Raises:
        FileNotFoundError: If the processed corpus does not exist.
    """
    logger.info(f"Initiating index construction from corpus: {corpus_path}")

    # Step 1: Validate input
    if not corpus_path.exists():
        error_msg: str = (
            f"Processed corpus not found at {corpus_path}. Run preprocessor.py first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Step 2: Load the corpus
    logger.info("Loading preprocessed corpus into memory...")
    try:
        df: pd.DataFrame = pd.read_pickle(corpus_path)  # type: ignore
        logger.info(f"Successfully loaded {len(df)} documents.")
    except Exception as e:
        logger.error(f"Failed to read corpus: {str(e)}")
        raise

    # Initialize the mathematical structures
    # inverted_index format: {"term": {"CELEX_1": 5, "CELEX_2": 1}}
    inverted_index: Dict[str, Dict[str, int]] = {}

    # document_lengths format: {"CELEX_1": 1500, "CELEX_2": 350}
    document_lengths: Dict[str, int] = {}
    total_corpus_length: int = 0

    logger.info("Constructing Inverted Index and calculating BM25 metadata metrics.")

    # Step 3 & 4: Iterate through documents using tqdm for progressive tracking
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Building Index"):
        doc_id: str = str(row["CELEX"])
        tokens: List[str] = row["processed_tokens"]  # type: ignore

        # Calculate and store the absolute document length (|d|)
        doc_length: int = len(tokens)
        document_lengths[doc_id] = doc_length
        total_corpus_length += doc_length

        # Count term frequencies for the current document efficiently using collections.Counter
        term_frequencies: Counter = Counter(tokens)

        # Populate the global inverted index
        for term, frequency in term_frequencies.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = frequency

    # Step 5: Calculate Average Document Length (avgdl)
    total_documents: int = len(df)
    avgdl: float = total_corpus_length / total_documents if total_documents > 0 else 0.0

    document_metadata: Dict[str, Any] = {
        "lengths": document_lengths,
        "avgdl": avgdl,
        "total_documents": total_documents,
    }

    logger.info(
        f"Index construction complete. Unique terms identified: {len(inverted_index)}."
    )
    logger.info(f"Corpus Average Document Length (avgdl): {avgdl:.2f} tokens.")

    # Step 6: Serialize to disk
    logger.info("Serializing data structures to disk...")
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Inverted Index saved to: {index_path}")

        with open(metadata_path, "wb") as f:
            pickle.dump(document_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Document Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to serialize index data: {str(e)}")
        raise

    logger.info("Indexer module execution finished flawlessly.")


if __name__ == "__main__":
    # Test execution block
    build_and_save_index()
