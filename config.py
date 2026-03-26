"""
config.py

This module contains all centralized hyperparameters, path configurations,
and systemic constants for the EUR-LEX Information Retrieval pipeline.
"""

import logging
from pathlib import Path
from typing import Tuple, List

# ==========================================
# 1. DIRECTORY AND PATH RESOLUTION
# ==========================================
# Resolves the absolute path to the root directory of the project
BASE_DIR: Path = Path(__file__).resolve().parent

# Data Paths
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "eurlex_dataset.csv"
PROCESSED_DATA_DIR: Path = (
    DATA_DIR / "processed"
)  # Add this line below PROCESSED_DATA_DIR
PROCESSED_CORPUS_PATH: Path = PROCESSED_DATA_DIR / "processed_corpus.pkl"
INDEX_DIR: Path = DATA_DIR / "indices"

# Model Paths
MODELS_DIR: Path = BASE_DIR / "models"
SVC_MODELS_DIR: Path = MODELS_DIR / "svc_classifiers"
EMBEDDINGS_DIR: Path = MODELS_DIR / "embeddings"

# Output paths for the core search engine data structures
INVERTED_INDEX_PATH: Path = INDEX_DIR / "inverted_index.pkl"
DOCUMENT_METADATA_PATH: Path = INDEX_DIR / "document_metadata.pkl"

# Artifact paths for the classification pipeline
TFIDF_VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_BINARIZER_PATH: Path = MODELS_DIR / "label_binarizer.pkl"
MULTI_LABEL_SVC_PATH: Path = SVC_MODELS_DIR / "one_vs_rest_svc.pkl"
# ==========================================
# 2. LOGGING CONFIGURATION
# ==========================================
# Ensures every piece of output is logged using the exact same format
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: int = logging.INFO

# ==========================================
# 3. PREPROCESSING HYPERPARAMETERS
# ==========================================
SPACY_MODEL: str = "en_core_web_sm"
MIN_TOKEN_LENGTH: int = 2
# Defines characters to keep during alphanumeric filtering
ALLOWED_CHARS_REGEX: str = r"[^a-zA-Z0-9\s]"  # The maximum number of rows to load into RAM at one time during preprocessing
PREPROCESSING_CHUNK_SIZE: int = 5000

SPACY_N_PROCESS: int = 1
# The maximum character limit per document to prevent Out-Of-Memory (OOM) crashes.
# 100,000 characters is approximately 15,000 words, which is more than sufficient
# for BM25 to capture the core semantic vocabulary of a legal document.
MAX_CHAR_LENGTH: int = 100000

# ==========================================
# 4. CLASSIFICATION HYPERPARAMETERS (SVC & TF-IDF)
# ==========================================
# We use Unigrams and Bigrams to capture compound legal terms
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
# Limit features to prevent memory overflow on 91,000 documents
TFIDF_MAX_FEATURES: int = 50000
# The regularization parameter for the Support Vector Classifier
SVC_C_PARAMETER: float = 1.0
# The total number of unique EURO-VOC labels expected in the dataset
TOTAL_EUROVOC_LABELS: int = 100

# ==========================================
# 5. RETRIEVAL HYPERPARAMETERS (BM25)
# ==========================================
# k1 controls non-linear term frequency saturation
BM25_K1: float = 1.5
# b controls document length normalization
BM25_B: float = 0.75

# ==========================================
# 6. SEMANTIC RERANKING HYPERPARAMETERS
# ==========================================
# The specific transformer model fine-tuned on legal text
TRANSFORMER_MODEL_NAME: str = "nlpaueb/legal-bert-base-uncased"
# The number of top documents retrieved by BM25 to pass to the semantic reranker
TOP_K_CANDIDATES: int = 1000
# The final number of documents returned to the user
FINAL_TOP_K_RESULTS: int = 10

# ==========================================
# 7. BOOLEAN QUERY PREFIXES
# ==========================================
# Standardized prefixes for the Application Domain layer to parse
PREFIX_NOT: str = "-"
PREFIX_OR: str = "|"
