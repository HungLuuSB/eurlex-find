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
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
PROCESSED_CORPUS_PATH: Path = PROCESSED_DATA_DIR / "processed_corpus.pkl"
INDEX_DIR: Path = DATA_DIR / "indices"

# Evaluation Data Paths (NEW)
EVALUATION_DATA_DIR: Path = DATA_DIR / "evaluation"
EURLEX_ID_MAPPINGS_PATH: Path = EVALUATION_DATA_DIR / "eurlex_ID_mappings.csv"
EURLEX_EUROVOC_QRELS_PATH: Path = EVALUATION_DATA_DIR / "id2class_eurlex_eurovoc.qrels"

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
# The maximum character limit per document to prevent Out-Of-Memory (OOM) crashes.
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
# 6. HYBRID RETRIEVAL HYPERPARAMETERS (NEW)
# ==========================================
# Weight for BM25 score in hybrid ranking (0.0 to 1.0)
BM25_WEIGHT: float = 0.3
# Weight for SVC classifier score in hybrid ranking (0.0 to 1.0)
SVC_WEIGHT: float = 0.7
# Number of candidates to retrieve before re-ranking
TOP_K_CANDIDATES: int = 1000
