"""
search_engine.py

Hybrid retrieval with pre-computed label index and two search modes.
"""

import pickle
import math
from typing import List, Dict, Tuple, Any, Optional, Set
from pathlib import Path
import numpy as np
import spacy
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class SearchEngine:
    def __init__(self, use_precomputed_index: bool = True):
        logger.info("Booting up Search Engine. Loading artifacts into memory...")

        # 1. Load NLP Pipeline
        try:
            self.nlp = spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

        # 2. Load Mathematical Indices
        self.inverted_index = self._load_pickle(config.INVERTED_INDEX_PATH)
        self.metadata = self._load_pickle(config.DOCUMENT_METADATA_PATH)
        self.doc_texts = self.metadata.get("doc_texts", {})

        # 3. Load Machine Learning Models
        self.tfidf_vectorizer = self._load_pickle(config.TFIDF_VECTORIZER_PATH)
        self.mlb = self._load_pickle(config.LABEL_BINARIZER_PATH)
        self.classifier = self._load_pickle(config.MULTI_LABEL_SVC_PATH)

        # Store label names (normalized to lowercase with underscores)
        self.label_names = [
            label.lower().replace(" ", "_") for label in self.mlb.classes_
        ]

        # Validate label format
        self._validate_label_format()

        # 4. Load or Build Label Index
        if use_precomputed_index:
            self.label_to_docs = self._load_precomputed_index()
        else:
            self.label_to_docs = self._build_label_index()

        logger.info(f"Engine ready. {len(self.label_names)} labels available.")
        avg_docs = (
            np.mean([len(docs) for docs in self.label_to_docs.values()])
            if self.label_to_docs
            else 0
        )
        logger.info(f"Average {avg_docs:.1f} docs per label")

    def _validate_label_format(self):
        """Ensure consistent label formatting."""
        if self.label_names:
            sample = self.label_names[0]
            assert "_" in sample, f"Label format error: {sample} (expected underscores)"
            assert sample.islower(), f"Label case error: {sample} (expected lowercase)"
            logger.info(f"Label format validated: {sample}")

    def _load_pickle(self, filepath: Path) -> Any:
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required artifact: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _load_precomputed_index(self) -> Dict[str, Set[str]]:
        """Load pre-computed label index from training."""
        index_path = config.SVC_MODELS_DIR / "label_to_docs.pkl"
        if index_path.exists():
            logger.info("Loading pre-computed label index...")
            with open(index_path, "rb") as f:
                return pickle.load(f)
        else:
            logger.warning("Pre-computed index not found, building from scratch...")
            return self._build_label_index()

    def _build_label_index(self) -> Dict[str, Set[str]]:
        """
        Build inverted index from classifier predictions (fallback).
        NOTE: This uses classifier predictions, not ground truth, so may have lower recall.
        """
        logger.info("Building label-to-documents index from classifier predictions...")

        import pandas as pd

        label_to_docs = {label: set() for label in self.label_names}

        if not config.PROCESSED_CORPUS_PATH.exists():
            logger.warning(f"Corpus not found at {config.PROCESSED_CORPUS_PATH}")
            return label_to_docs

        df = pd.read_pickle(config.PROCESSED_CORPUS_PATH)
        doc_ids = df["CELEX"].astype(str).str.lower().tolist()
        texts = df["processed_tokens"].apply(lambda tokens: " ".join(tokens)).tolist()

        batch_size = 5000
        for i in range(0, len(doc_ids), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = doc_ids[i : i + batch_size]

            X = self.tfidf_vectorizer.transform(batch_texts)
            predictions = self.classifier.predict(X)

            for idx, doc_id in enumerate(batch_ids):
                doc_labels = predictions[idx]
                for label_idx, has_label in enumerate(doc_labels):
                    if has_label:
                        label_to_docs[self.label_names[label_idx]].add(doc_id)

            if i % 10000 == 0:
                logger.info(
                    f"Indexed {min(i + batch_size, len(doc_ids))}/{len(doc_ids)} documents..."
                )

        nonempty = sum(1 for docs in label_to_docs.values() if docs)
        logger.info(
            f"Label index built: {nonempty}/{len(self.label_names)} labels have documents"
        )

        return label_to_docs

    def preprocess_query(self, raw_query: str) -> List[str]:
        """Process query with legal domain awareness."""
        doc = self.nlp(raw_query.lower())
        tokens = []
        for token in doc:
            # Keep short tokens if they look like acronyms (all caps in original)
            if not token.is_stop and len(token) >= config.MIN_TOKEN_LENGTH:
                tokens.append(token.lemma_)
            elif (
                token.text.isupper() and len(token.text) <= 5
            ):  # Acronyms like EU, VAT, GDPR
                tokens.append(token.text.lower())
        return tokens

    def predict_labels(self, raw_query: str) -> List[str]:
        """
        Predict EUROVOC labels from query text.
        Returns list of label strings.
        """
        query_vector = self.tfidf_vectorizer.transform([raw_query])
        prediction_matrix = self.classifier.predict(query_vector)
        predicted = self.mlb.inverse_transform(prediction_matrix)

        if predicted and len(predicted[0]) > 0:
            return [label.lower().replace(" ", "_") for label in predicted[0]]
        return []

    def boolean_retrieval(self, query_tokens: List[str]) -> Set[str]:
        """Standard boolean AND retrieval."""
        if not query_tokens:
            return set()

        candidate_docs = None
        for token in query_tokens:
            if token in self.inverted_index:
                if candidate_docs is None:
                    candidate_docs = set(self.inverted_index[token].keys())
                else:
                    candidate_docs &= set(self.inverted_index[token].keys())
            else:
                # Token not in index, intersection is empty
                return set()

        return candidate_docs or set()

    def calculate_bm25(
        self, query_tokens: List[str], candidate_docs: Set[str]
    ) -> Dict[str, float]:
        """BM25 scoring with parameters from config."""
        scores = {}
        total_docs = self.metadata.get("total_documents", 1)
        avgdl = self.metadata.get("avgdl", 1.0)
        doc_lengths = self.metadata.get("lengths", {})

        for doc_id in candidate_docs:
            score = 0.0
            doc_len = doc_lengths.get(doc_id, avgdl)

            for token in query_tokens:
                if (
                    token not in self.inverted_index
                    or doc_id not in self.inverted_index[token]
                ):
                    continue

                tf_td = self.inverted_index[token][doc_id]
                df_t = len(self.inverted_index[token])

                # BM25 IDF
                idf = math.log(((total_docs - df_t + 0.5) / (df_t + 0.5)) + 1)

                # BM25 score
                numerator = tf_td * (config.BM25_K1 + 1)
                denominator = tf_td + config.BM25_K1 * (
                    1 - config.BM25_B + config.BM25_B * (doc_len / avgdl)
                )
                score += idf * (numerator / denominator)

            scores[doc_id] = score

        return scores

    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        target_label: Optional[str] = None,
        use_predicted_labels: bool = False,
    ) -> dict:
        """
        Hybrid Retrieval with two modes:

        1. Oracle Mode (target_label provided): Uses ground truth label for evaluation
        2. Live Mode (use_predicted_labels=True): Predicts labels from query text

        Args:
            raw_query: Search query text
            top_k: Number of results to return
            target_label: Ground truth label (for evaluation only)
            use_predicted_labels: If True, predict labels from query text

        Returns:
            dict with results, metadata, and retrieval method used
        """
        query_tokens = self.preprocess_query(raw_query)
        predicted_tags = []
        retrieval_method = "unknown"

        # STAGE 1: Determine candidate set
        if target_label and target_label in self.label_to_docs:
            # ORACLE MODE: Use provided ground truth label
            candidate_docs = self.label_to_docs[target_label].copy()
            retrieval_method = "oracle_gt_label"
            predicted_tags = [target_label]

        elif use_predicted_labels:
            # LIVE MODE: Predict labels from query
            predicted_tags = self.predict_labels(raw_query)

            if predicted_tags:
                # Union of all predicted label document sets
                candidate_docs = set()
                valid_predictions = []

                for tag in predicted_tags:
                    if tag in self.label_to_docs:
                        candidate_docs.update(self.label_to_docs[tag])
                        valid_predictions.append(tag)

                predicted_tags = valid_predictions
                retrieval_method = "ml_prediction"

                # Fallback if predictions have no indexed docs
                if not candidate_docs:
                    candidate_docs = self.boolean_retrieval(query_tokens)
                    retrieval_method = "boolean_fallback"
            else:
                # No labels predicted, fallback to boolean
                candidate_docs = self.boolean_retrieval(query_tokens)
                retrieval_method = "boolean_only"
        else:
            # No label provided and not using predictions - boolean only
            candidate_docs = self.boolean_retrieval(query_tokens)
            retrieval_method = "boolean_only"

        # Handle empty results
        if not candidate_docs:
            return {
                "query_tokens": query_tokens,
                "predicted_tags": predicted_tags,
                "total_hits": 0,
                "top_results": [],
                "method": retrieval_method,
            }

        # STAGE 2: BM25 Ranking within candidate set
        bm25_scores = self.calculate_bm25(query_tokens, candidate_docs)

        # If BM25 returns no scores (e.g., query tokens not in docs), use uniform scoring
        if not bm25_scores:
            bm25_scores = {doc_id: 1.0 for doc_id in candidate_docs}

        # Sort and truncate
        sorted_results = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        return {
            "query_tokens": query_tokens,
            "predicted_tags": predicted_tags,
            "total_hits": len(candidate_docs),
            "top_results": sorted_results,
            "method": retrieval_method,
        }


if __name__ == "__main__":
    # Demo mode
    engine = SearchEngine()

    print("\n" + "=" * 60)
    print("EUR-LEX SEARCH ENGINE CLI")
    print("Mode: Live (predicting labels from query)")
    print("Type 'exit', 'quit', or Ctrl+C to stop.")
    print("=" * 60)

    while True:
        try:
            user_query = input("\n🔍 Enter search query: ").strip()

            if user_query.lower() in ["exit", "quit", "q"]:
                print("Shutting down Search Engine. Goodbye!")
                break

            if not user_query:
                continue

            # Live search with predicted labels
            response = engine.search(user_query, top_k=5, use_predicted_labels=True)

            print("-" * 60)
            print(f"Parsed Tokens:  {response['query_tokens']}")
            print(f"Predicted Tags: {response['predicted_tags']}")
            print(f"Total Hits:     {response['total_hits']}")
            print(f"Retrieval Mode: {response['method'].upper()}")
            print("Top Results:")

            results = response["top_results"]
            if not results:
                print("  No documents found.")
            else:
                for rank, (doc_id, score) in enumerate(results, 1):
                    print(f"  {rank}. CELEX: {doc_id} | Score: {score:.4f}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nShutting down Search Engine. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during search: {e}")
            import traceback

            traceback.print_exc()
