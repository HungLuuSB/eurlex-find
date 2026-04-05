"""
search_engine.py

Hybrid retrieval: Uses pre-computed classifier predictions to find documents
tagged with the target EUROVOC label, then BM25 ranks within that set.
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
    def __init__(self):
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

        # Load document texts for on-the-fly classification if needed
        self.doc_texts = self.metadata.get("doc_texts", {})  # {celex_id: raw_text}

        # 3. Load Machine Learning Models
        self.tfidf_vectorizer = self._load_pickle(config.TFIDF_VECTORIZER_PATH)
        self.mlb = self._load_pickle(config.LABEL_BINARIZER_PATH)
        self.classifier = self._load_pickle(config.MULTI_LABEL_SVC_PATH)

        # Store label names and create inverted index: label -> set(doc_ids)
        self.label_names = list(self.mlb.classes_)
        self.label_to_docs = self._build_label_index()

        logger.info(f"Engine ready. {len(self.label_names)} labels available.")
        logger.info(
            f"Average {np.mean([len(docs) for docs in self.label_to_docs.values()]):.1f} docs per label"
        )

    def _load_pickle(self, filepath: Path) -> Any:
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required artifact: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _build_label_index(self) -> Dict[str, Set[str]]:
        """
        Build inverted index: label -> set of doc IDs predicted to have that label.
        Loads the processed corpus directly to save memory without bloating the metadata file.
        """
        logger.info(
            "Building label-to-documents index from processed corpus... (This takes a moment)"
        )

        label_to_docs = {label: set() for label in self.label_names}

        if not config.PROCESSED_CORPUS_PATH.exists():
            logger.warning(
                f"Corpus not found at {config.PROCESSED_CORPUS_PATH} - label index empty"
            )
            return label_to_docs

        import pandas as pd

        # Load the corpus
        df = pd.read_pickle(config.PROCESSED_CORPUS_PATH)
        doc_ids = df["CELEX"].astype(str).tolist()

        # Join tokens back into strings for the TF-IDF vectorizer
        texts = df["processed_tokens"].apply(lambda tokens: " ".join(tokens)).tolist()

        # Process in batches to keep RAM usage perfectly flat
        batch_size = 5000
        for i in range(0, len(doc_ids), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = doc_ids[i : i + batch_size]

            # Vectorize and predict
            X = self.tfidf_vectorizer.transform(batch_texts)
            predictions = self.classifier.predict(X)

            # Update index
            for idx, doc_id in enumerate(batch_ids):
                doc_labels = predictions[idx]
                for label_idx, has_label in enumerate(doc_labels):
                    if has_label:
                        # Ensure lowercase for strict matching in evaluation
                        label_to_docs[self.label_names[label_idx]].add(doc_id.lower())

            logger.info(
                f"Indexed {min(i + batch_size, len(doc_ids))}/{len(doc_ids)} documents..."
            )

        nonempty = sum(1 for docs in label_to_docs.values() if docs)
        logger.info(
            f"Label index built: {nonempty}/{len(self.label_names)} labels have documents"
        )

        return label_to_docs

    def preprocess_query(self, raw_query: str) -> List[str]:
        doc = self.nlp(raw_query.lower())
        return [
            token.lemma_
            for token in doc
            if not token.is_stop and len(token) >= config.MIN_TOKEN_LENGTH
        ]

    def boolean_retrieval(self, query_tokens: List[str]) -> Set[str]:
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token].keys())
        return candidate_docs

    def calculate_bm25(
        self, query_tokens: List[str], candidate_docs: Set[str]
    ) -> Dict[str, float]:
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

                # IDF
                idf = math.log(((total_docs - df_t + 0.5) / (df_t + 0.5)) + 1)

                # BM25
                numerator = tf_td * (config.BM25_K1 + 1)
                denominator = tf_td + config.BM25_K1 * (
                    1 - config.BM25_B + config.BM25_B * (doc_len / avgdl)
                )
                score += idf * (numerator / denominator)

            scores[doc_id] = score

        return scores

    def search(
        self, raw_query: str, top_k: int = 10, target_label: Optional[str] = None
    ) -> dict:
        """
        Smart Hybrid Retrieval:
        - Evaluation Mode: Uses explicitly provided target_label.
        - Live User Mode: Predicts tags on-the-fly using the ML classifier.
        """
        query_tokens = self.preprocess_query(raw_query)
        predicted_tags = []

        # STAGE 1: Determine candidate set & Predict intent
        if target_label and target_label in self.label_to_docs:
            # Mode 1: Academic Evaluation (LexGLUE)
            candidate_docs = self.label_to_docs[target_label].copy()
            retrieval_method = "classifier_eval"
            predicted_tags = [target_label]

        else:
            # Mode 2: Live User / CLI (Predict tags on-the-fly)
            query_vector = self.tfidf_vectorizer.transform([raw_query])
            prediction_matrix = self.classifier.predict(query_vector)
            labels = self.mlb.inverse_transform(prediction_matrix)
            predicted_tags = list(labels[0]) if labels else []

            if predicted_tags:
                # If ML predicts tags, gather documents associated with ALL predicted tags
                candidate_docs = set()
                for tag in predicted_tags:
                    if tag in self.label_to_docs:
                        candidate_docs.update(self.label_to_docs[tag])
                retrieval_method = "ml_prediction + bm25"

                # Failsafe: if somehow the tag has no docs, fallback to boolean
                if not candidate_docs:
                    candidate_docs = self.boolean_retrieval(query_tokens)
                    retrieval_method = "boolean_fallback"
            else:
                # ML model couldn't predict a tag, fallback to Boolean completely
                if not query_tokens:
                    return {
                        "top_results": [],
                        "total_hits": 0,
                        "method": "none",
                        "query_tokens": [],
                        "predicted_tags": [],
                    }
                candidate_docs = self.boolean_retrieval(query_tokens)
                retrieval_method = "boolean_only"

        # If absolutely no documents found
        if not candidate_docs:
            return {
                "query_tokens": query_tokens,
                "predicted_tags": predicted_tags,
                "total_hits": 0,
                "top_results": [],
                "method": retrieval_method,
            }

        # STAGE 2: BM25 Ranking within the candidate set
        bm25_scores = self.calculate_bm25(query_tokens, candidate_docs)

        if not bm25_scores:
            bm25_scores = {doc_id: 1.0 for doc_id in candidate_docs}

        sorted_results = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Return everything needed for the CLI
        return {
            "query_tokens": query_tokens,
            "predicted_tags": predicted_tags,
            "total_hits": len(candidate_docs),
            "top_results": sorted_results,
            "method": retrieval_method,
        }


if __name__ == "__main__":
    # Boot up the engine once
    engine = SearchEngine()

    print("\n" + "=" * 60)
    print("EUR-LEX SEARCH ENGINE CLI")
    print("Type your query below. Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60)

    # Start the interactive loop
    while True:
        try:
            user_query = input("\n🔍 Enter search query: ").strip()

            # Check for exit commands
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Shutting down Search Engine. Goodbye!")
                break

            # Skip empty inputs
            if not user_query:
                continue

            # Execute search
            response = engine.search(user_query, top_k=5)

            # Display results
            print("-" * 60)
            print(f"Parsed Tokens:  {response.get('query_tokens', [])}")
            print(f"Predicted Tags: {response.get('predicted_tags', [])}")
            print(f"Total Hits:     {response.get('total_hits', 0)}")
            print(f"Retrieval Mode: {response.get('method', 'unknown').upper()}")
            print("Top Results:")

            results = response.get("top_results", [])
            if not results:
                print("  No documents found.")
            else:
                for rank, (doc_id, score) in enumerate(results, 1):
                    print(f"  {rank}. CELEX: {doc_id} | Score: {score:.4f}")
            print("-" * 60)

        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            print("\nShutting down Search Engine. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during search: {e}")
