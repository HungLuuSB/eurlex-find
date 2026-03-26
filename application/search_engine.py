"""
search_engine.py

The core Application Domain layer. It loads offline mathematical indices and
Machine Learning models to process user queries, execute Boolean retrieval,
calculate BM25 scores, and predict semantic EUROVOC labels in real-time.
"""

import pickle
import math
from typing import List, Dict, Tuple, Any
from pathlib import Path
import spacy
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class SearchEngine:
    def __init__(self):
        """
        Initializes the Search Engine by loading all offline artifacts into memory.
        This takes a few seconds but ensures lighting-fast query execution.
        """
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

        # 3. Load Machine Learning Models
        self.tfidf_vectorizer = self._load_pickle(config.TFIDF_VECTORIZER_PATH)
        self.mlb = self._load_pickle(config.LABEL_BINARIZER_PATH)
        self.classifier = self._load_pickle(config.MULTI_LABEL_SVC_PATH)

        # Extract global BM25 metrics for $O(1)$ access
        self.total_docs = self.metadata.get("total_documents", 0)
        self.avgdl = self.metadata.get("avgdl", 1.0)
        self.doc_lengths = self.metadata.get("lengths", {})

        logger.info("Search Engine is online and ready for queries.")

    def _load_pickle(self, filepath: Path) -> Any:
        """Helper to securely load pickle artifacts."""
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required artifact: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def preprocess_query(self, raw_query: str) -> List[str]:
        """Passes the user query through the exact same linguistic pipeline as the corpus."""
        doc = self.nlp(raw_query.lower())
        return [
            token.lemma_
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and len(token) >= config.MIN_TOKEN_LENGTH
        ]

    def boolean_retrieval(self, query_tokens: List[str]) -> set:
        """
        Retrieves a set of candidate document IDs (CELEX) that contain at least
        one of the query tokens (Logical OR).
        """
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                # Add all document IDs that contain this token
                candidate_docs.update(self.inverted_index[token].keys())
        return candidate_docs

    def calculate_bm25(
        self, query_tokens: List[str], candidate_docs: set
    ) -> List[Tuple[str, float]]:
        """
        Calculates the Okapi BM25 score for all candidate documents.
        Returns a sorted list of (doc_id, score) tuples.
        """
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in candidate_docs}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            posting_list = self.inverted_index[token]
            df_t = len(posting_list)

            # Calculate Inverse Document Frequency (IDF) with standard smoothing
            idf = math.log(((self.total_docs - df_t + 0.5) / (df_t + 0.5)) + 1)

            for doc_id in candidate_docs:
                if doc_id in posting_list:
                    tf_td = posting_list[doc_id]
                    doc_len = self.doc_lengths.get(doc_id, self.avgdl)

                    # BM25 Term Frequency Normalization
                    numerator = tf_td * (config.BM25_K1 + 1)
                    denominator = tf_td + config.BM25_K1 * (
                        1 - config.BM25_B + config.BM25_B * (doc_len / self.avgdl)
                    )

                    scores[doc_id] += idf * (numerator / denominator)

        # Sort documents by score in descending order
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results

    def predict_intent(self, raw_query: str) -> List[str]:
        """
        Passes the raw query through the TF-IDF Vectorizer and One-Vs-Rest SVC
        to predict the relevant EUROVOC semantic tags.
        """
        query_vector = self.tfidf_vectorizer.transform([raw_query])
        prediction_matrix = self.classifier.predict(query_vector)
        # Inverse transform the binary matrix back into human-readable strings
        predicted_labels = self.mlb.inverse_transform(prediction_matrix)

        # It returns a list of tuples, so we extract the first (and only) tuple
        return list(predicted_labels[0]) if predicted_labels else []

    def search(self, raw_query: str, top_k: int = 10) -> dict:
        """
        The main orchestrator. Executes the 4-pillar search pipeline.
        """
        logger.info(f"Processing Query: '{raw_query}'")

        # 1. Linguistic Preprocessing
        query_tokens = self.preprocess_query(raw_query)
        if not query_tokens:
            logger.warning("Query contained only stop words or invalid tokens.")
            return {"results": [], "predicted_tags": []}

        # 2. Boolean Retrieval (Logical OR)
        candidate_docs = self.boolean_retrieval(query_tokens)

        # 3. BM25 Ranking
        ranked_results = self.calculate_bm25(query_tokens, candidate_docs)
        top_results = ranked_results[:top_k]

        # 4. Machine Learning Semantic Prediction
        predicted_tags = self.predict_intent(raw_query)

        return {
            "query_tokens": query_tokens,
            "predicted_tags": predicted_tags,
            "total_hits": len(candidate_docs),
            "top_results": top_results,
        }


QUERIES = [
    "veterinary controls and health certificates for bovine animals and pigmeat",
    "financial aid and economic sanctions against china",
    "what are the specific rules and regulations regarding the labelling of sea fish within the ec",
    "automatic public tendering for cereals",
]

if __name__ == "__main__":
    # Test execution block
    engine = SearchEngine()

    for query in QUERIES:
        response = engine.search(query, top_k=5)

        print("\n" + "=" * 50)
        print(f"SEARCH RESULTS FOR: '{query}'")
        print("=" * 50)
        print(f"Parsed Tokens: {response['query_tokens']}")
        print(f"Predicted EUROVOC Tags: {response['predicted_tags']}")
        print(f"Total Documents Found: {response['total_hits']}")
        print("-" * 50)

        for rank, (doc_id, score) in enumerate(response["top_results"], 1):
            print(f"{rank}. CELEX: {doc_id} | BM25 Score: {score:.4f}")

        print("=" * 50 + "\n")
