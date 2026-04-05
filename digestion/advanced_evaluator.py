"""
advanced_evaluator.py

Evaluates using classifier-predicted labels as the primary retrieval mechanism.
"""

import csv
import pickle
from ranx import Qrels, Run, evaluate
from pathlib import Path
import config
from application.search_engine import SearchEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def load_doc_mappings(csv_path: Path) -> dict:
    """Reads CSV to map DocID to CelexID, filtering nulls."""
    logger.info(f"Loading ID mappings from {csv_path.name}...")
    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            celex_id = row.get("CelexID")
            doc_id = row.get("DocID")

            if not celex_id or celex_id.lower() == "null" or not doc_id:
                continue

            mapping[str(doc_id)] = str(celex_id).lower()

    logger.info(f"Loaded {len(mapping)} valid mappings")
    return mapping


def load_qrels_filtered(qrels_path: Path, mapping: dict, valid_labels: set) -> tuple:
    """
    Load qrels but only for labels we trained on.
    Returns qrels dict formatted for ranx: {query_text: {doc_id: relevance}}
    """
    logger.info(f"Loading qrels from {qrels_path.name}...")

    qrels = {}
    queries = []
    skipped = 0

    # Normalize valid labels for matching
    valid_normalized = {label.replace("_", " ").lower() for label in valid_labels}

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            label, doc_id, rel = parts
            query_text = label.replace("_", " ").lower()

            # Skip labels not in our 100
            if query_text not in valid_normalized:
                skipped += 1
                continue

            celex_id = mapping.get(str(doc_id))
            if not celex_id:
                continue

            if query_text not in qrels:
                qrels[query_text] = {}
                queries.append(query_text)

            qrels[query_text][celex_id] = int(rel)

    logger.info(
        f"Loaded {len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments, skipped {skipped} labels"
    )
    return qrels, queries


def main():
    # 1. Load models to get valid labels
    logger.info("Loading label binarizer...")
    with open(config.LABEL_BINARIZER_PATH, "rb") as f:
        mlb = pickle.load(f)
    valid_labels = set(mlb.classes_)

    # 2. Load data
    mapping = load_doc_mappings(config.EURLEX_ID_MAPPINGS_PATH)
    qrels_dict, queries = load_qrels_filtered(
        config.EURLEX_EUROVOC_QRELS_PATH, mapping, valid_labels
    )

    if not qrels_dict:
        logger.error("No valid qrels loaded!")
        return

    # 3. Initialize engine (this builds the label index)
    engine = SearchEngine()

    # 4. Run evaluation
    max_queries = min(500, len(queries))
    test_queries = queries[:max_queries]

    logger.info(f"Running evaluation on {len(test_queries)} queries...")

    run_dict = {}

    for i, query_text in enumerate(test_queries, 1):
        # Convert back to underscore format for label lookup
        target_label = query_text.replace(" ", "_")

        # Search using classifier index
        response = engine.search(
            raw_query=query_text, top_k=50, target_label=target_label
        )

        # Build run dict
        run_dict[query_text] = {}
        for doc_id, score in response["top_results"]:
            run_dict[query_text][doc_id] = float(score)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(test_queries)} queries")

    # 5. Evaluate
    qrels = Qrels({k: v for k, v in qrels_dict.items() if k in test_queries})
    run = Run(run_dict)
    run.name = "SVC-First + BM25-Rank"

    metrics = evaluate(
        qrels, run, ["map", "mrr", "ndcg@10", "precision@10", "recall@10"]
    )

    print("\n" + "=" * 60)
    print("EVALUATION REPORT (Classifier-Based Retrieval)")
    print("=" * 60)
    print(f"Queries: {len(test_queries)}")
    for m, v in metrics.items():
        print(f"{m.upper():<15}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
