"""
advanced_evaluator.py

Dual-mode evaluation:
1. ORACLE: Uses ground truth labels (tests BM25 ranking quality)
2. PREDICTED: Uses classifier-predicted labels (tests end-to-end performance)
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
    null_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            celex_id = row.get("CelexID")
            doc_id = row.get("DocID")

            if not celex_id or celex_id.lower() == "null" or not doc_id:
                null_count += 1
                continue

            mapping[str(doc_id)] = str(celex_id).lower()

    logger.info(f"Loaded {len(mapping)} valid mappings (skipped {null_count} nulls)")
    return mapping


def load_qrels_filtered(qrels_path: Path, mapping: dict, valid_labels: set) -> tuple:
    """
    Load qrels filtered to only include labels we trained on.
    Returns (qrels_dict, queries_list)
    """
    logger.info(f"Loading qrels from {qrels_path.name}...")

    qrels = {}
    queries = []
    skipped_labels = 0
    missing_mappings = 0

    # Normalize valid labels for matching (qrels use underscores)
    valid_normalized = {label.lower().replace(" ", "_") for label in valid_labels}

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            label, doc_id, rel = parts
            query_text = label.lower().replace("_", " ")  # Normalize for display

            # Skip labels not in our training set
            label_normalized = label.lower().replace(" ", "_")
            if label_normalized not in valid_normalized:
                skipped_labels += 1
                continue

            celex_id = mapping.get(str(doc_id))
            if not celex_id:
                missing_mappings += 1
                continue

            if query_text not in qrels:
                qrels[query_text] = {}
                queries.append(query_text)

            qrels[query_text][celex_id] = int(rel)

    logger.info(
        f"Loaded {len(queries)} queries, {sum(len(v) for v in qrels.values())} judgments"
    )
    logger.info(f"Skipped {skipped_labels} labels not in training set")
    logger.info(f"Skipped {missing_mappings} docs with missing ID mappings")

    return qrels, queries


def evaluate_oracle_mode(
    engine: SearchEngine, qrels: dict, queries: list, max_queries: int = 500
):
    """
    ORACLE EVALUATION: Uses ground truth labels to test BM25 ranking quality.
    This measures the upper bound of retrieval performance.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("ORACLE MODE EVALUATION (Ground Truth Labels)")
    logger.info(f"{'=' * 60}")

    test_queries = queries[: min(max_queries, len(queries))]
    run_dict = {}

    for i, query_text in enumerate(test_queries, 1):
        # Convert back to underscore format for label lookup
        target_label = query_text.replace(" ", "_")

        # Search using ground truth label (oracle mode)
        response = engine.search(
            raw_query=query_text, top_k=50, target_label=target_label
        )

        # Build run dict for ranx
        run_dict[query_text] = {}
        for doc_id, score in response["top_results"]:
            run_dict[query_text][doc_id] = float(score)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(test_queries)} queries")

    # Evaluate
    qrels_obj = Qrels({k: v for k, v in qrels.items() if k in test_queries})
    run = Run(run_dict)
    run.name = "Oracle-GT + BM25"

    metrics = evaluate(
        qrels_obj,
        run,
        ["map", "mrr", "ndcg@10", "precision@10", "recall@10", "recall@50"],
    )

    print(f"\n{'=' * 60}")
    print("ORACLE MODE RESULTS (Upper Bound)")
    print(f"{'=' * 60}")
    print(f"Queries Evaluated: {len(test_queries)}")
    for m, v in metrics.items():
        print(f"{m.upper():<15}: {v:.4f}")
    print(f"{'=' * 60}\n")

    return metrics


def evaluate_predicted_mode(
    engine: SearchEngine, qrels: dict, queries: list, max_queries: int = 500
):
    """
    PREDICTED MODE EVALUATION: Uses classifier-predicted labels from query text.
    This measures real-world end-to-end performance.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("PREDICTED MODE EVALUATION (End-to-End)")
    logger.info(f"{'=' * 60}")

    test_queries = queries[: min(max_queries, len(queries))]
    run_dict = {}
    prediction_stats = {"correct": 0, "incorrect": 0, "no_prediction": 0}

    for i, query_text in enumerate(test_queries, 1):
        # Search using predicted labels (real-world scenario)
        response = engine.search(
            raw_query=query_text, top_k=50, use_predicted_labels=True
        )

        # Check if any predicted label matches ground truth
        true_label = query_text.replace(" ", "_")
        predicted_tags = response.get("predicted_tags", [])

        if not predicted_tags:
            prediction_stats["no_prediction"] += 1
        elif true_label in predicted_tags:
            prediction_stats["correct"] += 1
        else:
            prediction_stats["incorrect"] += 1

        # Build run dict
        run_dict[query_text] = {}
        for doc_id, score in response["top_results"]:
            run_dict[query_text][doc_id] = float(score)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(test_queries)} queries")

    # Evaluate
    qrels_obj = Qrels({k: v for k, v in qrels.items() if k in test_queries})
    run = Run(run_dict)
    run.name = "SVC-Predicted + BM25"

    metrics = evaluate(
        qrels_obj,
        run,
        ["map", "mrr", "ndcg@10", "precision@10", "recall@10", "recall@50"],
    )

    print(f"\n{'=' * 60}")
    print("PREDICTED MODE RESULTS (Real-World Performance)")
    print(f"{'=' * 60}")
    print(f"Queries Evaluated: {len(test_queries)}")
    for m, v in metrics.items():
        print(f"{m.upper():<15}: {v:.4f}")
    print("-" * 60)
    print("Label Prediction Statistics:")
    print(
        f"  Correct:        {prediction_stats['correct']} ({prediction_stats['correct'] / len(test_queries) * 100:.1f}%)"
    )
    print(
        f"  Incorrect:      {prediction_stats['incorrect']} ({prediction_stats['incorrect'] / len(test_queries) * 100:.1f}%)"
    )
    print(
        f"  No Prediction:  {prediction_stats['no_prediction']} ({prediction_stats['no_prediction'] / len(test_queries) * 100:.1f}%)"
    )
    print(f"{'=' * 60}\n")

    return metrics, prediction_stats


def main():
    # 1. Load models to get valid labels
    logger.info("Loading label binarizer...")
    with open(config.LABEL_BINARIZER_PATH, "rb") as f:
        mlb = pickle.load(f)
    valid_labels = set(mlb.classes_)
    logger.info(f"Loaded {len(valid_labels)} trained labels")

    # 2. Load data
    mapping = load_doc_mappings(config.EURLEX_ID_MAPPINGS_PATH)
    qrels_dict, queries = load_qrels_filtered(
        config.EURLEX_EUROVOC_QRELS_PATH, mapping, valid_labels
    )

    if not qrels_dict:
        logger.error(
            "No valid qrels loaded! Check label overlap between training and evaluation."
        )
        return

    # 3. Initialize engine
    engine = SearchEngine(use_precomputed_index=True)

    # 4. Run both evaluations
    max_q = 500  # Limit for speed, or set to len(queries) for full eval

    # Oracle mode (upper bound)
    oracle_metrics = evaluate_oracle_mode(engine, qrels_dict, queries, max_q)

    # Predicted mode (real world)
    predicted_metrics, pred_stats = evaluate_predicted_mode(
        engine, qrels_dict, queries, max_q
    )

    # 5. Comparison
    print(f"\n{'=' * 60}")
    print("PERFORMANCE GAP ANALYSIS")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Oracle':<10} {'Predicted':<10} {'Gap':<10}")
    print("-" * 60)
    for metric in ["map", "mrr", "ndcg@10", "precision@10"]:
        oracle_val = oracle_metrics.get(metric, 0)
        pred_val = predicted_metrics.get(metric, 0)
        gap = oracle_val - pred_val
        print(
            f"{metric.upper():<15} {oracle_val:<10.4f} {pred_val:<10.4f} {gap:<10.4f}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
