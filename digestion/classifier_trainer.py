"""
classifier_trainer_memory_safe.py

Memory-efficient training using SGDClassifier (online learning) instead of LinearSVC.
Trains classifiers sequentially to prevent OOM.
"""

import pandas as pd
import pickle
import gc
from typing import Dict, Set
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
import numpy as np
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate_models(
    corpus_path: Path = config.PROCESSED_CORPUS_PATH,
    top_n_labels: int = config.TOTAL_EUROVOC_LABELS,
) -> None:
    """
    Memory-safe training using SGDClassifier (hinge loss = linear SVM).
    Processes classifiers sequentially to control memory usage.
    """
    logger.info("Initiating memory-safe training pipeline...")

    # Step 1: Load Data
    try:
        df: pd.DataFrame = pd.read_pickle(corpus_path)
        logger.info(f"Loaded {len(df)} preprocessed documents.")
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        raise

    # Step 2: Prepare Labels
    logger.info(f"Identifying the top {top_n_labels} most frequent EUROVOC labels.")

    all_tags = [tag for tags_list in df["EUROVOC"] for tag in tags_list]
    top_tags = pd.Series(all_tags).value_counts().nlargest(top_n_labels).index.tolist()
    top_tags = [tag.lower().replace(" ", "_") for tag in top_tags]

    logger.info(f"Top 5 labels: {top_tags[:5]}")

    # Filter documents to only those with top tags
    df["filtered_tags"] = df["EUROVOC"].apply(
        lambda tags: [
            t.lower().replace(" ", "_")
            for t in tags
            if t.lower().replace(" ", "_") in top_tags
        ]
    )
    df = df[df["filtered_tags"].apply(len) > 0]
    logger.info(
        f"{len(df)} documents have at least one of the top {top_n_labels} labels"
    )

    # Save metadata for later (before freeing memory)
    logger.info("Saving document metadata for indexing...")
    doc_metadata = pd.DataFrame(
        {
            "CELEX": df["CELEX"].astype(str).str.lower(),
            "filtered_tags": df["filtered_tags"],
        }
    )
    metadata_path = config.PROCESSED_CORPUS_PATH.parent / "doc_metadata_filtered.pkl"
    doc_metadata.to_pickle(metadata_path)

    # Binarize labels
    logger.info("Binarizing target variables...")
    mlb = MultiLabelBinarizer(classes=top_tags)
    Y = mlb.fit_transform(df["filtered_tags"])

    # Prepare text
    logger.info("Preparing text data...")
    X_raw = df["processed_tokens"].apply(lambda tokens: " ".join(tokens))

    # Split
    logger.info("Splitting train/test...")
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=0.20, random_state=42
    )

    # Clear heavy dataframes but keep metadata file
    del df, X_raw, doc_metadata
    gc.collect()

    # Step 3: TF-IDF with memory constraints
    logger.info("Fitting TF-IDF Vectorizer...")
    max_features = getattr(config, "TFIDF_MAX_FEATURES", 50000)
    ngram_range = getattr(config, "TFIDF_NGRAM_RANGE", (1, 2))

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=np.float32,
        min_df=2,
        max_df=0.95,
    )

    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    logger.info(f"TF-IDF matrix shape: {X_train.shape}")

    # Clear raw text
    del X_train_raw, X_test_raw
    gc.collect()

    # Step 4: Memory-Safe Training
    logger.info(f"Training {top_n_labels} classifiers using SGDClassifier...")

    base_classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
        verbose=0,
    )

    ovr_classifier = OneVsRestClassifier(base_classifier, n_jobs=1)

    try:
        ovr_classifier.fit(X_train, Y_train)
        logger.info("Training completed successfully.")
    except MemoryError:
        logger.error("MemoryError during training! Try reducing TFIDF_MAX_FEATURES.")
        raise

    # Step 5: Evaluation
    logger.info("Evaluating model...")
    Y_pred = ovr_classifier.predict(X_test)

    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION REPORT")
    print("=" * 60)

    report = classification_report(
        Y_test,
        Y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )
    print(report)

    h_loss = hamming_loss(Y_test, Y_pred)
    micro_f1 = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)

    print(f"Hamming Loss:     {h_loss:.4f}")
    print(f"Micro F1:         {micro_f1:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print("=" * 60 + "\n")

    # Step 6: Build Label Index from saved metadata
    logger.info("Building label-to-documents index from saved metadata...")

    doc_metadata = pd.read_pickle(metadata_path)
    doc_ids_all = doc_metadata["CELEX"].tolist()
    tags_all = doc_metadata["filtered_tags"].tolist()

    label_to_docs: Dict[str, Set[str]] = {label: set() for label in mlb.classes_}

    for doc_id, tags in zip(doc_ids_all, tags_all):
        for tag in tags:
            if tag in label_to_docs:
                label_to_docs[tag].add(doc_id)

    nonempty = sum(1 for docs in label_to_docs.values() if docs)
    logger.info(
        f"Label index built: {nonempty}/{len(mlb.classes_)} labels have documents"
    )

    # Cleanup temp file
    metadata_path.unlink(missing_ok=True)

    # Step 7: Save Everything
    logger.info("Saving models and indices...")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.SVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.MULTI_LABEL_SVC_PATH, "wb") as f:
        pickle.dump(ovr_classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config.TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config.LABEL_BINARIZER_PATH, "wb") as f:
        pickle.dump(mlb, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save label index
    label_index_path = config.SVC_MODELS_DIR / "label_to_docs.pkl"
    with open(label_index_path, "wb") as f:
        pickle.dump(label_to_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save label list
    label_list_path = config.SVC_MODELS_DIR / "valid_labels.txt"
    with open(label_list_path, "w", encoding="utf-8") as f:
        for label in sorted(mlb.classes_):
            f.write(f"{label}\n")

    logger.info(f"All artifacts saved to {config.SVC_MODELS_DIR}")
    logger.info("Training complete!")


if __name__ == "__main__":
    try:
        import psutil

        mem = psutil.virtual_memory()
        logger.info(
            f"Available memory: {mem.available / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB"
        )
    except ImportError:
        pass

    train_and_evaluate_models()
