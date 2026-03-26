"""
classifier_trainer.py

This module trains the Multi-Label Support Vector Classifiers for the EUROVOC tags.
It includes data splitting, TF-IDF feature extraction, model training, and a
rigorous mathematical evaluation of the results.
"""

import pandas as pd
import pickle
from typing import List, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, hamming_loss
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate_models(
    corpus_path: Path = config.PROCESSED_CORPUS_PATH, top_n_labels: int = 100
) -> None:
    """
    Executes the complete machine learning pipeline for the document classifier.

    Steps performed:
    1. Loads the preprocessed corpus.
    2. Filters the EUROVOC tags to the top N most frequent labels.
    3. Binarizes the targets (Y) and splits the data 80/20.
    4. Vectorizes the lemmatized tokens into a sparse TF-IDF matrix (X).
    5. Trains the OneVsRestClassifier wrapping a LinearSVC.
    6. Evaluates the predictions using F1-Scores and Hamming Loss.
    7. Serializes the Vectorizer, Binarizer, and Model to disk.
    """
    logger.info("Initiating machine learning pipeline...")

    # Step 1: Load Data
    try:
        df: pd.DataFrame = pd.read_pickle(corpus_path)  # type: ignore
        logger.info(f"Loaded {len(df)} preprocessed documents.")
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        raise

    # Step 2 & 3: Prepare the Target Variable (Y)
    logger.info(f"Identifying the top {top_n_labels} most frequent EUROVOC labels.")

    # Flatten all tags to count frequencies, then select the top 100
    all_tags = [tag for tags_list in df["EUROVOC"] for tag in tags_list]
    top_tags = pd.Series(all_tags).value_counts().nlargest(top_n_labels).index.tolist()

    # Filter the DataFrame's lists to only include the top 100 tags
    df["filtered_tags"] = df["EUROVOC"].apply(
        lambda tags: [t for t in tags if t in top_tags]
    )

    logger.info("Binarizing target variables into a 2D matrix.")
    mlb = MultiLabelBinarizer(classes=top_tags)
    Y = mlb.fit_transform(df["filtered_tags"])

    # Prepare the Input Variable (X) by joining tokens back into strings for TF-IDF
    logger.info("Joining token lists into text strings for vectorization.")
    X_raw = df["processed_tokens"].apply(lambda tokens: " ".join(tokens))

    # Perform the 80/20 Split
    logger.info("Executing 80/20 Train/Test split.")
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=0.20, random_state=42
    )

    # Step 4: Feature Extraction
    logger.info("Fitting TF-IDF Vectorizer and transforming text to sparse matrices.")
    tfidf = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES, ngram_range=(1, 2))

    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    # Step 5: Model Training
    # LinearSVC with dual=False is highly optimized for situations where n_samples > n_features
    logger.info(
        "Training 100 One-Vs-Rest Linear Support Vector Classifiers. This may take a few minutes..."
    )
    base_classifier = LinearSVC(dual=False, random_state=42)  # type: ignore
    ovr_classifier = OneVsRestClassifier(base_classifier, n_jobs=1)

    ovr_classifier.fit(X_train, Y_train)
    logger.info("Model training successfully converged.")

    # Step 6: Mathematical Evaluation
    logger.info("Predicting on the Test set for evaluation.")
    Y_pred = ovr_classifier.predict(X_test)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)

    # Print the detailed classification report (Precision, Recall, F1 for each label)
    # Using zero_division=0 to prevent warnings if a rare label gets no positive predictions
    report = classification_report(
        Y_test,
        Y_pred,
        target_names=mlb.classes_,
        zero_division=0,  # type: ignore
    )
    print(report)

    h_loss = hamming_loss(Y_test, Y_pred)
    print(f"Overall Hamming Loss: {h_loss:.4f} (Lower is better)")
    print("=" * 50 + "\n")

    # Step 7: Serialization
    logger.info("Serializing machine learning artifacts to disk...")

    # Ensure directories exist
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.SVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config.LABEL_BINARIZER_PATH, "wb") as f:
        pickle.dump(mlb, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config.MULTI_LABEL_SVC_PATH, "wb") as f:
        pickle.dump(ovr_classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Data Digestion Layer officially complete.")


if __name__ == "__main__":
    train_and_evaluate_models()
