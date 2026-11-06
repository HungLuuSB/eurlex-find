"""
pipeline_inspector.py

A utility script to inspect the Input/Output data transformations
at each stage of the Information Retrieval pipeline.
"""

import pickle
import pandas as pd
import random
from pathlib import Path
import config


def load_pickle_safe(filepath: Path):
    if not filepath.exists():
        print(f"[!] Lỗi: Không tìm thấy file {filepath.name}")
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)


def inspect_stage_1_nlp():
    print("\n" + "=" * 60)
    print("STAGE 1: TIỀN XỬ LÝ NLP (NLP PREPROCESSING)")
    print("=" * 60)

    df = load_pickle_safe(config.PROCESSED_CORPUS_PATH)
    if df is None:
        return

    print(f"Tổng số văn bản: {len(df):,}")

    # Lấy ngẫu nhiên 1 dòng để kiểm tra
    sample_idx = random.randint(0, len(df) - 1)
    sample_doc = df.iloc[sample_idx]

    print("\n[INPUT THÔ (Ví dụ 1 văn bản)]")
    print(f"- CELEX ID : {sample_doc['CELEX']}")
    print(f"- Nhãn gốc : {sample_doc['EUROVOC']}")

    print("\n[OUTPUT SAU KHI QUA SPACY]")
    print(f"- Số lượng token : {len(sample_doc['processed_tokens'])} từ")
    print(f"- Tokens (15 từ đầu): {sample_doc['processed_tokens'][:15]} ...")


def inspect_stage_2_indexing():
    print("\n" + "=" * 60)
    print("STAGE 2: CHỈ MỤC TOÁN HỌC (INVERTED INDEX)")
    print("=" * 60)

    inv_index = load_pickle_safe(config.INVERTED_INDEX_PATH)
    metadata = load_pickle_safe(config.DOCUMENT_METADATA_PATH)

    if not inv_index or not metadata:
        return

    vocab_size = len(inv_index)
    print(f"Kích thước từ điển (Vocabulary) : {vocab_size:,} từ duy nhất")
    print(f"Độ dài tài liệu trung bình (avgdl) : {metadata.get('avgdl', 0):.2f} từ")

    # Lấy thử một từ khóa phổ biến để xem cấu trúc
    sample_word = "import" if "import" in inv_index else list(inv_index.keys())[0]

    postings = inv_index[sample_word]
    print(f"\n[OUTPUT CỦA INVERTED INDEX]")
    print(f"- Tra cứu từ khóa: '{sample_word}'")
    print(f"- Xuất hiện trong: {len(postings):,} văn bản")

    # In ra 3 mapping đầu tiên
    print("- Cấu trúc Mapping (Top 3):")
    for celex, freq in list(postings.items())[:3]:
        print(f"   + Văn bản {celex} -> Xuất hiện {freq} lần")


def inspect_stage_3_machine_learning():
    print("\n" + "=" * 60)
    print("STAGE 3: HỌC MÁY (MACHINE LEARNING ARTIFACTS)")
    print("=" * 60)

    mlb = load_pickle_safe(config.LABEL_BINARIZER_PATH)
    tfidf = load_pickle_safe(config.TFIDF_VECTORIZER_PATH)
    label_to_docs = load_pickle_safe(config.SVC_MODELS_DIR / "label_to_docs.pkl")

    if mlb:
        print("\n[OUTPUT CỦA LABEL BINARIZER]")
        print(f"- Số lượng nhãn đã học : {len(mlb.classes_)}")
        print(f"- 5 Nhãn đầu tiên      : {list(mlb.classes_)[:5]}")

    if tfidf:
        print("\n[OUTPUT CỦA TF-IDF VECTORIZER]")
        print(f"- Giới hạn Features    : {tfidf.max_features:,}")

    if label_to_docs:
        print("\n[OUTPUT CỦA LABEL_TO_DOCS INDEX (Oracle Index)]")
        # Tìm nhãn có ít tài liệu nhất và nhiều tài liệu nhất
        sorted_labels = sorted(label_to_docs.items(), key=lambda x: len(x[1]))
        min_label = sorted_labels[0]
        max_label = sorted_labels[-1]

        print(f"- Nhãn hiếm nhất : '{min_label[0]}' ({len(min_label[1])} văn bản)")
        print(f"- Nhãn phổ biến nhất: '{max_label[0]}' ({len(max_label[1])} văn bản)")

        print(f"\n- Danh sách văn bản mẫu của nhãn '{min_label[0]}':")
        print(f"  {list(min_label[1])[:3]} ...")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        inspect_stage_1_nlp()
        inspect_stage_2_indexing()
        inspect_stage_3_machine_learning()
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")
