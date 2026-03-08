# scripts/preprocess.py
"""
Load raw data, apply the full cleaning pipeline, and save processed output.

WHY A SEPARATE SCRIPT (not combined with download):
The Single Responsibility Principle. If cleaning logic changes,
we re-run only this script — we don't re-download 18,000 documents.
This also makes debugging easier: you can inspect raw vs processed data.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_cleaner import clean_batch
from utils.logger import get_logger

logger = get_logger(__name__)

# Paths
RAW_FILE = os.path.join("data", "raw", "newsgroups_raw.json")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "newsgroups_clean.json")
STATS_FILE = os.path.join(PROCESSED_DIR, "preprocessing_stats.json")


def preprocess() -> None:
    """
    Load raw JSON, clean all documents, save results with statistics.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Load raw data ---
    logger.info(f"Loading raw data from {RAW_FILE}...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    logger.info(f"Loaded {len(raw_records)} raw documents")

    # Extract text for batch cleaning
    raw_texts = [record["text"] for record in raw_records]

    # --- Run cleaning pipeline ---
    logger.info("Running cleaning pipeline...")
    cleaned_results = clean_batch(raw_texts, min_length=50)

    # --- Merge cleaned text back with metadata ---
    # We use original_index to match cleaned text to its metadata record
    processed_records = []
    for result in cleaned_results:
        orig = raw_records[result["original_index"]]
        processed_records.append({
            "id": len(processed_records),          # new sequential ID
            "original_id": orig["id"],             # traceability
            "text": result["text"],                # cleaned text
            "category_id": orig["category_id"],
            "category_name": orig["category_name"],
            "char_length": len(result["text"]),
            "word_count": len(result["text"].split())
        })

    # --- Save processed data ---
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, ensure_ascii=False, indent=2)

    logger.info(f"Processed data saved → {PROCESSED_FILE}")

    # --- Compute and save statistics ---
    char_lengths = [r["char_length"] for r in processed_records]
    word_counts = [r["word_count"] for r in processed_records]

    # Category distribution
    from collections import Counter
    category_dist = Counter(r["category_name"] for r in processed_records)

    stats = {
        "total_raw": len(raw_records),
        "total_processed": len(processed_records),
        "discarded": len(raw_records) - len(processed_records),
        "discard_rate_pct": round(
            (len(raw_records) - len(processed_records)) / len(raw_records) * 100, 2
        ),
        "avg_char_length": round(sum(char_lengths) / len(char_lengths), 1),
        "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
        "min_char_length": min(char_lengths),
        "max_char_length": max(char_lengths),
        "category_distribution": dict(category_dist)
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # --- Print summary ---
    logger.info("=" * 50)
    logger.info(f"  Raw documents:       {stats['total_raw']:,}")
    logger.info(f"  After cleaning:      {stats['total_processed']:,}")
    logger.info(f"  Discarded:           {stats['discarded']:,} ({stats['discard_rate_pct']}%)")
    logger.info(f"  Avg word count:      {stats['avg_word_count']}")
    logger.info(f"  Avg char length:     {stats['avg_char_length']}")
    logger.info(f"  Stats saved →        {STATS_FILE}")
    logger.info("=" * 50)


if __name__ == "__main__":
    preprocess()