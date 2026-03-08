# scripts/download_data.py
"""
Download the 20 Newsgroups dataset and save it to data/raw/.

WHY SCIKIT-LEARN'S BUILT-IN FETCHER:
sklearn provides `fetch_20newsgroups` which handles downloading,
caching, and parsing automatically. This is more reliable than
manually downloading from UCI and avoids HTTP/zip parsing issues.

We fetch the dataset ONCE and serialize it to disk so:
- Subsequent steps never need to re-download
- The raw data is preserved untouched (data immutability principle)
"""

import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import fetch_20newsgroups
from utils.logger import get_logger

logger = get_logger(__name__)

# Output directory for raw data
RAW_DATA_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "newsgroups_raw.json")


def download_dataset() -> None:
    """
    Download all 20 Newsgroups categories (train + test subsets combined)
    and save as a JSON file.

    WHY BOTH SUBSETS:
        For an unsupervised search system, the train/test split used in
        classification experiments is meaningless. We want the full corpus
        for richer search results and better cluster coverage.

    WHY JSON:
        Human-readable, easy to inspect, works cross-platform.
        For very large datasets we'd use Parquet — 18,000 docs is fine as JSON.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    logger.info("Fetching 20 Newsgroups dataset (train split)...")
    train_data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),  # sklearn's own basic removal
        # WHY remove these: we do our own fine-grained cleaning later,
        # but sklearn's removal catches the worst structural noise first
    )

    logger.info("Fetching 20 Newsgroups dataset (test split)...")
    test_data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    # Combine both subsets into a unified corpus
    all_texts = list(train_data.data) + list(test_data.data)
    all_labels = list(train_data.target) + list(test_data.target)
    category_names = train_data.target_names  # same for both subsets

    logger.info(f"Total documents: {len(all_texts)}")
    logger.info(f"Categories ({len(category_names)}): {category_names}")

    # Build structured records
    records = []
    for idx, (text, label_idx) in enumerate(zip(all_texts, all_labels)):
        records.append({
            "id": idx,
            "text": text,
            "category_id": int(label_idx),
            "category_name": category_names[label_idx]
        })

    # Save to disk
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Raw data saved → {OUTPUT_FILE}")
    logger.info(f"File size: {os.path.getsize(OUTPUT_FILE) / 1_000_000:.1f} MB")


if __name__ == "__main__":
    download_dataset()