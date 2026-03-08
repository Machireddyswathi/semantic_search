# scripts/build_index.py
"""
Offline pipeline script: load embeddings from disk, build FAISS index,
and save it so the FastAPI app can load it instantly at startup.

WHY THIS IS A SEPARATE SCRIPT FROM build_embeddings.py:
Single Responsibility Principle — each script does exactly one thing.
If we want to rebuild the index with a different FAISS index type
(e.g. switch to IVF for a larger corpus), we run only this script.
The expensive embedding step is untouched.

PIPELINE ORDER:
  1. download_data.py       → data/raw/
  2. preprocess.py          → data/processed/
  3. build_embeddings.py    → models/embeddings.npy  ✅ done
  4. build_index.py         → models/faiss_index.bin ← we are here
  5. build_clusters.py      → models/gmm_model.pkl
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectorstore.faiss_store import FAISSVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

# Inputs — produced by build_embeddings.py
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.npy")
DOC_IDS_FILE    = os.path.join(MODELS_DIR, "doc_ids.json")
DOC_TEXTS_FILE  = os.path.join(MODELS_DIR, "doc_texts.json")
DOC_META_FILE   = os.path.join(MODELS_DIR, "doc_metadata.json")

# Outputs — consumed by FastAPI at startup
FAISS_INDEX_FILE     = os.path.join(MODELS_DIR, "faiss_index.bin")
FAISS_IDS_FILE       = os.path.join(MODELS_DIR, "faiss_doc_ids.json")
FAISS_TEXTS_FILE     = os.path.join(MODELS_DIR, "faiss_doc_texts.json")
FAISS_META_FILE      = os.path.join(MODELS_DIR, "faiss_doc_metadata.json")

EMBEDDING_DIM = 384


def build_index() -> None:
    """
    Load persisted embeddings and build + save a FAISS index.
    """
    # --- Validate inputs exist ---
    required_files = [
        EMBEDDINGS_FILE, DOC_IDS_FILE, DOC_TEXTS_FILE, DOC_META_FILE
    ]
    for path in required_files:
        if not os.path.exists(path):
            logger.error(f"Required file missing: {path}")
            logger.error("Run: python scripts/build_embeddings.py first")
            sys.exit(1)

    # --- Load embedding matrix ---
    logger.info(f"Loading embedding matrix from {EMBEDDINGS_FILE}...")
    embedding_matrix = np.load(EMBEDDINGS_FILE)
    logger.info(
        f"Matrix loaded: shape={embedding_matrix.shape}, "
        f"dtype={embedding_matrix.dtype}, "
        f"memory={embedding_matrix.nbytes / 1_000_000:.1f}MB"
    )

    # --- Load mappings ---
    with open(DOC_IDS_FILE, "r") as f:
        doc_ids = json.load(f)

    with open(DOC_TEXTS_FILE, "r", encoding="utf-8") as f:
        doc_texts = json.load(f)

    with open(DOC_META_FILE, "r", encoding="utf-8") as f:
        doc_meta = json.load(f)

    logger.info(
        f"Mappings loaded: "
        f"{len(doc_ids)} IDs | "
        f"{len(doc_texts)} texts | "
        f"{len(doc_meta)} metadata records"
    )

    # --- Build FAISS index ---
    store = FAISSVectorStore(dim=EMBEDDING_DIM)
    store.build(
        embedding_matrix=embedding_matrix,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_meta=doc_meta,
    )

    logger.info(repr(store))

    # --- Save index to disk ---
    store.save(
        index_path=FAISS_INDEX_FILE,
        ids_path=FAISS_IDS_FILE,
        texts_path=FAISS_TEXTS_FILE,
        meta_path=FAISS_META_FILE,
    )

    # --- Run search validation ---
    _validate_index(store)


def _validate_index(store: FAISSVectorStore) -> None:
    """
    Validate the index works correctly with known queries.

    WHY VALIDATE:
        A silently broken index (wrong dim, corrupt file, wrong mapping)
        will cause the API to return garbage results. Catching this at
        build time is far better than debugging it at query time.
    """
    logger.info("Validating index with test queries...")

    # We need an embedder to encode test queries
    from embeddings.embedder import Embedder
    embedder = Embedder()

    test_cases = [
        {
            "query": "NASA space exploration astronauts",
            "expected_category_contains": "sci.space",
        },
        {
            "query": "Windows operating system computer",
            "expected_category_contains": "comp.",
        },
        {
            "query": "religion christianity church faith",
            "expected_category_contains": "religion",
        },
    ]

    all_passed = True

    for tc in test_cases:
        query_vec = embedder.encode(tc["query"])
        results = store.search(query_vec, top_k=3)

        top_result = results[0]
        passed = tc["expected_category_contains"] in top_result["category"]

        status = "✅ PASS" if passed else "⚠️  WARN"
        logger.info(
            f"{status} | Query: '{tc['query']}'\n"
            f"         Top result: category='{top_result['category']}' "
            f"score={top_result['score']:.4f}\n"
            f"         Snippet: {top_result['snippet'][:100]}..."
        )

        if not passed:
            all_passed = False

    if all_passed:
        logger.info("All validation checks passed ✅")
    else:
        logger.warning(
            "Some checks did not match expected categories. "
            "This may be acceptable — review manually."
        )


if __name__ == "__main__":
    build_index()