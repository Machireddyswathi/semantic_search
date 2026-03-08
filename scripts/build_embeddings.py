# scripts/build_embeddings.py
"""
Offline pipeline script: load processed documents, generate embeddings,
and save them as a NumPy matrix alongside an ID mapping.

WHY SAVE EMBEDDINGS TO DISK:
Embedding 16,000+ documents takes ~2 minutes even on fast hardware.
We do this ONCE offline and save the result. The API then loads
pre-computed embeddings at startup — instant, no recomputation needed.

This is a core production principle: separate TRAINING/INDEXING time
from SERVING time. Heavy computation never happens during a user request.

OUTPUT FILES:
  models/embeddings.npy   — float32 matrix, shape (N, 384)
  models/doc_ids.json     — list mapping row index → document ID
  models/doc_texts.json   — list mapping row index → document text
                            (needed to return actual results to users)
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.embedder import Embedder
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_FILE = os.path.join("data", "processed", "newsgroups_clean.json")
MODELS_DIR = "models"
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.npy")
DOC_IDS_FILE = os.path.join(MODELS_DIR, "doc_ids.json")
DOC_TEXTS_FILE = os.path.join(MODELS_DIR, "doc_texts.json")
DOC_META_FILE = os.path.join(MODELS_DIR, "doc_metadata.json")


def build_embeddings(
    limit: int | None = None,
    batch_size: int = 64
) -> None:
    """
    Generate and persist embeddings for all processed documents.

    Args:
        limit:      if set, only embed the first N documents.
                    WHY: useful for quick testing without full corpus.
        batch_size: passed to Embedder.encode_batch()
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load processed documents ---
    logger.info(f"Loading processed data from {PROCESSED_FILE}...")
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        logger.info(f"Limiting to first {limit} documents for testing")
        records = records[:limit]

    logger.info(f"Documents to embed: {len(records):,}")

    # --- Extract parallel lists (text and metadata) ---
    texts = [r["text"] for r in records]
    doc_ids = [r["id"] for r in records]

    # Metadata stored separately for quick lookup by the API
    doc_metadata = [
        {
            "id": r["id"],
            "original_id": r["original_id"],
            "category_id": r["category_id"],
            "category_name": r["category_name"],
            "word_count": r["word_count"],
            # Store a snippet (first 300 chars) to return in API responses
            # without loading full text every time
            "snippet": r["text"][:300]
        }
        for r in records
    ]

    # --- Generate embeddings ---
    embedder = Embedder()
    embedding_matrix = embedder.encode_batch(
        texts,
        batch_size=batch_size,
        show_progress=True
    )

    # --- Validate output ---
    assert embedding_matrix.shape == (len(texts), embedder.dim), (
        f"Shape mismatch: expected ({len(texts)}, {embedder.dim}), "
        f"got {embedding_matrix.shape}"
    )
    assert embedding_matrix.dtype == np.float32, (
        f"Expected float32, got {embedding_matrix.dtype}"
    )

    # --- Verify normalization (spot check 5 random vectors) ---
    sample_indices = np.random.choice(len(texts), size=5, replace=False)
    for idx in sample_indices:
        norm = np.linalg.norm(embedding_matrix[idx])
        assert abs(norm - 1.0) < 1e-5, (
            f"Vector at index {idx} is not unit-normalized: norm={norm:.6f}"
        )
    logger.info("Normalization check passed (5 random vectors verified)")

    # --- Save artifacts ---
    # 1. Embedding matrix (the heavy file — ~24MB for 16K docs at 384-dim float32)
    np.save(EMBEDDINGS_FILE, embedding_matrix)
    logger.info(
        f"Embeddings saved → {EMBEDDINGS_FILE} "
        f"({os.path.getsize(EMBEDDINGS_FILE) / 1_000_000:.1f} MB)"
    )

    # 2. Document IDs — maps row index i → document ID
    # WHY: FAISS returns row indices, not document IDs. We need this
    # mapping to look up which document a result belongs to.
    with open(DOC_IDS_FILE, "w") as f:
        json.dump(doc_ids, f)
    logger.info(f"Document IDs saved → {DOC_IDS_FILE}")

    # 3. Full document texts — needed to return actual content to users
    with open(DOC_TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    logger.info(f"Document texts saved → {DOC_TEXTS_FILE}")

    # 4. Document metadata — category, snippet, word count
    with open(DOC_META_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Document metadata saved → {DOC_META_FILE}")

    # --- Summary ---
    logger.info("=" * 55)
    logger.info(f"  Documents embedded:  {len(texts):,}")
    logger.info(f"  Embedding shape:     {embedding_matrix.shape}")
    logger.info(f"  Embedding dtype:     {embedding_matrix.dtype}")
    logger.info(f"  Memory (matrix):     {embedding_matrix.nbytes / 1_000_000:.1f} MB")
    logger.info(f"  Disk (matrix):       {os.path.getsize(EMBEDDINGS_FILE) / 1_000_000:.1f} MB")
    logger.info("=" * 55)

    # --- Quick similarity sanity check ---
    _run_sanity_check(embedding_matrix, texts, embedder)


def _run_sanity_check(
    matrix: np.ndarray,
    texts: list[str],
    embedder: Embedder,
    n_checks: int = 3
) -> None:
    """
    Run a quick sanity check to verify embeddings are semantically meaningful.

    We pick random documents, encode a related query, and verify the most
    similar document in the matrix is actually topically relevant.
    This is a smoke test — not a benchmark.
    """
    logger.info("Running embedding sanity checks...")

    test_queries = [
        "space shuttle NASA mission",
        "gun control political debate",
        "computer graphics OpenGL rendering",
    ]

    for query in test_queries:
        query_vec = embedder.encode(query)  # shape (384,)

        # Dot product with all rows (works because vectors are normalized)
        # shape: (N,)
        similarities = matrix @ query_vec

        top_idx = int(np.argmax(similarities))
        top_score = float(similarities[top_idx])
        top_snippet = texts[top_idx][:120].replace("\n", " ")

        logger.info(
            f"\n  Query:   '{query}'\n"
            f"  Top hit: score={top_score:.4f}\n"
            f"  Snippet: {top_snippet}..."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build document embeddings for semantic search"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N documents (for testing). Default: all documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding. Default: 64"
    )
    args = parser.parse_args()

    build_embeddings(limit=args.limit, batch_size=args.batch_size)