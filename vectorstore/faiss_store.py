# vectorstore/faiss_store.py
"""
FAISS vector store abstraction layer.

DESIGN PATTERN — Repository Pattern:
This class is the ONLY place in the codebase that touches FAISS directly.
The rest of the system calls clean methods like .search() and .add().

WHY THIS MATTERS:
If we ever want to switch from FAISS to Chroma, Pinecone, or Weaviate,
we only rewrite THIS file. Zero changes to API, cache, or clustering code.

FAISS INDEX TYPES (for reference):
┌─────────────────┬────────────┬──────────┬─────────────────────────┐
│ Index Type      │ Speed      │ Accuracy │ Best For                │
├─────────────────┼────────────┼──────────┼─────────────────────────┤
│ IndexFlatIP     │ Moderate   │ Exact    │ < 100K vectors (us!)    │
│ IndexIVFFlat    │ Fast       │ ~95%     │ 100K - 1M vectors       │
│ IndexHNSWFlat   │ Very Fast  │ ~98%     │ 1M+ vectors             │
│ IndexIVFPQ      │ Very Fast  │ ~90%     │ Billion vectors         │
└─────────────────┴────────────┴──────────┴─────────────────────────┘

We use IndexFlatIP: exact results, no hyperparameter tuning needed.
"""

import os
import json
import numpy as np
import faiss
from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    Manages a FAISS index for fast semantic similarity search.

    Responsibilities:
    - Build index from embedding matrix
    - Save / load index from disk
    - Search for top-K nearest neighbors
    - Map FAISS row indices back to document IDs

    Attributes:
        index:      the FAISS index object
        doc_ids:    list mapping FAISS row index → document ID
        doc_texts:  list mapping FAISS row index → document text
        doc_meta:   list mapping FAISS row index → metadata dict
        dim:        embedding dimension (must match embedder)
    """

    def __init__(self, dim: int = 384):
        """
        Initialize an empty vector store.

        Args:
            dim: embedding dimensionality. MUST match the embedder.
                 Mismatched dims cause silent wrong results — hence the
                 assertion in .build().
        """
        self.dim = dim
        self.index = None          # FAISS index — None until built or loaded
        self.doc_ids: list = []    # row i → document ID
        self.doc_texts: list = []  # row i → full text
        self.doc_meta: list = []   # row i → {category, snippet, ...}

        logger.info(f"FAISSVectorStore initialized (dim={dim})")

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(
        self,
        embedding_matrix: np.ndarray,
        doc_ids: list,
        doc_texts: list,
        doc_meta: list,
    ) -> None:
        """
        Build a FAISS IndexFlatIP from a precomputed embedding matrix.

        Args:
            embedding_matrix: float32 array of shape (N, dim)
            doc_ids:          list of length N, maps row → document ID
            doc_texts:        list of length N, maps row → document text
            doc_meta:         list of length N, maps row → metadata dict

        WHY IndexFlatIP:
            IP = Inner Product. On L2-normalized vectors, inner product
            IS cosine similarity. No approximation, no quantization.
            For 16K vectors this is ~2ms per query — fast enough.
        """
        # --- Validate inputs ---
        assert isinstance(embedding_matrix, np.ndarray), \
            "embedding_matrix must be a numpy array"
        assert embedding_matrix.dtype == np.float32, \
            f"Expected float32, got {embedding_matrix.dtype}. FAISS requires float32."
        assert embedding_matrix.shape[1] == self.dim, \
            f"Dimension mismatch: index dim={self.dim}, matrix dim={embedding_matrix.shape[1]}"
        assert len(doc_ids) == len(embedding_matrix), \
            "doc_ids length must match number of embeddings"

        n_vectors = embedding_matrix.shape[0]
        logger.info(f"Building FAISS IndexFlatIP with {n_vectors:,} vectors...")

        # --- Create index ---
        # IndexFlatIP: exhaustive inner product search, exact results
        self.index = faiss.IndexFlatIP(self.dim)

        # --- Add vectors ---
        # FAISS requires C-contiguous float32 arrays
        # np.ascontiguousarray ensures memory layout is correct
        vectors = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
        self.index.add(vectors)

        # --- Store mappings ---
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.doc_meta = doc_meta

        logger.info(
            f"Index built successfully. "
            f"Total vectors: {self.index.ntotal:,}"
        )

    # ------------------------------------------------------------------
    # Persistence — save and load
    # ------------------------------------------------------------------

    def save(
        self,
        index_path: str,
        ids_path: str,
        texts_path: str,
        meta_path: str,
    ) -> None:
        """
        Persist the FAISS index and all mappings to disk.

        WHY SEPARATE FILES FOR MAPPINGS:
            FAISS only stores vectors — it has no concept of document IDs
            or text. We store those mappings in JSON files alongside the
            binary index so they can be loaded together atomically.

        Args:
            index_path: path for the binary FAISS index (.bin)
            ids_path:   path for doc_ids JSON
            texts_path: path for doc_texts JSON
            meta_path:  path for doc_meta JSON
        """
        if self.index is None:
            raise RuntimeError("Index has not been built. Call .build() first.")

        # Save FAISS binary index
        faiss.write_index(self.index, index_path)
        logger.info(
            f"FAISS index saved → {index_path} "
            f"({os.path.getsize(index_path) / 1_000_000:.1f} MB)"
        )

        # Save ID mapping
        with open(ids_path, "w") as f:
            json.dump(self.doc_ids, f)
        logger.info(f"Doc IDs saved → {ids_path}")

        # Save texts (needed to return results)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_texts, f, ensure_ascii=False)
        logger.info(f"Doc texts saved → {texts_path}")

        # Save metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Doc metadata saved → {meta_path}")

    def load(
        self,
        index_path: str,
        ids_path: str,
        texts_path: str,
        meta_path: str,
    ) -> None:
        """
        Load a previously saved FAISS index and mappings from disk.

        This is what the FastAPI app calls at startup — fast load,
        no recomputation.

        Args:
            index_path: path to the binary FAISS index (.bin)
            ids_path:   path to doc_ids JSON
            texts_path: path to doc_texts JSON
            meta_path:  path to doc_meta JSON
        """
        # Validate all files exist before attempting load
        for path in [index_path, ids_path, texts_path, meta_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Run: python scripts/build_index.py"
                )

        logger.info(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        with open(ids_path, "r") as f:
            self.doc_ids = json.load(f)

        with open(texts_path, "r", encoding="utf-8") as f:
            self.doc_texts = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.doc_meta = json.load(f)

        logger.info(
            f"Index loaded. "
            f"Total vectors: {self.index.ntotal:,} | "
            f"Dimension: {self.index.d}"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-K most similar documents to a query vector.

        Args:
            query_vector: float32 array of shape (384,) — L2 normalized
            top_k:        number of results to return

        Returns:
            List of dicts, each containing:
            {
                "rank":        1,          ← 1-indexed, 1 = most similar
                "doc_id":      42,         ← original document ID
                "score":       0.87,       ← cosine similarity (0-1 range)
                "text":        "...",      ← full document text
                "snippet":     "...",      ← first 300 chars
                "category":    "sci.space",← newsgroup category
            }

        HOW FAISS SEARCH WORKS:
            1. FAISS computes inner product between query and ALL stored vectors
            2. Returns indices and scores of the top-K highest values
            3. We map indices back to document IDs using our stored mapping

        WHY top_k=5 default:
            Enough to show variety in results without overwhelming the user.
            The API consumer can override this.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call .build() or .load() first.")

        if top_k > self.index.ntotal:
            logger.warning(
                f"top_k={top_k} exceeds index size {self.index.ntotal}. "
                f"Clamping to {self.index.ntotal}."
            )
            top_k = self.index.ntotal

        # FAISS expects shape (1, dim) for a single query vector
        # WHY reshape: FAISS search always expects a 2D array (batch of queries)
        query_2d = query_vector.reshape(1, -1).astype(np.float32)

        # Run the search
        # scores shape: (1, top_k) — similarity scores
        # indices shape: (1, top_k) — row indices into the index
        scores, indices = self.index.search(query_2d, top_k)

        # Flatten from (1, top_k) → (top_k,) since we have one query
        scores = scores[0]
        indices = indices[0]

        # Build results list
        results = []
        for rank, (faiss_idx, score) in enumerate(zip(indices, scores), start=1):

            # FAISS returns -1 for unfilled slots (shouldn't happen with FlatIP)
            if faiss_idx == -1:
                continue

            # Map FAISS row index → our document data
            doc_id = self.doc_ids[faiss_idx]
            text = self.doc_texts[faiss_idx]
            meta = self.doc_meta[faiss_idx]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 6),
                "text": text,
                "snippet": meta.get("snippet", text[:300]),
                "category": meta.get("category_name", "unknown"),
                "category_id": meta.get("category_id", -1),
                "word_count": meta.get("word_count", 0),
            })

        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """
        Search for multiple queries simultaneously.

        WHY THIS EXISTS:
            FAISS is optimized for batch queries — running 10 queries
            together is faster than 10 individual calls.
            The semantic cache uses this for bulk similarity checks.

        Args:
            query_vectors: float32 array of shape (M, dim)
            top_k:         results per query

        Returns:
            List of M result lists (one per query)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded.")

        query_2d = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores_batch, indices_batch = self.index.search(query_2d, top_k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for rank, (faiss_idx, score) in enumerate(
                zip(indices, scores), start=1
            ):
                if faiss_idx == -1:
                    continue
                doc_id = self.doc_ids[faiss_idx]
                text = self.doc_texts[faiss_idx]
                meta = self.doc_meta[faiss_idx]
                results.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(float(score), 6),
                    "text": text,
                    "snippet": meta.get("snippet", text[:300]),
                    "category": meta.get("category_name", "unknown"),
                    "category_id": meta.get("category_id", -1),
                })
            all_results.append(results)

        return all_results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore("
            f"dim={self.dim}, "
            f"size={self.size:,}, "
            f"index_type={'IndexFlatIP' if self.index else 'None'})"
        )