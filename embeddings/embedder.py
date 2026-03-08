# embeddings/embedder.py
"""
Embedding generation module.

DESIGN DECISION — WHY A CLASS WRAPPER:
We wrap SentenceTransformer in our own class so that:
1. The rest of the codebase never imports SentenceTransformer directly
   — if we ever swap models, we change ONE file, not ten.
2. We can add preprocessing, caching, batching logic in one place.
3. It's easier to mock in unit tests.

This is the "Adapter" design pattern.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WHY this model: best balance of speed, size, and semantic quality.
# It was distilled from larger BERT models specifically for sentence
# similarity tasks — exactly what we need for search.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimensionality for this model
EMBEDDING_DIM = 384


class Embedder:
    """
    Wraps a SentenceTransformer model with batched encoding,
    normalization, and a clean interface for the rest of the system.

    Usage:
        embedder = Embedder()
        vector = embedder.encode("space exploration news")
        matrix = embedder.encode_batch(["doc1 text", "doc2 text"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Load the embedding model.

        Args:
            model_name: HuggingFace model identifier or local path.
                        First call downloads the model (~22MB).
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name

        # SentenceTransformer handles model downloading + caching automatically
        self._model = SentenceTransformer(model_name)

        logger.info(
            f"Model loaded. Embedding dimension: {EMBEDDING_DIM}"
        )

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single string into a dense vector.

        Args:
            text:      input string
            normalize: if True, L2-normalize the vector to unit length

        Returns:
            numpy array of shape (384,)

        WHY NORMALIZE:
            L2 normalization (making the vector length = 1) means that
            cosine similarity becomes equivalent to dot product.
            Dot product is MUCH faster to compute, especially in FAISS.
            All vectors on the unit hypersphere — direction = meaning.
        """
        if not text or not text.strip():
            # Return a zero vector for empty input — safe fallback
            logger.warning("Empty text passed to encode(); returning zero vector")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )

        return vector.astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of documents into an embedding matrix.

        Args:
            texts:          list of document strings
            batch_size:     documents per forward pass through the model.
                            WHY 64: fits comfortably in CPU RAM;
                            larger batches = faster but more memory.
            normalize:      L2-normalize each row vector
            show_progress:  display tqdm progress bar

        Returns:
            numpy array of shape (N, 384) where N = len(texts)

        WHY BATCHING:
            Sending all 16,000 docs at once would spike RAM.
            Sending one-by-one is slow (no parallelism).
            Batches of 64 balance throughput and memory usage.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        logger.info(
            f"Encoding {len(texts):,} documents "
            f"(batch_size={batch_size})..."
        )

        matrix = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

        logger.info(
            f"Encoding complete. Matrix shape: {matrix.shape} "
            f"| dtype: {matrix.dtype} "
            f"| Memory: {matrix.nbytes / 1_000_000:.1f} MB"
        )

        return matrix.astype(np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        WHY STATIC: Doesn't need model state — pure math utility.

        Returns:
            float in range [-1.0, 1.0]
            1.0  = identical direction (same meaning)
            0.0  = orthogonal (unrelated)
           -1.0  = opposite direction (antonyms — rare in practice)
        """
        # If both vectors are L2-normalized, dot product == cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return EMBEDDING_DIM

    def __repr__(self) -> str:
        return f"Embedder(model='{self.model_name}', dim={self.dim})"