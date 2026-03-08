# api/state.py
"""
Application-level shared state.

THE PROBLEM THIS SOLVES:
    FastAPI handles requests in async coroutines across multiple threads.
    All requests need access to the same embedder, FAISS index, clusterer,
    and cache — but we can't recreate them per request (too expensive).

    Solution: A single AppState instance, initialized once at startup,
    shared across all requests via FastAPI's app.state.

WHY NOT GLOBAL VARIABLES:
    Global variables in Python are:
    - Hard to test (can't reset between tests)
    - Not explicit (hidden dependencies)
    - Tricky with async (import order, circular imports)

    A state class makes dependencies explicit and injectable.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSVectorStore
from clustering.fuzzy_cluster import FuzzyClusterer
from cache.semantic_cache import SemanticCache
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model artifact paths — all consumed at startup
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

FAISS_INDEX_PATH  = os.path.join(MODELS_DIR, "faiss_index.bin")
FAISS_IDS_PATH    = os.path.join(MODELS_DIR, "faiss_doc_ids.json")
FAISS_TEXTS_PATH  = os.path.join(MODELS_DIR, "faiss_doc_texts.json")
FAISS_META_PATH   = os.path.join(MODELS_DIR, "faiss_doc_metadata.json")
GMM_MODEL_PATH    = os.path.join(MODELS_DIR, "gmm_model.pkl")

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------
CACHE_SIMILARITY_THRESHOLD = 0.75
CACHE_MAX_SIZE             = 1000
CACHE_TTL_SECONDS          = 3600.0   # 1 hour
N_CLUSTERS                 = 20

# How often to run TTL cleanup (every N queries)
# WHY NOT a background thread: keeps the app simple and stateless.
# At 1000 queries/hour, cleanup runs every ~10 seconds — sufficient.
CLEANUP_INTERVAL = 100


@dataclass
class AppState:
    """
    Container for all shared application components.

    Initialized once during FastAPI lifespan startup.
    All fields start as None and are populated by initialize().
    The is_ready flag gates request handling — requests before
    initialization complete will receive a 503 Service Unavailable.
    """
    embedder:   Optional[Embedder]          = field(default=None)
    vector_store: Optional[FAISSVectorStore] = field(default=None)
    clusterer:  Optional[FuzzyClusterer]    = field(default=None)
    cache:      Optional[SemanticCache]     = field(default=None)
    is_ready:   bool                        = field(default=False)
    query_count: int                        = field(default=0)

    def initialize(self) -> None:
        """
        Load all model artifacts and initialize components.

        STARTUP ORDER MATTERS:
            1. Embedder    — needed to encode queries at runtime
            2. VectorStore — needs dim=384 matching embedder
            3. Clusterer   — needs fitted PCA + GMM from disk
            4. Cache       — needs n_clusters matching clusterer

        Called once from the FastAPI lifespan context manager.
        Any failure here raises an exception that prevents startup —
        better to fail fast than to serve broken results silently.
        """
        logger.info("=" * 55)
        logger.info("  Initializing application state...")
        logger.info("=" * 55)

        # 1. Embedding model
        logger.info("[1/4] Loading embedding model...")
        self.embedder = Embedder()

        # 2. FAISS vector store
        logger.info("[2/4] Loading FAISS vector store...")
        self.vector_store = FAISSVectorStore(dim=384)
        self.vector_store.load(
            index_path=FAISS_INDEX_PATH,
            ids_path=FAISS_IDS_PATH,
            texts_path=FAISS_TEXTS_PATH,
            meta_path=FAISS_META_PATH,
        )
        logger.info(f"  Vector store ready: {self.vector_store.size:,} documents")

        # 3. Fuzzy clusterer
        logger.info("[3/4] Loading GMM cluster model...")
        self.clusterer = FuzzyClusterer.load(GMM_MODEL_PATH)
        logger.info(
            f"  Clusterer ready: "
            f"{self.clusterer.n_clusters} clusters"
        )

        # 4. Semantic cache
        logger.info("[4/4] Initializing semantic cache...")
        self.cache = SemanticCache(
            similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
            max_size=CACHE_MAX_SIZE,
            n_clusters=N_CLUSTERS,
            ttl_seconds=CACHE_TTL_SECONDS,
        )
        logger.info(
            f"  Cache ready: "
            f"threshold={CACHE_SIMILARITY_THRESHOLD}, "
            f"max_size={CACHE_MAX_SIZE}"
        )

        self.is_ready = True
        logger.info("=" * 55)
        logger.info("  ✅ Application ready to serve requests")
        logger.info("=" * 55)

    def maybe_cleanup_cache(self) -> None:
        """
        Run TTL cleanup every CLEANUP_INTERVAL queries.

        WHY HERE (not background thread):
            Simpler architecture for a single-process service.
            Cleanup is fast (microseconds per entry scan) and
            amortized across requests.
        """
        self.query_count += 1
        if self.query_count % CLEANUP_INTERVAL == 0:
            removed = self.cache.cleanup_expired()
            if removed:
                logger.info(
                    f"Periodic cleanup: removed {removed} expired entries "
                    f"(query #{self.query_count})"
                )


# Singleton instance — imported by app.py
app_state = AppState()