# api/app.py
"""
FastAPI application — route definitions and request handling.

ARCHITECTURE DECISIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LIFESPAN (not @app.on_event):
   FastAPI deprecated on_event in favor of the lifespan context manager.
   It clearly separates startup and shutdown logic, and supports
   async resource management (database connections, etc.)

2. DEPENDENCY INJECTION for state:
   Instead of importing app_state directly in every route,
   we use a FastAPI Depends() function. This makes routes
   testable — you can inject a mock state in tests.

3. SYNCHRONOUS ROUTES (not async):
   Our operations are CPU-bound (numpy, FAISS) not I/O-bound.
   Making them async wouldn't help — they can't yield the event loop
   during numpy operations. Uvicorn runs sync routes in a threadpool
   automatically via run_in_executor, giving us concurrency anyway.

4. STRUCTURED ERROR HANDLING:
   Every route has explicit try/except with meaningful HTTP status codes.
   500 errors include enough detail for debugging without leaking internals.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentResult,
    ClusterMembership,
    CacheStatsResponse,
    CacheClearResponse,
    HealthResponse,
)
from api.state import AppState, app_state
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Code BEFORE yield → runs at startup
    Code AFTER yield  → runs at shutdown

    WHY THIS PATTERN:
        Guarantees cleanup even if startup fails partway through.
        Equivalent to try/finally but cleaner for async contexts.
    """
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("FastAPI application starting up...")
    try:
        app_state.initialize()
    except FileNotFoundError as e:
        logger.error(f"Startup failed — missing model artifact: {e}")
        logger.error(
            "Run the full pipeline first:\n"
            "  python scripts/download_data.py\n"
            "  python scripts/preprocess.py\n"
            "  python scripts/build_embeddings.py\n"
            "  python scripts/build_index.py\n"
            "  python scripts/build_clusters.py"
        )
        raise
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield   # ← application runs here, handling requests

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("FastAPI application shutting down...")
    # In production: close DB connections, flush buffers, etc.
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search System",
    description=(
        "Production-quality semantic search over the 20 Newsgroups dataset.\n\n"
        "Features:\n"
        "- Dense vector search with FAISS\n"
        "- Fuzzy clustering with GMM (documents belong to multiple clusters)\n"
        "- Semantic cache (finds similar past queries — no Redis needed)\n"
        "- Sentence-transformers embeddings (all-MiniLM-L6-v2)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
)

# CORS middleware — allows the API to be called from browsers/frontends
# WHY allow_origins=["*"]: development convenience.
# In production: restrict to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def get_state() -> AppState:
    """
    FastAPI dependency: provides the shared AppState to route handlers.

    WHY A DEPENDENCY:
        Routes declare `state: Annotated[AppState, Depends(get_state)]`
        instead of importing app_state directly.
        This means in tests we can override this dependency with a
        mock state: app.dependency_overrides[get_state] = mock_state_fn
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing. Please retry in a moment."
        )
    return app_state


# Type alias for cleaner route signatures
State = Annotated[AppState, Depends(get_state)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(state: State) -> HealthResponse:
    """
    GET /health

    Returns the health status of all system components.
    Used by load balancers, monitoring systems, and Docker healthchecks.

    Returns 200 if all components are loaded and ready.
    Returns 503 (via dependency) if still initializing.
    """
    return HealthResponse(
        status="healthy",
        components={
            "embedder":     "ready" if state.embedder else "not loaded",
            "vector_store": "ready" if state.vector_store else "not loaded",
            "clusterer":    "ready" if state.clusterer else "not loaded",
            "cache":        "ready" if state.cache else "not loaded",
        },
        index_size=state.vector_store.size,
        cache_size=len(state.cache),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Search"],
    summary="Semantic search with cache",
    responses={
        200: {"description": "Search results (cache hit or fresh search)"},
        422: {"description": "Invalid request body"},
        500: {"description": "Internal server error during search"},
        503: {"description": "Service not ready"},
    }
)
def query(request: QueryRequest, state: State) -> QueryResponse:
    """
    POST /query

    Core endpoint. Performs semantic search with the following pipeline:

    1. Check semantic cache (returns instantly if similar query was seen)
    2. Encode query with SentenceTransformer
    3. Assign fuzzy cluster memberships via GMM
    4. Search FAISS index for top-K nearest documents
    5. Store result in semantic cache
    6. Return structured response

    The `cache_hit` field tells you if step 1 short-circuited the rest.
    """
    start_time = time.perf_counter()

    try:
        # ── Step 1: Encode the query ──────────────────────────────────
        # We ALWAYS encode first — we need the embedding for both
        # cache lookup AND (on miss) for FAISS search.
        query_embedding = state.embedder.encode(
            request.query,
            normalize=True
        )

        # ── Step 2: Get cluster assignment ───────────────────────────
        # Needed for:
        # (a) routing the cache lookup to the right bucket
        # (b) including cluster info in the response
        cluster_result = state.clusterer.predict_single(query_embedding)
        dominant_cluster = cluster_result["dominant_cluster"]

        # ── Step 3: Check semantic cache ─────────────────────────────
        # Use request's threshold override if provided, else default
        threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else state.cache.similarity_threshold
        )

        # Temporarily set threshold if overridden
        original_threshold = state.cache.similarity_threshold
        if request.similarity_threshold is not None:
            state.cache.similarity_threshold = threshold

        cache_result = state.cache.get(
            query=request.query,
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
        )

        # Restore threshold
        state.cache.similarity_threshold = original_threshold

        # Periodic TTL cleanup
        state.maybe_cleanup_cache()

        # ── Step 4: Cache HIT path ────────────────────────────────────
        if cache_result is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            cached_response = cache_result["result"]

            return QueryResponse(
                query=request.query,
                processing_time_ms=round(elapsed_ms, 3),
                cache_hit=True,
                matched_query=cache_result["matched_query"],
                similarity_score=cache_result["similarity_score"],
                cluster_info=ClusterMembership(
                    dominant_cluster=dominant_cluster,
                    dominant_prob=cluster_result["dominant_prob"],
                    top_clusters=cluster_result["top_clusters"],
                ),
                results=cached_response["results"],
                total_results=cached_response["total_results"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # ── Step 5: Cache MISS — run full search ──────────────────────
        faiss_results = state.vector_store.search(
            query_vector=query_embedding,
            top_k=request.top_k,
        )

        # ── Step 6: Shape results ─────────────────────────────────────
        document_results = [
            DocumentResult(
                rank=r["rank"],
                doc_id=r["doc_id"],
                score=r["score"],
                snippet=r["snippet"],
                category=r["category"],
                word_count=r["word_count"],
            )
            for r in faiss_results
        ]

        # ── Step 7: Store in cache ────────────────────────────────────
        # We store the shaped results (not raw FAISS output) so that
        # cache hits return the exact same format as cache misses.
        cache_payload = {
            "results": document_results,
            "total_results": len(document_results),
        }

        state.cache.put(
            query=request.query,
            query_embedding=query_embedding,
            result=cache_payload,
            dominant_cluster=dominant_cluster,
            cluster_probs=cluster_result["all_probabilities"],
        )

        # ── Step 8: Build and return response ────────────────────────
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            processing_time_ms=round(elapsed_ms, 3),
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            cluster_info=ClusterMembership(
                dominant_cluster=dominant_cluster,
                dominant_prob=cluster_result["dominant_prob"],
                top_clusters=cluster_result["top_clusters"],
            ),
            results=document_results,
            total_results=len(document_results),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise   # re-raise FastAPI exceptions unchanged

    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Cache"],
    summary="Get semantic cache statistics",
)
def get_cache_stats(state: State) -> CacheStatsResponse:
    """
    GET /cache/stats

    Returns detailed cache performance metrics including:
    - Hit/miss rates (how effective is the cache?)
    - Bucket distribution (which clusters are most queried?)
    - Eviction stats (is max_size too small?)
    - Recent entries (what's currently cached?)

    Useful for monitoring and tuning the similarity threshold.
    """
    raw_stats = state.cache.get_stats()
    recent_entries = state.cache.get_all_entries()[:20]  # last 20

    # Convert bucket keys from int to str for JSON serialization
    bucket_dist = {
        str(k): v
        for k, v in raw_stats["bucket_distribution"].items()
    }

    return CacheStatsResponse(
        total_queries=raw_stats["total_queries"],
        cache_hits=raw_stats["cache_hits"],
        cache_misses=raw_stats["cache_misses"],
        hit_rate_pct=raw_stats["hit_rate_pct"],
        current_size=raw_stats["current_size"],
        max_size=raw_stats["max_size"],
        capacity_pct=raw_stats["capacity_pct"],
        total_stored=raw_stats["total_stored"],
        evictions=raw_stats["evictions"],
        expired_removals=raw_stats["expired_removals"],
        similarity_threshold=raw_stats["similarity_threshold"],
        ttl_seconds=raw_stats["ttl_seconds"],
        n_clusters=raw_stats["n_clusters"],
        avg_hits_per_entry=raw_stats["avg_hits_per_entry"],
        bucket_distribution=bucket_dist,
        recent_entries=recent_entries,
    )


@app.delete(
    "/cache",
    response_model=CacheClearResponse,
    tags=["Cache"],
    summary="Clear all cached queries",
)
def clear_cache(state: State) -> CacheClearResponse:
    """
    DELETE /cache

    Clears all entries from the semantic cache.

    WHEN TO USE:
    - After reindexing documents (cached results are now stale)
    - When testing with a fresh cache
    - After changing the similarity threshold significantly

    This is idempotent — calling it on an empty cache is safe.
    """
    removed = state.cache.clear()
    logger.info(f"Cache cleared via API: {removed} entries removed")

    return CacheClearResponse(
        message=f"Cache cleared successfully. {removed} entries removed.",
        entries_removed=removed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "endpoint_not_found",
            "message": f"The endpoint '{request.url.path}' does not exist.",
            "available_endpoints": [
                "POST /query",
                "GET  /cache/stats",
                "DELETE /cache",
                "GET  /health",
                "GET  /docs",
            ]
        }
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Check server logs.",
        }
    )