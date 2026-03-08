# api/schemas.py
"""
Pydantic models for all API request and response payloads.

WHY PYDANTIC:
    FastAPI uses Pydantic for automatic:
    - Request body validation (wrong types → 422 error with clear message)
    - Response serialization (Python objects → JSON)
    - OpenAPI schema generation (powers /docs Swagger UI)

    Defining schemas here means validation logic lives in ONE place —
    not scattered across route handlers.

DESIGN PRINCIPLE — Separate Request and Response models:
    Request models validate INCOMING data (strict)
    Response models shape OUTGOING data (permissive, with defaults)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Input for POST /query

    WHY min_length=2 on query:
        Single-character queries ("a", "?") produce meaningless
        embeddings and waste compute. Enforce at the schema level.

    WHY max_length=1000:
        Very long queries (essays) are unusual for search and would
        slow down the embedding step. Sentence transformers also
        have a token limit (~256 tokens = ~1000 chars).
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Natural language search query",
        examples=["latest developments in space exploration"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to return (1-20)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override cache similarity threshold for this request. "
            "If None, uses the server default (0.85)."
        ),
    )

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        """Strip whitespace and reject blank queries."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "space shuttle missions NASA",
                    "top_k": 5,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DocumentResult(BaseModel):
    """
    A single document returned from the vector search.
    Nested inside QueryResponse.results
    """
    rank: int = Field(description="1-indexed rank (1 = most similar)")
    doc_id: int = Field(description="Internal document ID")
    score: float = Field(description="Cosine similarity score (0-1)")
    snippet: str = Field(description="First 300 characters of document")
    category: str = Field(description="Newsgroup category name")
    word_count: int = Field(description="Document word count")


class ClusterMembership(BaseModel):
    """
    Fuzzy cluster membership for a query.
    Shows which topic clusters the query belongs to and with what probability.
    """
    dominant_cluster: int = Field(
        description="Cluster with highest membership probability"
    )
    dominant_prob: float = Field(
        description="Probability of dominant cluster (0-1)"
    )
    top_clusters: list[tuple[int, float]] = Field(
        description="Top 3 clusters as (cluster_id, probability) pairs"
    )


class QueryResponse(BaseModel):
    """
    Full response for POST /query

    Designed to be informative for both end users and developers:
    - cache_hit tells you if this was a fresh search or served from cache
    - matched_query shows WHICH cached query was matched (transparency)
    - similarity_score shows HOW similar the cache match was
    - cluster_info shows WHERE in topic space this query lives
    - results contains the actual documents
    """
    # Query metadata
    query: str = Field(description="The original query string")
    processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )

    # Cache information
    cache_hit: bool = Field(description="True if result was served from cache")
    matched_query: Optional[str] = Field(
        default=None,
        description="The cached query that matched (only present on cache hit)"
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Similarity between query and cached query (cache hit only)"
    )

    # Cluster information
    cluster_info: ClusterMembership = Field(
        description="Fuzzy cluster membership probabilities for this query"
    )

    # Search results
    results: list[DocumentResult] = Field(
        description="Top-K most similar documents"
    )
    total_results: int = Field(description="Number of documents returned")

    # Server metadata
    timestamp: str = Field(description="ISO timestamp of the response")


class CacheStatsResponse(BaseModel):
    """Response for GET /cache/stats"""
    # Traffic
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float

    # Storage
    current_size: int
    max_size: int
    capacity_pct: float
    total_stored: int

    # Eviction
    evictions: int
    expired_removals: int

    # Config
    similarity_threshold: float
    ttl_seconds: Optional[float]
    n_clusters: int

    # Efficiency
    avg_hits_per_entry: float
    bucket_distribution: dict[str, int]

    # Entries (recent)
    recent_entries: list[dict]


class CacheClearResponse(BaseModel):
    """Response for DELETE /cache"""
    message: str
    entries_removed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status: str
    components: dict[str, str]
    index_size: int
    cache_size: int
    timestamp: str