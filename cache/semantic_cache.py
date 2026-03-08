# cache/semantic_cache.py
"""
Semantic Cache — built entirely from scratch using Python data structures.

NO Redis. NO Memcached. NO caching libraries.

ARCHITECTURE OVERVIEW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Incoming query embedding
          │
          ▼
  ┌───────────────────┐
  │  Cluster Router   │  ← GMM assigns query to dominant cluster
  └────────┬──────────┘       (O(1) — just a matrix multiply)
           │
           ▼
  ┌───────────────────┐
  │  Cluster Bucket   │  ← only ONE bucket searched (not all cache)
  │  [entry, entry,   │       average O(C/n_clusters) comparisons
  │   entry, ...]     │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ Similarity Check  │  ← cosine similarity (dot product, normalized)
  │ score ≥ threshold │       threshold = 0.85 by default
  └────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
   HIT           MISS
    │             │
  return        compute
  cached        result,
  result        store in
                cache
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CACHE ENTRY STRUCTURE:
{
    "id":               str,        ← UUID, unique per entry
    "query":            str,        ← original query text
    "embedding":        np.ndarray, ← shape (384,), L2-normalized
    "result":           dict,       ← full search result payload
    "dominant_cluster": int,        ← which bucket this lives in
    "cluster_probs":    list[float],← full GMM probability vector
    "timestamp":        float,      ← unix time of insertion
    "hits":             int,        ← how many times this was matched
    "last_accessed":    float,      ← unix time of last cache hit
}

EVICTION POLICY — LRU (Least Recently Used):
When cache reaches max_size, evict the entry with the oldest
last_accessed timestamp. Simple, effective, industry standard.

WHY NOT LFU (Least Frequently Used):
LFU unfairly punishes new entries — a query asked 100 times
yesterday but never again would block newer, relevant entries.
LRU naturally keeps the working set of recent queries.
"""

import uuid
import time
import threading
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    Single entry in the semantic cache.

    WHY A DATACLASS:
    Cleaner than a raw dict — type hints, default values,
    and __repr__ for free. Slightly more memory than a dict
    but much more readable and maintainable.
    """
    id: str
    query: str
    embedding: np.ndarray          # shape (384,), L2-normalized float32
    result: dict                   # full API response payload
    dominant_cluster: int          # GMM argmax cluster
    cluster_probs: list            # full GMM probability vector
    timestamp: float               # insertion time (unix)
    hits: int = 0                  # number of cache hits on this entry
    last_accessed: float = field(  # updated on every cache hit
        default_factory=time.time
    )

    def touch(self) -> None:
        """Update access time and increment hit counter."""
        self.last_accessed = time.time()
        self.hits += 1

    def age_seconds(self) -> float:
        """How long has this entry been in cache?"""
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (exclude numpy array)."""
        return {
            "id": self.id,
            "query": self.query,
            "dominant_cluster": self.dominant_cluster,
            "timestamp": self.timestamp,
            "hits": self.hits,
            "last_accessed": self.last_accessed,
            "age_seconds": round(self.age_seconds(), 1),
        }


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Thread-safe, cluster-partitioned semantic cache.

    Key properties:
    ─────────────────────────────────────────────────
    • Similarity matching  — finds semantically equivalent queries
    • Cluster partitioning — O(C/n) lookup instead of O(C)
    • LRU eviction         — bounded memory, always stays under max_size
    • Thread safety        — RLock protects all mutations
    • Zero dependencies    — pure Python + numpy only

    Args:
        similarity_threshold:
            Minimum cosine similarity to declare a cache HIT.
            WHY 0.85:
                • < 0.80: too aggressive — unrelated queries match
                • 0.80–0.90: good range for semantic equivalence
                • > 0.92: too strict — paraphrases are missed
                Tunable via constructor for different use cases.

        max_size:
            Maximum total entries before LRU eviction kicks in.
            WHY 1000: enough for a busy demo/staging system.
            Each entry uses ~6KB (384 floats × 4 bytes + overhead).
            1000 entries ≈ 6MB RAM — negligible.

        n_clusters:
            Must match the GMM model's n_clusters.
            Used to initialize the bucket structure.

        ttl_seconds:
            Time-to-live per entry. Entries older than this are
            treated as expired even if they were recently accessed.
            WHY 3600 (1 hour): search results don't change minute-to-minute
            but SHOULD refresh after an hour (new documents indexed, etc.)
            Set to None to disable TTL.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        n_clusters: int = 20,
        ttl_seconds: Optional[float] = 3600.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.n_clusters = n_clusters
        self.ttl_seconds = ttl_seconds

        # ── Core storage ──────────────────────────────────────────────
        # Primary store: entry_id → CacheEntry
        # O(1) lookup/delete by ID (used during eviction)
        self._store: dict[str, CacheEntry] = {}

        # Cluster buckets: cluster_id → list of entry IDs
        # WHY list of IDs (not entries): avoids data duplication.
        # We store the full entry once in _store, reference by ID here.
        self._buckets: dict[int, list[str]] = defaultdict(list)

        # ── Thread safety ─────────────────────────────────────────────
        # WHY RLock (reentrant lock) not Lock:
        # Some methods call other methods that also acquire the lock.
        # RLock allows the same thread to re-acquire without deadlocking.
        self._lock = threading.RLock()

        # ── Statistics ────────────────────────────────────────────────
        self._stats = {
            "total_queries":    0,   # every lookup (hit or miss)
            "cache_hits":       0,   # similarity threshold exceeded
            "cache_misses":     0,   # threshold not met, new computation
            "evictions":        0,   # LRU evictions performed
            "expired_removals": 0,   # TTL expiry removals
            "total_stored":     0,   # cumulative entries ever stored
        }

        logger.info(
            f"SemanticCache initialized: "
            f"threshold={similarity_threshold}, "
            f"max_size={max_size}, "
            f"n_clusters={n_clusters}, "
            f"ttl={ttl_seconds}s"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        query_embedding: np.ndarray,
        dominant_cluster: int,
    ) -> Optional[dict]:
        """
        Look up a query in the cache.

        LOOKUP ALGORITHM:
        ┌─────────────────────────────────────────────────────┐
        │ 1. Acquire lock (thread safety)                     │
        │ 2. Increment total_queries counter                  │
        │ 3. Get the bucket for dominant_cluster              │
        │ 4. For each entry ID in bucket:                     │
        │      a. Fetch entry from _store                     │
        │      b. Check TTL — skip if expired                 │
        │      c. Compute cosine similarity (dot product)     │
        │      d. If similarity ≥ threshold → CACHE HIT       │
        │           - touch() the entry (LRU update)          │
        │           - return cached result with metadata      │
        │ 5. No match found → return None (CACHE MISS)        │
        └─────────────────────────────────────────────────────┘

        Args:
            query:            raw query string (for response metadata)
            query_embedding:  L2-normalized float32 array (384,)
            dominant_cluster: GMM argmax cluster for this query

        Returns:
            dict with cache metadata + original result, or None on miss
        """
        with self._lock:
            self._stats["total_queries"] += 1

            # Get candidate entry IDs from this cluster's bucket
            bucket = self._buckets.get(dominant_cluster, [])

            if not bucket:
                # Empty bucket → definite miss, skip all comparisons
                self._stats["cache_misses"] += 1
                return None

            # Search within bucket for a similar cached query
            best_score = -1.0
            best_entry: Optional[CacheEntry] = None

            for entry_id in bucket:
                entry = self._store.get(entry_id)
                if entry is None:
                    continue  # stale ID in bucket (shouldn't happen)

                # TTL check — skip expired entries
                if self._is_expired(entry):
                    continue

                # Cosine similarity via dot product
                # WHY dot product works: both vectors are L2-normalized,
                # so ||a|| = ||b|| = 1, and dot(a,b) = cos(angle(a,b))
                score = float(np.dot(query_embedding, entry.embedding))

                if score > best_score:
                    best_score = score
                    best_entry = entry

            # Threshold check
            if best_entry is not None and best_score >= self.similarity_threshold:
                # ── CACHE HIT ──
                best_entry.touch()      # update LRU timestamp + hit count
                self._stats["cache_hits"] += 1

                logger.info(
                    f"CACHE HIT | "
                    f"score={best_score:.4f} | "
                    f"cluster={dominant_cluster} | "
                    f"matched='{best_entry.query[:60]}...'"
                )

                # Return hit metadata + the cached result
                return {
                    "cache_hit": True,
                    "matched_query": best_entry.query,
                    "similarity_score": round(best_score, 6),
                    "dominant_cluster": dominant_cluster,
                    "entry_id": best_entry.id,
                    "entry_hits": best_entry.hits,
                    "result": best_entry.result,
                }

            # ── CACHE MISS ──
            self._stats["cache_misses"] += 1
            logger.info(
                f"CACHE MISS | "
                f"best_score={best_score:.4f} | "
                f"threshold={self.similarity_threshold} | "
                f"cluster={dominant_cluster} | "
                f"bucket_size={len(bucket)}"
            )
            return None

    def put(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: dict,
        dominant_cluster: int,
        cluster_probs: list,
    ) -> str:
        """
        Store a new query + result in the cache.

        STORAGE ALGORITHM:
        ┌─────────────────────────────────────────────────────┐
        │ 1. Acquire lock                                     │
        │ 2. Evict if at capacity (LRU eviction)              │
        │ 3. Create CacheEntry with UUID                      │
        │ 4. Store in _store (primary)                        │
        │ 5. Add entry ID to correct cluster bucket           │
        │ 6. Increment stats                                  │
        └─────────────────────────────────────────────────────┘

        Args:
            query:            raw query string
            query_embedding:  L2-normalized float32 array (384,)
            result:           full search result dict to cache
            dominant_cluster: GMM argmax cluster for routing
            cluster_probs:    full probability vector (for inspection)

        Returns:
            entry_id: UUID string of the created cache entry
        """
        with self._lock:
            # Evict before inserting to maintain max_size invariant
            if len(self._store) >= self.max_size:
                self._evict_lru()

            # Create entry
            entry_id = str(uuid.uuid4())
            entry = CacheEntry(
                id=entry_id,
                query=query,
                embedding=query_embedding.copy(),  # WHY copy: prevent mutation
                result=result,
                dominant_cluster=dominant_cluster,
                cluster_probs=cluster_probs,
                timestamp=time.time(),
                last_accessed=time.time(),
            )

            # Store in primary dict
            self._store[entry_id] = entry

            # Add to cluster bucket
            self._buckets[dominant_cluster].append(entry_id)

            # Update stats
            self._stats["total_stored"] += 1

            logger.info(
                f"CACHE STORE | "
                f"cluster={dominant_cluster} | "
                f"bucket_size={len(self._buckets[dominant_cluster])} | "
                f"total_entries={len(self._store)} | "
                f"query='{query[:60]}'"
            )

            return entry_id

    def invalidate(self, entry_id: str) -> bool:
        """
        Remove a specific cache entry by ID.

        WHY THIS EXISTS:
            If underlying documents are updated/reindexed, cached results
            become stale. The API exposes DELETE /cache which calls
            clear(). Individual invalidation is for future fine-grained
            cache management.

        Returns:
            True if entry was found and removed, False otherwise
        """
        with self._lock:
            entry = self._store.pop(entry_id, None)
            if entry is None:
                return False

            # Remove from bucket
            bucket = self._buckets.get(entry.dominant_cluster, [])
            if entry_id in bucket:
                bucket.remove(entry_id)

            logger.info(f"Entry invalidated: {entry_id}")
            return True

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._buckets.clear()

            # Reset hit/miss stats but keep cumulative totals
            self._stats["cache_hits"] = 0
            self._stats["cache_misses"] = 0
            self._stats["total_queries"] = 0

            logger.info(f"Cache cleared: {count} entries removed")
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all TTL-expired entries from cache and buckets.

        WHEN TO CALL THIS:
            The FastAPI app calls this periodically (every N requests)
            to prevent expired entries from accumulating.
            In production you'd run this in a background thread.

        Returns:
            Number of expired entries removed
        """
        with self._lock:
            if self.ttl_seconds is None:
                return 0

            expired_ids = [
                eid for eid, entry in self._store.items()
                if self._is_expired(entry)
            ]

            for eid in expired_ids:
                entry = self._store.pop(eid)
                bucket = self._buckets.get(entry.dominant_cluster, [])
                if eid in bucket:
                    bucket.remove(eid)

            self._stats["expired_removals"] += len(expired_ids)

            if expired_ids:
                logger.info(
                    f"Expired {len(expired_ids)} cache entries "
                    f"(TTL={self.ttl_seconds}s)"
                )

            return len(expired_ids)

    # ------------------------------------------------------------------
    # Statistics & inspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return cache statistics for the GET /cache/stats endpoint.

        All reads are inside the lock to get a consistent snapshot.
        """
        with self._lock:
            total_q = self._stats["total_queries"]
            hits = self._stats["cache_hits"]

            # Hit rate: fraction of queries served from cache
            hit_rate = (hits / total_q) if total_q > 0 else 0.0

            # Bucket distribution — shows which clusters are busiest
            bucket_sizes = {
                cluster_id: len(ids)
                for cluster_id, ids in self._buckets.items()
                if ids   # skip empty buckets
            }

            # Average hits per entry (cache efficiency measure)
            avg_hits_per_entry = (
                np.mean([e.hits for e in self._store.values()])
                if self._store else 0.0
            )

            return {
                # Traffic stats
                "total_queries":     self._stats["total_queries"],
                "cache_hits":        self._stats["cache_hits"],
                "cache_misses":      self._stats["cache_misses"],
                "hit_rate_pct":      round(hit_rate * 100, 2),

                # Storage stats
                "current_size":      len(self._store),
                "max_size":          self.max_size,
                "capacity_pct":      round(len(self._store) / self.max_size * 100, 1),
                "total_stored":      self._stats["total_stored"],

                # Eviction stats
                "evictions":         self._stats["evictions"],
                "expired_removals":  self._stats["expired_removals"],

                # Config
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds":          self.ttl_seconds,
                "n_clusters":           self.n_clusters,

                # Efficiency
                "avg_hits_per_entry":   round(float(avg_hits_per_entry), 2),
                "bucket_distribution":  bucket_sizes,
            }

    def get_all_entries(self) -> list[dict]:
        """
        Return serialized view of all cache entries.
        Used by GET /cache/stats for detailed inspection.
        Excludes embedding arrays (not JSON-serializable).
        """
        with self._lock:
            return [
                entry.to_dict()
                for entry in sorted(
                    self._store.values(),
                    key=lambda e: e.last_accessed,
                    reverse=True   # most recently accessed first
                )
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """
        Evict the least recently used cache entry.

        LRU EVICTION LOGIC:
        ┌──────────────────────────────────────────────────────┐
        │ Find entry with smallest last_accessed timestamp     │
        │  → this is the one unused for the longest time       │
        │ Remove from _store and from its cluster bucket       │
        └──────────────────────────────────────────────────────┘

        WHY NOT A DOUBLY-LINKED LIST (like OrderedDict):
            For max_size=1000, a linear scan over 1000 entries is
            microseconds — not worth the complexity of a full LRU
            data structure. At max_size=100,000+ we'd switch to
            collections.OrderedDict for O(1) LRU.

        CALLED BY: put() before inserting when at capacity.
        Must be called with lock held.
        """
        if not self._store:
            return

        # Find LRU entry — min by last_accessed
        lru_id = min(
            self._store.keys(),
            key=lambda eid: self._store[eid].last_accessed
        )
        lru_entry = self._store.pop(lru_id)

        # Remove from bucket
        bucket = self._buckets.get(lru_entry.dominant_cluster, [])
        if lru_id in bucket:
            bucket.remove(lru_id)

        self._stats["evictions"] += 1

        logger.info(
            f"LRU EVICTION | "
            f"entry='{lru_entry.query[:50]}' | "
            f"age={lru_entry.age_seconds():.0f}s | "
            f"hits={lru_entry.hits}"
        )

    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry has exceeded its TTL.

        WHY CHECK last_accessed NOT timestamp:
            We reset TTL on each access. A frequently-used cached result
            should NOT expire just because it was stored a long time ago.
            last_accessed = "last time it was useful"

        Returns:
            True if expired, False if still valid
        """
        if self.ttl_seconds is None:
            return False
        return (time.time() - entry.last_accessed) > self.ttl_seconds

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SemanticCache("
            f"size={stats['current_size']}/{self.max_size}, "
            f"hit_rate={stats['hit_rate_pct']}%, "
            f"threshold={self.similarity_threshold})"
        )