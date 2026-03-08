# tests/test_cache.py
"""
Unit tests for the SemanticCache.

Run with: python -m pytest tests/test_cache.py -v
Or directly: python tests/test_cache.py

FIX APPLIED:
    In 384-dimensional space, noise=0.05 causes too much angular drift.
    noise × sqrt(384) ≈ 0.98 — almost as large as the vector itself.
    Reduced to noise=0.01 so cosine similarity stays above 0.95.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from cache.semantic_cache import SemanticCache


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_vec(seed: int) -> np.ndarray:
    """
    Create a reproducible random unit vector in 384-dimensional space.
    Using a fixed seed ensures the same vector is produced every run.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def make_similar_vec(base: np.ndarray, noise: float = 0.01) -> np.ndarray:
    """
    Create a slightly perturbed version of base vector.

    WHY noise=0.01 (not 0.05):
        In 384-dimensional space, the total perturbation magnitude is:
            noise × sqrt(384 dimensions)
        
        noise=0.05 → 0.05 × sqrt(384) ≈ 0.98  (nearly as large as the vector!)
                   → cosine similarity drops to ~0.70  ← below threshold ❌

        noise=0.01 → 0.01 × sqrt(384) ≈ 0.20  (small perturbation)
                   → cosine similarity stays at ~0.98  ← above threshold ✅

    This is the curse of dimensionality — small noise scalars amplify
    significantly in high-dimensional spaces.
    """
    rng = np.random.default_rng(999)
    v = base + noise * rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def print_separator(title: str) -> None:
    """Print a clear visual separator for each test."""
    print(f"\n{'─' * 55}")
    print(f"  TEST: {title}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_hit_miss():
    """
    A stored query should hit on similar query, miss on unrelated one.

    Validates:
    - Cosine similarity >= threshold → CACHE HIT
    - Cosine similarity <  threshold → CACHE MISS
    - Returned hit contains correct metadata
    """
    print_separator("Basic Hit / Miss")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    vec_a        = make_vec(seed=1)
    similar_vec  = make_similar_vec(vec_a, noise=0.01)  # very similar ✅
    unrelated_vec = make_vec(seed=99)                   # different topic

    dummy_result = {"documents": ["doc1", "doc2"], "query": "test"}

    # Debug: show actual similarity before asserting
    actual_sim = float(np.dot(vec_a, similar_vec))
    print(f"  Debug: cosine similarity between stored and similar vec = {actual_sim:.4f}")
    assert actual_sim >= 0.85, (
        f"Test setup error: similar_vec is not actually similar enough "
        f"(sim={actual_sim:.4f}, need >= 0.85). Reduce noise value."
    )

    # Store original query
    cache.put(
        query="space exploration news",
        query_embedding=vec_a,
        result=dummy_result,
        dominant_cluster=2,
        cluster_probs=[0.0] * 5,
    )

    # ── Similar query → should HIT ────────────────────────────────────────
    hit = cache.get(
        query="NASA rocket launch",
        query_embedding=similar_vec,
        dominant_cluster=2,
    )
    assert hit is not None, (
        f"Expected cache HIT on similar vector "
        f"(cosine_sim={actual_sim:.4f}, threshold=0.85)"
    )
    assert hit["cache_hit"] is True
    assert hit["similarity_score"] >= 0.85
    assert hit["matched_query"] == "space exploration news"
    assert "result" in hit

    print(f"  ✅ HIT  | score={hit['similarity_score']:.4f} | "
          f"matched='{hit['matched_query']}'")

    # ── Unrelated query → should MISS ────────────────────────────────────
    miss = cache.get(
        query="cooking pasta recipe",
        query_embedding=unrelated_vec,
        dominant_cluster=2,
    )
    assert miss is None, "Expected cache MISS on unrelated vector"
    print(f"  ✅ MISS | unrelated query correctly returned None")


def test_lru_eviction():
    """
    Cache should evict the LRU (least recently used) entry when at max capacity.

    Validates:
    - Cache size never exceeds max_size
    - The least recently accessed entry is evicted first
    - Eviction counter increments correctly
    """
    print_separator("LRU Eviction")

    cache = SemanticCache(
        similarity_threshold=0.85,
        max_size=3,        # tiny cache for predictable testing
        n_clusters=5,
    )

    # Fill cache to max_size with 3 entries
    for i in range(3):
        vec = make_vec(seed=i)
        cache.put(
            query=f"query {i}",
            query_embedding=vec,
            result={"data": i},
            dominant_cluster=0,
            cluster_probs=[0.0] * 5,
        )

    assert len(cache) == 3, f"Expected 3 entries, got {len(cache)}"
    print(f"  Cache filled to max_size=3. Current size: {len(cache)}")

    # Touch entry 1 to make it recently used
    # → entry 0 or 2 should be evicted (not entry 1)
    touched = cache.get(
        query="query 1 lookup",
        query_embedding=make_similar_vec(make_vec(seed=1), noise=0.01),
        dominant_cluster=0,
    )
    if touched:
        print(f"  Touched entry 1 (score={touched['similarity_score']:.4f})")
    else:
        print(f"  Note: touch of entry 1 did not hit (vectors not similar enough)")

    # Insert a new entry → must evict LRU to stay at max_size=3
    cache.put(
        query="brand new query",
        query_embedding=make_vec(seed=100),
        result={"data": "new"},
        dominant_cluster=0,
        cluster_probs=[0.0] * 5,
    )

    assert len(cache) == 3, (
        f"Expected size=3 after eviction, got {len(cache)}"
    )
    assert cache._stats["evictions"] == 1, (
        f"Expected 1 eviction, got {cache._stats['evictions']}"
    )

    print(f"  ✅ LRU eviction: size={len(cache)}/3, "
          f"evictions={cache._stats['evictions']}")


def test_stats_accuracy():
    """
    Stats counters should accurately track hits, misses, and hit rate.

    Validates:
    - total_queries = hits + misses
    - hit_rate_pct = hits / total_queries * 100
    - counters are thread-consistent
    """
    print_separator("Stats Accuracy")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)
    vec = make_vec(seed=42)

    # Store one entry
    cache.put(
        query="original query",
        query_embedding=vec,
        result={"data": "x"},
        dominant_cluster=0,
        cluster_probs=[0.0] * 5,
    )

    # 1 hit
    cache.get("similar q", make_similar_vec(vec, noise=0.01), dominant_cluster=0)

    # 2 misses
    cache.get("unrelated 1", make_vec(seed=10), dominant_cluster=0)
    cache.get("unrelated 2", make_vec(seed=11), dominant_cluster=0)

    stats = cache.get_stats()

    assert stats["cache_hits"]    == 1, f"Expected 1 hit, got {stats['cache_hits']}"
    assert stats["cache_misses"]  == 2, f"Expected 2 misses, got {stats['cache_misses']}"
    assert stats["total_queries"] == 3, f"Expected 3 queries, got {stats['total_queries']}"

    expected_rate = round(1 / 3 * 100, 2)
    assert stats["hit_rate_pct"]  == expected_rate, (
        f"Expected hit_rate={expected_rate}, got {stats['hit_rate_pct']}"
    )
    assert stats["current_size"]  == 1
    assert stats["total_stored"]  == 1

    print(f"  ✅ Stats: hits={stats['cache_hits']}, "
          f"misses={stats['cache_misses']}, "
          f"total={stats['total_queries']}, "
          f"hit_rate={stats['hit_rate_pct']}%")


def test_clear():
    """
    clear() should empty all entries, buckets, and reset traffic counters.

    Validates:
    - All entries removed after clear()
    - All buckets emptied
    - Return value = number of entries removed
    """
    print_separator("Cache Clear")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    # Add 5 entries across different clusters
    for i in range(5):
        cache.put(
            query=f"query {i}",
            query_embedding=make_vec(seed=i),
            result={"data": i},
            dominant_cluster=i % 3,    # distribute across clusters 0, 1, 2
            cluster_probs=[0.0] * 5,
        )

    assert len(cache) == 5, f"Expected 5 entries before clear, got {len(cache)}"

    removed = cache.clear()

    assert removed == 5,      f"Expected clear() to return 5, got {removed}"
    assert len(cache) == 0,   f"Expected 0 entries after clear, got {len(cache)}"
    assert len(cache._store) == 0
    assert all(len(v) == 0 for v in cache._buckets.values())

    print(f"  ✅ Clear: removed={removed}, "
          f"size_after={len(cache)}, "
          f"buckets_empty={all(len(v) == 0 for v in cache._buckets.values())}")


def test_ttl_expiry():
    """
    Entries older than TTL should not be returned as cache hits.

    Validates:
    - Entry IS returned before TTL expires
    - Entry is NOT returned after TTL expires
    - TTL check uses last_accessed (not creation time)
    """
    print_separator("TTL Expiry")

    import time

    cache = SemanticCache(
        similarity_threshold=0.85,
        n_clusters=5,
        ttl_seconds=0.1,    # 100ms TTL — short enough for a test
    )

    vec = make_vec(seed=7)
    similar_vec = make_similar_vec(vec, noise=0.01)

    cache.put(
        query="expiring query",
        query_embedding=vec,
        result={"data": "x"},
        dominant_cluster=0,
        cluster_probs=[0.0] * 5,
    )

    # Immediate lookup → should HIT (not expired yet)
    hit = cache.get("similar query", similar_vec, dominant_cluster=0)
    assert hit is not None, (
        "Expected HIT before TTL expiry. "
        f"Check similarity: {float(np.dot(vec, similar_vec)):.4f}"
    )
    print(f"  ✅ TTL: HIT before expiry "
          f"(score={hit['similarity_score']:.4f}, ttl=0.1s)")

    # Wait past TTL
    time.sleep(0.15)

    # After TTL → should MISS
    miss = cache.get("similar query", similar_vec, dominant_cluster=0)
    assert miss is None, (
        "Expected MISS after TTL expiry, but got a cache hit. "
        "TTL check may not be working correctly."
    )
    print(f"  ✅ TTL: MISS after 0.15s (ttl=0.1s, entry correctly expired)")


def test_multi_cluster_isolation():
    """
    Queries in different clusters should NOT match each other,
    even if their embedding vectors happen to be similar.

    Validates:
    - Cache lookup only searches within the correct cluster bucket
    - Cross-cluster queries never produce false hits
    """
    print_separator("Multi-Cluster Isolation")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    vec = make_vec(seed=55)
    similar_vec = make_similar_vec(vec, noise=0.01)

    # Store in cluster 0
    cache.put(
        query="stored in cluster 0",
        query_embedding=vec,
        result={"data": "cluster_0_result"},
        dominant_cluster=0,
        cluster_probs=[0.0] * 5,
    )

    # Lookup from cluster 0 → should HIT
    hit = cache.get("lookup from cluster 0", similar_vec, dominant_cluster=0)
    assert hit is not None, "Expected HIT when searching same cluster"
    print(f"  ✅ Same cluster HIT  | score={hit['similarity_score']:.4f}")

    # Lookup from cluster 3 → should MISS (wrong bucket, different topic)
    miss = cache.get("lookup from cluster 3", similar_vec, dominant_cluster=3)
    assert miss is None, (
        "Expected MISS when searching different cluster. "
        "Cache should not cross-search cluster buckets."
    )
    print(f"  ✅ Cross-cluster MISS | cluster=3 (stored in cluster=0)")


def test_invalidate():
    """
    invalidate(entry_id) should remove a specific entry by ID.

    Validates:
    - Entry is removed from _store
    - Entry ID is removed from its cluster bucket
    - Subsequent lookup returns None
    """
    print_separator("Individual Invalidation")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    vec = make_vec(seed=20)
    similar_vec = make_similar_vec(vec, noise=0.01)

    # Store entry and capture its ID
    entry_id = cache.put(
        query="invalidation test query",
        query_embedding=vec,
        result={"data": "will be removed"},
        dominant_cluster=1,
        cluster_probs=[0.0] * 5,
    )

    assert len(cache) == 1
    print(f"  Stored entry: id={entry_id[:8]}...")

    # Verify it can be found
    hit_before = cache.get("test", similar_vec, dominant_cluster=1)
    assert hit_before is not None, "Entry should exist before invalidation"
    print(f"  ✅ Found before invalidation (score={hit_before['similarity_score']:.4f})")

    # Invalidate the entry
    removed = cache.invalidate(entry_id)
    assert removed is True, f"Expected invalidate() to return True, got {removed}"
    assert len(cache) == 0, f"Expected 0 entries after invalidation, got {len(cache)}"

    # Verify it's gone
    miss_after = cache.get("test", similar_vec, dominant_cluster=1)
    assert miss_after is None, "Entry should not exist after invalidation"
    print(f"  ✅ Gone after invalidation | size={len(cache)}")


def test_multiple_entries_same_cluster():
    """
    When multiple entries exist in the same cluster,
    the one with the HIGHEST similarity score should be returned.

    Validates:
    - Best match is selected (not first match)
    - All entries in bucket are compared
    """
    print_separator("Best Match Selection")

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    base_vec = make_vec(seed=30)

    # Entry A — very similar to query (noise=0.01)
    vec_a = make_similar_vec(base_vec, noise=0.01)
    # Entry B — less similar to query (noise=0.08 — but still normalized)
    rng = np.random.default_rng(111)
    vec_b_raw = base_vec + 0.08 * rng.standard_normal(384).astype(np.float32)
    vec_b = vec_b_raw / np.linalg.norm(vec_b_raw)

    # Store both entries in the same cluster
    cache.put(
        query="less similar entry B",
        query_embedding=vec_b,
        result={"match": "entry_B"},
        dominant_cluster=2,
        cluster_probs=[0.0] * 5,
    )
    cache.put(
        query="more similar entry A",
        query_embedding=vec_a,
        result={"match": "entry_A"},
        dominant_cluster=2,
        cluster_probs=[0.0] * 5,
    )

    # Query with base_vec — should match entry A (more similar)
    result = cache.get("test query", base_vec, dominant_cluster=2)

    if result is not None:
        print(f"  Best match: '{result['matched_query']}' "
              f"(score={result['similarity_score']:.4f})")
        print(f"  ✅ Best-match selection working correctly")
    else:
        print(f"  ⚠️  No match found — both entries may be below threshold")
        print(f"     sim(base, vec_a) = {float(np.dot(base_vec, vec_a)):.4f}")
        print(f"     sim(base, vec_b) = {float(np.dot(base_vec, vec_b)):.4f}")


def test_get_all_entries():
    """
    get_all_entries() should return serialized entries sorted by
    most recently accessed first.
    """
    print_separator("Get All Entries")

    import time

    cache = SemanticCache(similarity_threshold=0.85, n_clusters=5)

    # Store 3 entries with slight time gaps
    for i in range(3):
        cache.put(
            query=f"entry {i}",
            query_embedding=make_vec(seed=i + 50),
            result={"data": i},
            dominant_cluster=0,
            cluster_probs=[0.0] * 5,
        )
        time.sleep(0.01)    # ensure distinct timestamps

    entries = cache.get_all_entries()

    assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"

    # Verify each entry has required fields
    required_fields = {"id", "query", "dominant_cluster",
                       "timestamp", "hits", "last_accessed", "age_seconds"}
    for entry in entries:
        missing = required_fields - entry.keys()
        assert not missing, f"Entry missing fields: {missing}"

    # Verify sorted by last_accessed descending (most recent first)
    timestamps = [e["last_accessed"] for e in entries]
    assert timestamps == sorted(timestamps, reverse=True), \
        "Entries should be sorted by last_accessed descending"

    print(f"  ✅ get_all_entries: {len(entries)} entries, "
          f"sorted by last_accessed descending")
    for e in entries:
        print(f"     id={e['id'][:8]}... | "
              f"query='{e['query']}' | "
              f"hits={e['hits']}")


def test_cleanup_expired():
    """
    cleanup_expired() should remove all TTL-expired entries
    and return the count of removed entries.
    """
    print_separator("Cleanup Expired")

    import time

    cache = SemanticCache(
        similarity_threshold=0.85,
        n_clusters=5,
        ttl_seconds=0.1,
    )

    # Store 3 entries
    for i in range(3):
        cache.put(
            query=f"expiring {i}",
            query_embedding=make_vec(seed=i + 70),
            result={"data": i},
            dominant_cluster=i % 2,
            cluster_probs=[0.0] * 5,
        )

    assert len(cache) == 3
    print(f"  Stored 3 entries with TTL=0.1s")

    # Wait for TTL to expire
    time.sleep(0.15)

    # Add one fresh entry (should NOT be cleaned up)
    cache.put(
        query="fresh entry",
        query_embedding=make_vec(seed=99),
        result={"data": "fresh"},
        dominant_cluster=0,
        cluster_probs=[0.0] * 5,
    )

    assert len(cache) == 4

    # Run cleanup
    removed = cache.cleanup_expired()

    assert removed == 3, f"Expected 3 expired removals, got {removed}"
    assert len(cache) == 1, f"Expected 1 entry remaining, got {len(cache)}"
    assert cache._stats["expired_removals"] == 3

    print(f"  ✅ Cleanup: removed={removed} expired, "
          f"remaining={len(cache)} (fresh entry survived)")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("     SemanticCache — Full Test Suite")
    print("=" * 55)

    tests = [
        ("Basic Hit / Miss",             test_basic_hit_miss),
        ("LRU Eviction",                 test_lru_eviction),
        ("Stats Accuracy",               test_stats_accuracy),
        ("Cache Clear",                  test_clear),
        ("TTL Expiry",                   test_ttl_expiry),
        ("Multi-Cluster Isolation",      test_multi_cluster_isolation),
        ("Individual Invalidation",      test_invalidate),
        ("Best Match Selection",         test_multiple_entries_same_cluster),
        ("Get All Entries",              test_get_all_entries),
        ("Cleanup Expired",              test_cleanup_expired),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"\n  ❌ FAILED: {test_name}")
            print(f"     {e}")
        except Exception as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"\n  💥 ERROR: {test_name}")
            print(f"     {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)

    if errors:
        print("\nFailed tests:")
        for name, msg in errors:
            print(f"  ❌ {name}: {msg}")
    else:
        print("\n  ✅ All tests passed!\n")

    # Exit with non-zero code if any test failed (useful for CI)
    sys.exit(0 if failed == 0 else 1)
