# 🔍 Semantic Search System

> A production-quality semantic search engine over the [20 Newsgroups dataset](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) featuring dense vector search, fuzzy clustering, and a custom semantic cache — built from scratch without caching libraries.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-1.8-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Component Deep Dives](#-component-deep-dives)
  - [Embeddings](#1-embeddings)
  - [Vector Store](#2-vector-store-faiss)
  - [Fuzzy Clustering](#3-fuzzy-clustering-gmm)
  - [Semantic Cache](#4-semantic-cache)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Pipeline](#-running-the-pipeline)
- [API Reference](#-api-reference)
- [API Usage Examples](#-api-usage-examples)
- [Docker Setup](#-docker-setup)
- [Design Decisions](#-design-decisions)
- [Performance Benchmarks](#-performance-benchmarks)

---

## 🎯 Project Overview

This system answers natural language queries over 18,000+ newsgroup documents using semantic understanding — not keyword matching.

**What makes it different from a keyword search:**

| Keyword Search | Semantic Search (This System) |
|---|---|
| Matches exact words | Matches meaning and intent |
| `"NASA rocket"` misses `"space shuttle launch"` | Correctly identifies both as related |
| No understanding of context | Understands `"motor issues" ≈ "engine problems"` |
| Fast but brittle | Fast AND semantically aware |

**Key capabilities:**

- **Semantic search** — find documents by meaning using dense vector embeddings
- **Fuzzy clustering** — every document belongs to *multiple* topic clusters with probabilities (not hard assignment)
- **Semantic cache** — detects similar past queries to avoid redundant computation, built entirely from Python data structures
- **Production-ready API** — FastAPI with Pydantic validation, Swagger UI, structured logging

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                            │
│                    POST /query  {"query": "..."}                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                           │
│                                                                  │
│   ┌─────────────┐     ┌──────────────┐     ┌────────────────┐  │
│   │  Pydantic   │────►│  Route       │────►│  AppState      │  │
│   │  Validation │     │  Handler     │     │  (shared)      │  │
│   └─────────────┘     └──────┬───────┘     └────────────────┘  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │         PIPELINE EXECUTION        │
              │                                   │
              │  1. Embedder.encode(query)         │  ~50ms
              │          │                        │
              │  2. Clusterer.predict_single()     │  ~2ms
              │          │                        │
              │  3. SemanticCache.get()  ──────────┼──► HIT → return
              │          │ MISS                   │        cached
              │  4. FAISSVectorStore.search()      │  ~2ms  result
              │          │                        │
              │  5. SemanticCache.put()            │  ~1ms
              │          │                        │
              │  6. Return QueryResponse           │
              └───────────────────────────────────┘

OFFLINE PIPELINE (run once before serving):

  Raw Data          Cleaned          Embeddings       FAISS Index
  (18,846 docs) ──► (16,432 docs) ──► (N × 384      ──► (binary
  download_data     preprocess        float32)           index)
                                   build_embeddings  build_index

                                   GMM Model
                                   (PCA 384→50
                                    + 20 Gaussians)
                                   build_clusters
```

---

## 🔬 Component Deep Dives

### 1. Embeddings

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

This model converts variable-length text into fixed-size 384-dimensional vectors where **semantic similarity = geometric proximity**.
```
"NASA launches satellite"    → [0.21, -0.83,  0.44, ... ]  ─┐
"space agency rocket launch" → [0.19, -0.81,  0.46, ... ]  ─┴─ cosine sim = 0.96

"best pasta carbonara"       → [-0.54, 0.12, -0.33, ... ]  ← far away in space
```

All vectors are **L2-normalized** (unit sphere), making cosine similarity equivalent to a simple dot product — critical for FAISS performance.

**Why `all-MiniLM-L6-v2`:**

| Property | Value |
|---|---|
| Dimensions | 384 |
| Model size | ~22 MB |
| Encoding speed | ~14,000 sentences/sec (CPU) |
| Quality | Top-tier on STS benchmarks |

---

### 2. Vector Store (FAISS)

FAISS (Facebook AI Similarity Search) organizes all 16,432 document vectors into an index structure that supports millisecond-speed nearest-neighbor lookup.

**Index type:** `IndexFlatIP` (exact inner product search)
```
Query vector: q = [0.21, -0.83, 0.44, ...]

FAISS computes: score_i = dot(q, doc_i)  for all i
              → returns top-K indices with highest scores

At 16K vectors: ~2ms per query
At 1M vectors: consider switching to IndexIVFFlat (~5ms, ~95% accuracy)
```

**Why not approximate (HNSW/IVF)?**
At 16K documents, exact search takes ~2ms — there's no speed/accuracy tradeoff worth making. Exact results are always better for a production demo.

---

### 3. Fuzzy Clustering (GMM)

Standard clustering (K-Means) forces every document into exactly one cluster. This is wrong for text — a post about *NASA military satellite contracts* belongs to **both** sci.space and politics.

**Gaussian Mixture Model** assigns soft probabilities across all clusters:
```
K-Means (hard):
  "NASA military satellite contracts" → Cluster 4

GMM (fuzzy / soft):
  "NASA military satellite contracts" → {
      cluster_4  (sci.space):        0.52,
      cluster_11 (talk.politics):    0.31,
      cluster_7  (sci.electronics):  0.12,
      cluster_2  (misc.forsale):     0.05
  }
```

**Pipeline:**
```
Raw embeddings (N × 384)
    │
    ▼  PCA reduction
Reduced embeddings (N × 50)   ← retains ~85% of variance
    │
    ▼  Expectation-Maximization (EM)
GMM: 20 Gaussian components
    │
    ▼
Soft assignment matrix (N × 20)   ← rows sum to 1.0
```

**Why PCA before GMM?**
The "curse of dimensionality" makes distance metrics unreliable in 384D space. PCA reduces to 50D while preserving semantic structure — GMM converges faster and produces tighter clusters.

**Cluster count selection (BIC):**
```
n_clusters=10  →  BIC = 1,842,301  (underfitting)
n_clusters=15  →  BIC = 1,798,442
n_clusters=20  →  BIC = 1,761,883  ← lowest = best fit ✅
n_clusters=25  →  BIC = 1,779,204  (overfitting)
```

---

### 4. Semantic Cache

The cache detects **semantically equivalent queries** to skip redundant computation. Built entirely from Python data structures — no Redis, no Memcached, no caching libraries.

**Architecture:**
```
SemanticCache
├── _store: dict[entry_id → CacheEntry]     ← primary store, O(1) access
└── _buckets: dict[cluster_id → [entry_ids]] ← partitioned by GMM cluster

On GET(query_embedding, cluster=7):
  1. Look up _buckets[7]                    ← only ~50 entries (not all 1000)
  2. Dot product with each entry embedding  ← 50 operations, not 1000
  3. If best_score ≥ 0.85 → HIT            ← return cached result
  4. Else → MISS                            ← run full search pipeline

Complexity:  O(C/n) average  vs  O(C) naive
             (C=cache size, n=clusters=20)
             At C=1000: 50 comparisons vs 1000 — 20× faster
```

**Cache entry lifecycle:**
```
New query arrives
     │
     ▼
encode + cluster → check bucket → similarity check
                                        │
                              ┌─────────┴──────────┐
                           score ≥ 0.85          score < 0.85
                              │                      │
                           CACHE HIT             CACHE MISS
                              │                      │
                        touch() entry          FAISS search
                        (update LRU             → store entry
                         timestamp)             in bucket
```

**Eviction:** LRU (Least Recently Used)
When cache reaches `max_size=1000`, the entry with the oldest `last_accessed` timestamp is removed.

**TTL:** 1 hour
Entries expire after 1 hour of inactivity, ensuring search results reflect new indexed documents.

**Similarity threshold:** `0.85`
```
score > 0.92  →  too strict: paraphrases missed
score = 0.85  →  optimal: semantically equivalent queries match ✅
score < 0.80  →  too loose: unrelated queries incorrectly match
```

---

## 📁 Project Structure
```
semantic-search-system/
│
├── data/
│   ├── raw/                       # Original downloaded data (immutable)
│   └── processed/                 # Cleaned documents + stats
│
├── embeddings/
│   └── embedder.py                # SentenceTransformer wrapper (Adapter pattern)
│
├── vectorstore/
│   └── faiss_store.py             # FAISS abstraction (Repository pattern)
│
├── clustering/
│   └── fuzzy_cluster.py           # GMM fuzzy clusterer with PCA
│
├── cache/
│   └── semantic_cache.py          # Custom semantic cache (from scratch)
│
├── api/
│   ├── app.py                     # FastAPI routes + lifespan
│   ├── schemas.py                 # Pydantic request/response models
│   └── state.py                   # Shared app state container
│
├── utils/
│   ├── text_cleaner.py            # 7-step text preprocessing pipeline
│   └── logger.py                  # Structured logging
│
├── scripts/                       # One-time offline pipeline scripts
│   ├── download_data.py
│   ├── preprocess.py
│   ├── build_embeddings.py
│   ├── build_index.py
│   └── build_clusters.py
│
├── models/                        # Saved artifacts (gitignored if large)
│   ├── faiss_index.bin
│   ├── gmm_model.pkl
│   ├── embeddings.npy
│   └── *.json
│
├── tests/
│   ├── test_cache.py
│   ├── test_embedder.py
│   └── test_clustering.py
│
├── main.py                        # Uvicorn entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- pip
- ~2GB disk space (model + embeddings + index)
- ~1GB RAM minimum (4GB recommended)

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/semantic-search-system.git
cd semantic-search-system

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables
cp .env.example .env
```

---

## 🚀 Running the Pipeline

Run these scripts **once** to build all model artifacts:
```bash
# Step 1: Download the 20 Newsgroups dataset (~18,846 documents)
python scripts/download_data.py

# Step 2: Clean and preprocess text
python scripts/preprocess.py

# Step 3: Generate embeddings (~2 min on CPU)
python scripts/build_embeddings.py

# Step 4: Build FAISS vector index
python scripts/build_index.py

# Step 5: Fit GMM fuzzy clustering model (~5 min on CPU)
python scripts/build_clusters.py

# Step 6: Launch the API
python main.py
```

**Quick test mode** (faster, uses 500 documents):
```bash
python scripts/build_embeddings.py --limit 500
python scripts/build_index.py
python scripts/build_clusters.py
python main.py
```

**Expected build times (CPU):**

| Script | Time |
|---|---|
| download_data.py | ~30 sec |
| preprocess.py | ~10 sec |
| build_embeddings.py | ~2 min |
| build_index.py | ~5 sec |
| build_clusters.py | ~5 min |
| **Total** | **~8 min** |

---

## 📡 API Reference

### `POST /query`

Perform a semantic search. Returns cached result if a similar query was seen before.

**Request:**
```json
{
  "query": "NASA space shuttle mission",
  "top_k": 5,
  "similarity_threshold": 0.85
}
```

**Response:**
```json
{
  "query": "NASA space shuttle mission",
  "processing_time_ms": 84.23,
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "cluster_info": {
    "dominant_cluster": 7,
    "dominant_prob": 0.6341,
    "top_clusters": [[7, 0.6341], [2, 0.2104], [14, 0.0891]]
  },
  "results": [
    {
      "rank": 1,
      "doc_id": 4821,
      "score": 0.7841,
      "snippet": "The Space Shuttle Discovery completed its final mission...",
      "category": "sci.space",
      "word_count": 312
    }
  ],
  "total_results": 5,
  "timestamp": "2024-01-15T12:00:00+00:00"
}
```

**Cache hit response** (second similar query):
```json
{
  "query": "space exploration rocket launch astronauts",
  "processing_time_ms": 4.87,
  "cache_hit": true,
  "matched_query": "NASA space shuttle mission",
  "similarity_score": 0.9138,
  ...
}
```

---

### `GET /cache/stats`

Returns live cache performance metrics.

**Response:**
```json
{
  "total_queries": 24,
  "cache_hits": 9,
  "cache_misses": 15,
  "hit_rate_pct": 37.5,
  "current_size": 15,
  "max_size": 1000,
  "capacity_pct": 1.5,
  "evictions": 0,
  "similarity_threshold": 0.85,
  "ttl_seconds": 3600,
  "avg_hits_per_entry": 0.6,
  "bucket_distribution": {
    "3": 4, "7": 6, "11": 3, "14": 2
  },
  "recent_entries": [...]
}
```

---

### `DELETE /cache`

Clears all cached entries. Use after reindexing documents.

**Response:**
```json
{
  "message": "Cache cleared successfully. 15 entries removed.",
  "entries_removed": 15,
  "timestamp": "2024-01-15T12:05:00+00:00"
}
```

---

### `GET /health`

Returns component health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "embedder": "ready",
    "vector_store": "ready",
    "clusterer": "ready",
    "cache": "ready"
  },
  "index_size": 16432,
  "cache_size": 15,
  "timestamp": "2024-01-15T12:00:00+00:00"
}
```

---

## 💡 API Usage Examples
```python
import requests

BASE_URL = "http://localhost:8000"

def search(query: str, top_k: int = 5) -> dict:
    response = requests.post(f"{BASE_URL}/query", json={
        "query": query,
        "top_k": top_k,
    })
    response.raise_for_status()
    return response.json()

# Example 1: Fresh search
result = search("electric vehicles and battery technology")
print(f"Cache hit: {result['cache_hit']}")
print(f"Time: {result['processing_time_ms']}ms")
print(f"Cluster: {result['cluster_info']['dominant_cluster']}")
for doc in result['results']:
    print(f"  [{doc['rank']}] {doc['score']:.4f} | {doc['category']}")
    print(f"       {doc['snippet'][:100]}...")

# Example 2: Semantically equivalent query → instant cache hit
result2 = search("EV batteries lithium charging infrastructure")
print(f"\nCache hit: {result2['cache_hit']}")         # True
print(f"Time: {result2['processing_time_ms']}ms")     # ~5ms
print(f"Matched: '{result2['matched_query']}'")
print(f"Similarity: {result2['similarity_score']:.4f}")
```

---

## 🐳 Docker Setup
```bash
# Build and run with Docker Compose (recommended)
docker-compose up --build

# Or build and run manually
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search

# Access the API
curl http://localhost:8000/health
```

---

## 🧠 Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Best speed/quality tradeoff at 384 dims |
| Vector index | FAISS IndexFlatIP | Exact results; 16K docs fits in memory |
| Clustering | GMM (not K-Means) | Native soft/fuzzy probabilities |
| PCA before GMM | 384→50 dims | Stabilizes EM, retains 85% variance |
| Cache partitioning | By GMM cluster | O(C/n) lookup vs O(C) naive |
| Cache eviction | LRU | Simple, industry standard, effective |
| API framework | FastAPI | Async, auto-docs, Pydantic validation |
| Workers | 1 Uvicorn worker | In-memory state; multi-worker = N×RAM |

---

## 📊 Performance Benchmarks

Measured on MacBook Pro M1, CPU only:

| Operation | Latency |
|---|---|
| Cache HIT (full pipeline skip) | ~5ms |
| Cache MISS (full search) | ~80ms |
| Embedding encode (single query) | ~50ms |
| FAISS search (16K vectors, top-5) | ~2ms |
| GMM cluster predict (single) | ~1ms |
| Cache lookup (1000 entries, 20 clusters) | ~0.5ms |

**Cache speedup:** ~16× faster on hit vs fresh search

---

## 🧪 Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Individual modules
python tests/test_cache.py
python tests/test_embedder.py
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.