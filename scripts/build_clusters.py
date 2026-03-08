# scripts/build_clusters.py
"""
Offline pipeline: load embeddings, fit GMM, save model + assignments.

PIPELINE ORDER:
  1. download_data.py       ✅
  2. preprocess.py          ✅
  3. build_embeddings.py    ✅
  4. build_index.py         ✅
  5. build_clusters.py      ← we are here
  6. [API is now ready to launch]

WHAT THIS SCRIPT PRODUCES:
  models/gmm_model.pkl             — fitted PCA + GMM (loaded by API)
  models/cluster_assignments.json  — soft assignments for all documents
  models/cluster_stats.json        — metrics, boundary docs, evaluation
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clustering.fuzzy_cluster import FuzzyClusterer
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
EMBEDDINGS_FILE          = os.path.join(MODELS_DIR, "embeddings.npy")
DOC_TEXTS_FILE           = os.path.join(MODELS_DIR, "doc_texts.json")
DOC_META_FILE            = os.path.join(MODELS_DIR, "doc_metadata.json")

GMM_MODEL_FILE           = os.path.join(MODELS_DIR, "gmm_model.pkl")
ASSIGNMENTS_FILE         = os.path.join(MODELS_DIR, "cluster_assignments.json")
CLUSTER_STATS_FILE       = os.path.join(MODELS_DIR, "cluster_stats.json")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_CLUSTERS     = 20    # matches 20 newsgroup categories
N_PCA_DIMS     = 50    # PCA target dimensions
SOFT_THRESHOLD = 0.10  # minimum probability to count as a membership
RANDOM_STATE   = 42


def build_clusters() -> None:
    """
    Full cluster-building pipeline with BIC-guided model selection.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load embeddings ---
    logger.info(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    embedding_matrix = np.load(EMBEDDINGS_FILE)
    logger.info(f"Embedding matrix: {embedding_matrix.shape}")

    # Load texts for boundary document analysis
    with open(DOC_TEXTS_FILE, "r", encoding="utf-8") as f:
        doc_texts = json.load(f)

    with open(DOC_META_FILE, "r", encoding="utf-8") as f:
        doc_meta = json.load(f)

    # --- Optional: BIC-based cluster count selection ---
    # WHY BIC: Bayesian Information Criterion penalizes model complexity.
    # The n_clusters with lowest BIC is optimal.
    # We run a quick search over a small range to verify n=20 is sensible.
    logger.info("Running BIC model selection (n_clusters: 10, 15, 20, 25)...")
    bic_results = _bic_search(
        embedding_matrix,
        cluster_range=[10, 15, 20, 25],
        n_pca_dims=N_PCA_DIMS,
    )
    best_n = min(bic_results, key=bic_results.get)
    logger.info(f"BIC scores: {bic_results}")
    logger.info(f"BIC-optimal n_clusters: {best_n} (using {N_CLUSTERS})")

    # --- Fit full model ---
    logger.info(f"Fitting final GMM with n_clusters={N_CLUSTERS}...")
    clusterer = FuzzyClusterer(
        n_clusters=N_CLUSTERS,
        n_components=N_PCA_DIMS,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        max_iter=200,
    )
    clusterer.fit(embedding_matrix)

    # --- Evaluate model ---
    metrics = clusterer.score_model(embedding_matrix)
    logger.info(f"Model metrics: {metrics}")

    # --- Compute assignments for all documents ---
    assignments = clusterer.get_cluster_assignments(
        embedding_matrix,
        soft_threshold=SOFT_THRESHOLD,
    )

    # --- Find boundary documents (multi-topic) ---
    boundary_docs = clusterer.find_boundary_documents(
        assignments,
        doc_texts,
        n_examples=10,
    )

    logger.info("\n=== BOUNDARY DOCUMENTS (most ambiguous) ===")
    for bd in boundary_docs[:3]:
        logger.info(
            f"  Doc {bd['doc_index']} | "
            f"entropy={bd['entropy']} | "
            f"memberships={bd['memberships']}\n"
            f"  Snippet: {bd['snippet'][:120]}..."
        )

    # --- Build cluster summary stats ---
    cluster_stats = _build_cluster_stats(
        assignments,
        doc_meta,
        n_clusters=N_CLUSTERS,
    )

    # --- Save everything ---
    clusterer.save(GMM_MODEL_FILE)

    with open(ASSIGNMENTS_FILE, "w") as f:
        json.dump(assignments, f, indent=2)
    logger.info(f"Assignments saved → {ASSIGNMENTS_FILE}")

    full_stats = {
        "model_metrics": metrics,
        "cluster_summary": cluster_stats,
        "boundary_documents": boundary_docs,
        "hyperparameters": {
            "n_clusters": N_CLUSTERS,
            "n_pca_dims": N_PCA_DIMS,
            "soft_threshold": SOFT_THRESHOLD,
            "covariance_type": "diag",
        },
        "bic_search": bic_results,
    }

    with open(CLUSTER_STATS_FILE, "w") as f:
        json.dump(full_stats, f, indent=2)
    logger.info(f"Cluster stats saved → {CLUSTER_STATS_FILE}")

    logger.info("=" * 55)
    logger.info("  Clustering pipeline complete ✅")
    logger.info(f"  Model:        {GMM_MODEL_FILE}")
    logger.info(f"  Assignments:  {ASSIGNMENTS_FILE}")
    logger.info(f"  Stats:        {CLUSTER_STATS_FILE}")
    logger.info("=" * 55)


def _bic_search(
    embedding_matrix: np.ndarray,
    cluster_range: list[int],
    n_pca_dims: int,
    subsample: int = 3000,
) -> dict[int, float]:
    """
    Run GMM with different n_clusters values and return BIC scores.

    WHY SUBSAMPLE:
        Full BIC search on 16K docs × 4 cluster counts × 3 inits
        would take ~20 minutes. We subsample 3000 docs for BIC
        selection — enough for a reliable estimate, fast to compute.

    Args:
        embedding_matrix: full embedding matrix
        cluster_range:    list of n_clusters values to try
        n_pca_dims:       PCA reduction dimensions
        subsample:        use this many randomly sampled docs for BIC

    Returns:
        Dict mapping n_clusters → BIC score
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize

    # Subsample for speed
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(embedding_matrix), size=min(subsample, len(embedding_matrix)),
                     replace=False)
    X_sub = normalize(embedding_matrix[idx], norm="l2")

    # PCA on subsample
    pca = PCA(n_components=n_pca_dims, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X_sub)

    bic_scores = {}
    for n in cluster_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="diag",
            max_iter=100,
            random_state=RANDOM_STATE,
            n_init=1,   # fast — just for BIC comparison
        )
        gmm.fit(X_reduced)
        bic_scores[n] = round(float(gmm.bic(X_reduced)), 2)
        logger.info(f"  n_clusters={n:3d} → BIC={bic_scores[n]:,.2f}")

    return bic_scores


def _build_cluster_stats(
    assignments: list[dict],
    doc_meta: list[dict],
    n_clusters: int,
) -> list[dict]:
    """
    Build a human-readable summary of each cluster.

    For each cluster:
    - How many documents are assigned (dominant)?
    - What newsgroup categories dominate it?
    - What is the average dominance probability?

    This tells us whether the GMM discovered meaningful topic groupings
    or just random partitions.
    """
    from collections import Counter

    # Build {cluster_id: [doc_indices]}
    cluster_to_docs: dict[int, list[int]] = {k: [] for k in range(n_clusters)}
    for a in assignments:
        cluster_to_docs[a["dominant"]].append(a["doc_index"])

    stats = []
    for cluster_id in range(n_clusters):
        doc_indices = cluster_to_docs[cluster_id]

        if not doc_indices:
            stats.append({"cluster_id": cluster_id, "size": 0})
            continue

        # Get categories of documents in this cluster
        categories = [
            doc_meta[i]["category_name"]
            for i in doc_indices
            if i < len(doc_meta)
        ]
        category_counts = Counter(categories)
        top_categories = category_counts.most_common(3)

        # Average dominant probability (higher = tighter cluster)
        avg_prob = float(np.mean([
            assignments[i]["dominant_prob"]
            for i in doc_indices
            if i < len(assignments)
        ]))

        stats.append({
            "cluster_id": cluster_id,
            "size": len(doc_indices),
            "avg_dominant_prob": round(avg_prob, 4),
            "top_categories": [
                {"category": cat, "count": cnt}
                for cat, cnt in top_categories
            ],
        })

    return stats


if __name__ == "__main__":
    build_clusters()