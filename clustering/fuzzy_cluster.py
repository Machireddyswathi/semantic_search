# clustering/fuzzy_cluster.py
"""
Fuzzy clustering using Gaussian Mixture Models (GMM).

WHY GMM OVER OTHER METHODS:
┌─────────────────────┬─────────────────────────────────────────────────┐
│ Method              │ Why NOT chosen                                  │
├─────────────────────┼─────────────────────────────────────────────────┤
│ K-Means             │ Hard assignment only — no probabilities         │
│ DBSCAN              │ Labels noise points, no soft membership         │
│ Fuzzy C-Means       │ Less stable on high-dim data, manual impl.      │
│ LDA (topic models)  │ Works on raw text, not embeddings               │
│ GMM ✅              │ Native soft probabilities, sklearn integrated,  │
│                     │ works on embedding space, mathematically sound  │
└─────────────────────┴─────────────────────────────────────────────────┘

DIMENSIONALITY NOTE:
Raw embeddings are 384-dim. GMM struggles with high-dim data because:
- Covariance matrices become huge (384×384 per cluster)
- The "curse of dimensionality" makes distances uniform
- EM convergence is slow and unstable

SOLUTION: Reduce to 50 dimensions with PCA first.
PCA preserves ~85% of variance while making GMM tractable.
This is standard practice in production clustering pipelines.
"""

import os
import json
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.logger import get_logger

logger = get_logger(__name__)


class FuzzyClusterer:
    """
    Fits a Gaussian Mixture Model on document embeddings to produce
    soft (fuzzy) cluster assignments.

    Pipeline:
        raw embeddings (N, 384)
            → PCA reduction (N, n_components)
                → GMM fitting
                    → soft probabilities (N, n_clusters)

    Attributes:
        n_clusters:    number of Gaussian components
        n_components:  PCA dimensions before GMM
        pca:           fitted sklearn PCA object
        gmm:           fitted sklearn GaussianMixture object
        is_fitted:     whether the model has been trained
    """

    def __init__(
        self,
        n_clusters: int = 20,
        n_components: int = 50,
        covariance_type: str = "diag",
        random_state: int = 42,
        max_iter: int = 200,
    ):
        """
        Args:
            n_clusters:
                Number of Gaussian components (soft clusters).
                WHY 20: The dataset has 20 newsgroup categories.
                Setting n_clusters = 20 aligns with the known structure
                without forcing the model — it discovers natural groupings.
                In production, use BIC/AIC selection (shown in build script).

            n_components:
                PCA dimensions before GMM.
                WHY 50: Retains ~85% of variance for 384-dim embeddings
                while making GMM stable and fast to converge.

            covariance_type:
                GMM covariance matrix structure.
                WHY "diag" (diagonal covariance):
                - "full"  = unrestricted covariance — too many parameters,
                            overfits on 16K samples in 50D space
                - "tied"  = all clusters share one matrix — too restrictive
                - "diag"  = diagonal only — assumes feature independence,
                            good balance of expressiveness and stability
                - "spherical" = scalar covariance — too simple

            random_state:
                Seed for reproducibility. Always set this in production.

            max_iter:
                Maximum EM iterations. 200 is enough for convergence
                on this dataset; sklearn warns if not converged.
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter

        # PCA for dimensionality reduction before GMM
        self.pca = PCA(
            n_components=n_components,
            random_state=random_state
        )

        # GMM for soft clustering
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1,          # print EM iteration progress
            verbose_interval=20,
            n_init=3,           # WHY n_init=3: run EM 3 times with different
                                # random initializations, keep the best result.
                                # Prevents getting stuck in poor local optima.
        )

        self.is_fitted = False

        logger.info(
            f"FuzzyClusterer initialized: "
            f"n_clusters={n_clusters}, "
            f"pca_dims={n_components}, "
            f"cov_type={covariance_type}"
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embedding_matrix: np.ndarray) -> "FuzzyClusterer":
        """
        Fit PCA + GMM on the full embedding matrix.

        Args:
            embedding_matrix: float32 array of shape (N, 384)

        Returns:
            self (for method chaining)

        PIPELINE:
            1. L2-normalize embeddings (already done, but reconfirm)
            2. Reduce dimensions with PCA (384 → 50)
            3. Fit GMM on reduced embeddings using EM algorithm
        """
        logger.info(
            f"Fitting FuzzyClusterer on {embedding_matrix.shape[0]:,} "
            f"documents..."
        )

        # Step 1: Ensure L2-normalization
        # WHY: Cosine similarity is our distance metric. Normalizing ensures
        # PCA operates on directions (semantics), not magnitudes.
        X = normalize(embedding_matrix, norm="l2")

        # Step 2: PCA dimensionality reduction
        logger.info(
            f"Running PCA: {embedding_matrix.shape[1]}D → {self.n_components}D..."
        )
        X_reduced = self.pca.fit_transform(X)

        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA complete. Explained variance: {explained:.1%} "
            f"retained in {self.n_components} components"
        )

        # Step 3: Fit GMM
        logger.info(
            f"Fitting GMM with {self.n_clusters} components "
            f"(covariance='{self.covariance_type}', n_init=3)..."
        )
        self.gmm.fit(X_reduced)

        if not self.gmm.converged_:
            logger.warning(
                "GMM did not converge! Consider increasing max_iter "
                "or reducing n_clusters."
            )
        else:
            logger.info(
                f"GMM converged in {self.gmm.n_iter_} iterations. "
                f"Log-likelihood: {self.gmm.lower_bound_:.4f}"
            )

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Compute soft cluster membership probabilities for all documents.

        Args:
            embedding_matrix: float32 array of shape (N, 384)

        Returns:
            float64 array of shape (N, n_clusters)
            Each row sums to 1.0.
            Entry [i, k] = probability that document i belongs to cluster k.

        This is the CORE of fuzzy clustering — every document belongs
        to EVERY cluster with some probability. The dominant cluster is
        just argmax, but boundary documents have spread-out distributions.
        """
        self._check_fitted()

        X = normalize(embedding_matrix, norm="l2")
        X_reduced = self.pca.transform(X)

        # GMM predict_proba: shape (N, n_clusters), rows sum to 1
        proba_matrix = self.gmm.predict_proba(X_reduced)

        return proba_matrix

    def predict_single(self, embedding: np.ndarray) -> dict:
        """
        Predict cluster memberships for a single embedding vector.
        Used by the API and cache for query-time cluster assignment.

        Args:
            embedding: float32 array of shape (384,)

        Returns:
            {
                "dominant_cluster":    int,
                "dominant_prob":       float,
                "top_clusters":        [(cluster_id, prob), ...],  # top 3
                "all_probabilities":   [float, ...]  # all n_clusters values
            }
        """
        self._check_fitted()

        # Reshape single vector for sklearn: (1, 384)
        vec_2d = embedding.reshape(1, -1)
        proba = self.predict_proba(vec_2d)[0]  # shape (n_clusters,)

        # Sort clusters by probability descending
        sorted_indices = np.argsort(proba)[::-1]

        dominant_cluster = int(sorted_indices[0])
        dominant_prob = float(proba[dominant_cluster])

        # Top 3 clusters with their probabilities
        top_clusters = [
            (int(idx), round(float(proba[idx]), 6))
            for idx in sorted_indices[:3]
        ]

        return {
            "dominant_cluster": dominant_cluster,
            "dominant_prob": round(dominant_prob, 6),
            "top_clusters": top_clusters,
            "all_probabilities": proba.tolist(),
        }

    def get_cluster_assignments(
        self,
        embedding_matrix: np.ndarray,
        soft_threshold: float = 0.10,
    ) -> list[dict]:
        """
        Build a rich assignment record for every document.

        Args:
            embedding_matrix: shape (N, 384)
            soft_threshold:
                Minimum probability to include a cluster in "memberships".
                WHY 0.10: clusters below 10% probability are noise.
                Documents with 3+ clusters above threshold are truly
                multi-topic — exactly what fuzzy clustering finds.

        Returns:
            List of N dicts:
            {
                "doc_index":      int,   ← row index in the matrix
                "dominant":       int,   ← argmax cluster
                "dominant_prob":  float,
                "memberships":    {cluster_id: probability, ...}
                                  ← only clusters above soft_threshold
            }
        """
        self._check_fitted()

        logger.info(
            f"Computing cluster assignments for "
            f"{embedding_matrix.shape[0]:,} documents..."
        )

        proba_matrix = self.predict_proba(embedding_matrix)

        assignments = []
        for doc_idx, proba_row in enumerate(proba_matrix):

            dominant = int(np.argmax(proba_row))
            dominant_prob = float(proba_row[dominant])

            # Only keep clusters above threshold
            memberships = {
                int(k): round(float(p), 6)
                for k, p in enumerate(proba_row)
                if p >= soft_threshold
            }

            assignments.append({
                "doc_index": doc_idx,
                "dominant": dominant,
                "dominant_prob": round(dominant_prob, 6),
                "memberships": memberships,
            })

        # Log cluster size distribution
        dominant_counts = np.bincount(
            [a["dominant"] for a in assignments],
            minlength=self.n_clusters
        )
        logger.info("Cluster size distribution (dominant assignments):")
        for cluster_id, count in enumerate(dominant_counts):
            bar = "█" * (count // 50)
            logger.info(f"  Cluster {cluster_id:2d}: {count:5d} docs  {bar}")

        # Log multi-membership stats
        multi_member = sum(
            1 for a in assignments if len(a["memberships"]) > 1
        )
        logger.info(
            f"Documents with multiple memberships "
            f"(threshold={soft_threshold}): "
            f"{multi_member:,} / {len(assignments):,} "
            f"({multi_member/len(assignments):.1%})"
        )

        return assignments

    # ------------------------------------------------------------------
    # Boundary document analysis
    # ------------------------------------------------------------------

    def find_boundary_documents(
        self,
        assignments: list[dict],
        texts: list[str],
        n_examples: int = 5,
    ) -> list[dict]:
        """
        Find documents that sit on the boundary between clusters —
        i.e., documents with the most evenly spread probabilities.

        WHY THIS IS INTERESTING:
            Boundary documents are multi-topic posts. They reveal where
            clusters overlap and validate that fuzzy clustering is
            working correctly.

            A document with probabilities like:
              {0: 0.34, 3: 0.31, 7: 0.28} is deeply ambiguous —
            K-Means would arbitrarily assign it to cluster 0,
            GMM correctly keeps all three signals.

        Returns:
            List of boundary document dicts with text snippet
        """
        # Entropy measures spread of a probability distribution.
        # Higher entropy = more evenly spread = more "boundary"
        # H(p) = -sum(p_i * log(p_i))
        def entropy(memberships: dict) -> float:
            probs = np.array(list(memberships.values()))
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log(probs + 1e-10)))

        # Sort by entropy descending
        scored = sorted(
            assignments,
            key=lambda a: entropy(a["memberships"]),
            reverse=True
        )

        boundary_docs = []
        for assignment in scored[:n_examples]:
            doc_idx = assignment["doc_index"]
            boundary_docs.append({
                **assignment,
                "snippet": texts[doc_idx][:200] if doc_idx < len(texts) else "",
                "entropy": round(entropy(assignment["memberships"]), 4)
            })

        return boundary_docs

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def score_model(self, embedding_matrix: np.ndarray) -> dict:
        """
        Compute model quality metrics.

        Returns dict with:
        - log_likelihood:   higher = better fit (GMM objective)
        - bic:              Bayesian Information Criterion
                            lower = better (penalizes complexity)
        - aic:              Akaike Information Criterion
                            lower = better (less penalty than BIC)
        - converged:        did EM converge?
        - n_iter:           how many EM iterations to converge

        WHY BIC/AIC:
            These metrics help choose the optimal number of clusters.
            Plot BIC vs n_clusters — the "elbow" is optimal n_clusters.
        """
        self._check_fitted()
        X = normalize(embedding_matrix, norm="l2")
        X_reduced = self.pca.transform(X)

        return {
            "log_likelihood": round(float(self.gmm.score(X_reduced)), 4),
            "bic": round(float(self.gmm.bic(X_reduced)), 2),
            "aic": round(float(self.gmm.aic(X_reduced)), 2),
            "converged": bool(self.gmm.converged_),
            "n_iter": int(self.gmm.n_iter_),
            "n_clusters": self.n_clusters,
            "pca_variance_retained": round(
                float(self.pca.explained_variance_ratio_.sum()), 4
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the fitted PCA + GMM to disk using pickle.

        WHY PICKLE (not joblib):
            Both work. pickle is stdlib, no extra dependency.
            For very large models, joblib is faster due to numpy
            memory mapping — not needed here.
        """
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(
            f"FuzzyClusterer saved → {path} "
            f"({os.path.getsize(path) / 1_000_000:.2f} MB)"
        )

    @classmethod
    def load(cls, path: str) -> "FuzzyClusterer":
        """
        Load a previously fitted FuzzyClusterer from disk.

        Args:
            path: path to the .pkl file

        Returns:
            Fitted FuzzyClusterer instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cluster model not found: {path}\n"
                f"Run: python scripts/build_clusters.py"
            )
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(
            f"FuzzyClusterer loaded from {path}. "
            f"n_clusters={obj.n_clusters}, fitted={obj.is_fitted}"
        )
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "FuzzyClusterer is not fitted yet. "
                "Call .fit(embedding_matrix) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"FuzzyClusterer("
            f"n_clusters={self.n_clusters}, "
            f"pca_dims={self.n_components}, "
            f"cov='{self.covariance_type}', "
            f"status={status})"
        )