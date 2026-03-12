"""Evaluation metrics: AUROC, F1, MRR@K, and retrieval metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return 0.0


def compute_mrr_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at K for retrieval.

    Args:
        query_embeddings: Query vectors [N_q, D].
        gallery_embeddings: Gallery vectors [N_g, D].
        query_labels: Labels for queries.
        gallery_labels: Labels for gallery items.
        k: Top-K to consider.

    Returns:
        MRR@K score.
    """
    similarities = query_embeddings @ gallery_embeddings.T
    reciprocal_ranks = []

    for i in range(len(query_embeddings)):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        for rank, idx in enumerate(top_k_indices, 1):
            if gallery_labels[idx] == query_labels[i]:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))
