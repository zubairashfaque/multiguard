"""Training metrics for multimodal classification and retrieval."""

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Model logits or probabilities [B, C] or [B].
        labels: Ground truth labels [B].
        threshold: Decision threshold for binary classification.

    Returns:
        Dict with accuracy, f1, precision, recall, auroc.
    """
    if predictions.dim() > 1 and predictions.shape[1] == 2:
        probs = torch.softmax(predictions, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= threshold).astype(int)
    elif predictions.dim() > 1:
        probs = torch.softmax(predictions, dim=-1).cpu().numpy()
        preds = predictions.argmax(dim=-1).cpu().numpy()
    else:
        probs = torch.sigmoid(predictions).cpu().numpy()
        preds = (probs >= threshold).astype(int)

    labels_np = labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(labels_np, preds),
        "f1": f1_score(labels_np, preds, average="macro", zero_division=0),
        "precision": precision_score(labels_np, preds, average="macro", zero_division=0),
        "recall": recall_score(labels_np, preds, average="macro", zero_division=0),
    }

    try:
        if predictions.dim() > 1 and predictions.shape[1] == 2:
            metrics["auroc"] = roc_auc_score(labels_np, probs)
        else:
            metrics["auroc"] = roc_auc_score(labels_np, probs, multi_class="ovr")
    except ValueError:
        metrics["auroc"] = 0.0

    return metrics
