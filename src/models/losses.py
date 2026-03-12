"""Loss functions for multi-task multimodal training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Weighted combination of multiple task losses.

    Supports automatic loss weighting via learnable parameters.
    """

    def __init__(
        self,
        task_names: list[str],
        task_weights: dict[str, float] | None = None,
        learnable_weights: bool = False,
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.learnable_weights = learnable_weights

        if learnable_weights:
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1)) for name in task_names
            })
        else:
            self.task_weights = task_weights or {name: 1.0 for name in task_names}

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            losses: Dict mapping task name to scalar loss tensor.

        Returns:
            Combined scalar loss.
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

        for name in self.task_names:
            if name not in losses:
                continue
            if self.learnable_weights:
                precision = torch.exp(-self.log_vars[name])
                total_loss = total_loss + precision * losses[name] + self.log_vars[name]
            else:
                total_loss = total_loss + self.task_weights[name] * losses[name]

        return total_loss


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for retrieval training."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss between two sets of embeddings.

        Args:
            embeddings_a: Normalized embeddings [B, D].
            embeddings_b: Normalized embeddings [B, D].

        Returns:
            Scalar loss.
        """
        logits = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [B, C].
            targets: Ground truth labels [B].

        Returns:
            Scalar loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
