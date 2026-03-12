"""Multi-label classification head."""

import torch
import torch.nn as nn

from src.utils.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("classifier")
class ClassificationHead(nn.Module):
    """Classification head for multimodal content analysis.

    Takes fused multimodal features and outputs class logits.
    Supports binary and multi-class classification.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_labels: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = dim
        layers.append(nn.Linear(in_dim, num_labels))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict class logits.

        Args:
            features: Fused features [B, input_dim].

        Returns:
            Logits [B, num_labels].
        """
        return self.classifier(features)
