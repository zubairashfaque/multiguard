"""Visual Question Answering head."""

import torch
import torch.nn as nn

from src.utils.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("vqa")
class VQAHead(nn.Module):
    """VQA answer generation head.

    Takes fused multimodal features and predicts answer from a fixed vocabulary.
    """

    def __init__(
        self,
        input_dim: int = 256,
        answer_vocab_size: int = 3129,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, answer_vocab_size))

        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict VQA answer logits.

        Args:
            features: Fused features [B, input_dim].

        Returns:
            Answer logits [B, answer_vocab_size].
        """
        return self.head(features)
