"""Contrastive embedding head for multimodal retrieval."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("retrieval")
class RetrievalHead(nn.Module):
    """Retrieval head that projects features into a shared embedding space.

    Used for multimodal retrieval via contrastive learning (InfoNCE).
    """

    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.normalize = normalize

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features into embedding space.

        Args:
            features: Input features [B, input_dim].

        Returns:
            Normalized embeddings [B, embedding_dim].
        """
        embeddings = self.projection(features)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
