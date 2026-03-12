"""Low-rank tensor fusion for multimodal features."""

import torch
import torch.nn as nn

from src.utils.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("tensor_fusion")
class TensorFusion(nn.Module):
    """Low-rank tensor fusion network.

    Computes outer product of modality representations in a
    low-rank factorized form to capture cross-modal interactions.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        rank: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim

        # Low-rank factors
        self.vision_factor = nn.Linear(vision_dim, rank, bias=False)
        self.text_factor = nn.Linear(text_dim, rank, bias=False)
        self.fusion_weights = nn.Linear(rank, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Low-rank tensor fusion.

        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]

        Returns:
            Fused features [B, hidden_dim].
        """
        v_factor = self.vision_factor(vision_features)  # [B, rank]
        t_factor = self.text_factor(text_features)  # [B, rank]

        # Element-wise product in low-rank space
        fused = v_factor * t_factor  # [B, rank]
        fused = self.fusion_weights(fused)  # [B, hidden_dim]
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)

        return fused
