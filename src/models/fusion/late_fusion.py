"""Late fusion: concatenate unimodal features + MLP projection."""

import torch
import torch.nn as nn

from src.utils.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("late_fusion")
class LateFusion(nn.Module):
    """Late fusion via concatenation and MLP.

    Concatenates vision and language features, then projects through MLP layers.
    Simplest fusion strategy — serves as baseline.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        input_dim = vision_dim + text_dim

        layers: list[nn.Module] = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_dim = dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse vision and text features via concatenation + MLP.

        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]

        Returns:
            Fused features [B, output_dim].
        """
        combined = torch.cat([vision_features, text_features], dim=-1)
        return self.mlp(combined)
