"""Bidirectional cross-attention fusion for multimodal features."""

import torch
import torch.nn as nn

from src.utils.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion.

    Vision attends to language and language attends to vision,
    then results are combined via learned gating.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim

        # Vision-to-text cross-attention layers
        self.v2t_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Text-to-vision cross-attention layers
        self.t2v_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Bidirectional cross-attention fusion.

        Args:
            vision_features: [B, hidden_dim] (unsqueezed to [B, 1, hidden_dim]).
            text_features: [B, hidden_dim] (unsqueezed to [B, 1, hidden_dim]).

        Returns:
            Fused features [B, hidden_dim].
        """
        v = vision_features.unsqueeze(1)  # [B, 1, D]
        t = text_features.unsqueeze(1)  # [B, 1, D]

        # Vision attending to text
        v2t = v
        for layer in self.v2t_layers:
            v2t = layer(v2t, t)

        # Text attending to vision
        t2v = t
        for layer in self.t2v_layers:
            t2v = layer(t2v, v)

        v2t = v2t.squeeze(1)  # [B, D]
        t2v = t2v.squeeze(1)  # [B, D]

        # Gated combination
        gate_input = torch.cat([v2t, t2v], dim=-1)
        gate_weight = self.gate(gate_input)
        fused = gate_weight * v2t + (1 - gate_weight) * t2v

        return fused
