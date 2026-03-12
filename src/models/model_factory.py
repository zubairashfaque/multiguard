"""Factory for building multimodal models from config."""

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

from src.utils.logging import get_logger
from src.utils.registry import BACKBONE_REGISTRY, FUSION_REGISTRY, HEAD_REGISTRY

logger = get_logger(__name__)


class MultiGuardModel(nn.Module):
    """Full multimodal model: vision backbone + text backbone + fusion + head."""

    def __init__(
        self,
        vision_backbone: nn.Module,
        text_backbone: nn.Module,
        fusion: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.fusion = fusion
        self.head = head

    def forward(self, batch: dict) -> dict:
        """Forward pass through the full pipeline.

        Args:
            batch: Dict with 'pixel_values', 'input_ids', 'attention_mask'.

        Returns:
            Dict with 'logits' and optionally 'embeddings'.
        """
        vision_features = self.vision_backbone(batch["pixel_values"])
        text_features = self.text_backbone(
            batch["input_ids"], batch.get("attention_mask")
        )
        fused = self.fusion(vision_features, text_features)
        logits = self.head(fused)
        return {"logits": logits, "fused_features": fused}


def build_model(config: DictConfig) -> MultiGuardModel:
    """Build a MultiGuard model from config.

    Args:
        config: Model configuration with backbone, fusion, and head specs.

    Returns:
        Assembled MultiGuardModel.
    """
    logger.info("Building MultiGuard model from config...")

    # Build vision backbone
    vision_cfg = config.get("model", {})
    vision_backbone_name = vision_cfg.get("vision_backbone", "openai/clip-vit-large-patch14")
    logger.info(f"Vision backbone: {vision_backbone_name}")

    # Build text backbone
    text_backbone_name = vision_cfg.get("text_backbone", "roberta-base")
    logger.info(f"Text backbone: {text_backbone_name}")

    # Build fusion
    fusion_name = vision_cfg.get("fusion", "late_fusion")
    logger.info(f"Fusion strategy: {fusion_name}")

    # Build head
    num_labels = vision_cfg.get("num_labels", 2)
    logger.info(f"Classification head: {num_labels} labels")

    raise NotImplementedError(
        "Full model assembly requires backbone loading. "
        "Implement after backbones are tested individually."
    )
