"""Vision backbone wrappers: ViT, DINOv2, Swin."""

import torch
import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.registry import BACKBONE_REGISTRY

logger = get_logger(__name__)


class VisionBackbone(nn.Module):
    """Base vision encoder using timm or HuggingFace ViT models.

    Extracts image features from pretrained vision transformers.
    Supports ViT, DINOv2, and Swin architectures via model_id.
    """

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        hidden_size: int = 768,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.encoder: nn.Module | None = None

        if freeze and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

        logger.info(f"VisionBackbone initialized: {model_id}, hidden_size={hidden_size}")

    def load_encoder(self) -> None:
        """Lazy-load the vision encoder from HuggingFace."""
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(self.model_id)
        logger.info(f"Loaded vision encoder: {self.model_id}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            pixel_values: Image tensor [B, C, H, W].

        Returns:
            Image embeddings [B, hidden_size].
        """
        if self.encoder is None:
            self.load_encoder()
            self.encoder.to(pixel_values.device)
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]  # CLS token


@BACKBONE_REGISTRY.register("vit")
class ViTBackbone(VisionBackbone):
    """ViT backbone (google/vit-base-patch16-224)."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "google/vit-base-patch16-224")
        kwargs.setdefault("hidden_size", 768)
        super().__init__(**kwargs)


@BACKBONE_REGISTRY.register("dinov2")
class DINOv2Backbone(VisionBackbone):
    """DINOv2 backbone (facebook/dinov2-base)."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "facebook/dinov2-base")
        kwargs.setdefault("hidden_size", 768)
        super().__init__(**kwargs)


@BACKBONE_REGISTRY.register("swin")
class SwinBackbone(VisionBackbone):
    """Swin Transformer backbone."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "microsoft/swin-base-patch4-window7-224")
        kwargs.setdefault("hidden_size", 1024)
        super().__init__(**kwargs)
