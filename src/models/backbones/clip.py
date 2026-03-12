"""CLIP / SigLIP multimodal backbone wrappers."""

import torch
import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.registry import BACKBONE_REGISTRY

logger = get_logger(__name__)


@BACKBONE_REGISTRY.register("clip")
class CLIPBackbone(nn.Module):
    """CLIP backbone for joint vision-language encoding.

    Provides separate vision and text encoders with aligned embedding spaces.
    Can be used for zero-shot classification and as a feature extractor.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        projection_dim: int = 768,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.projection_dim = projection_dim
        self.model: nn.Module | None = None
        self.processor: object | None = None
        self._freeze = freeze
        logger.info(f"CLIPBackbone initialized: {model_id}")

    def load_model(self) -> None:
        """Lazy-load CLIP model and processor."""
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        if self._freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        logger.info(f"Loaded CLIP model: {self.model_id}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to CLIP embedding space.

        Args:
            pixel_values: Image tensor [B, C, H, W].

        Returns:
            Image embeddings [B, projection_dim].
        """
        if self.model is None:
            self.load_model()
        return self.model.get_image_features(pixel_values=pixel_values)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode text to CLIP embedding space.

        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Attention mask [B, seq_len].

        Returns:
            Text embeddings [B, projection_dim].
        """
        if self.model is None:
            self.load_model()
        return self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both image and text.

        Returns:
            Tuple of (image_embeddings, text_embeddings).
        """
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        return image_features, text_features


@BACKBONE_REGISTRY.register("siglip")
class SigLIPBackbone(CLIPBackbone):
    """SigLIP backbone — sigmoid-based contrastive learning."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "google/siglip-base-patch16-224")
        super().__init__(**kwargs)
