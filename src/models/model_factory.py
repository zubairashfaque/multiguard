"""Factory for building multimodal models from config."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

import src.models.fusion.cross_attention  # noqa: F401

# Import modules to trigger registry decorators
import src.models.fusion.late_fusion  # noqa: F401
import src.models.fusion.tensor_fusion  # noqa: F401
import src.models.heads.classifier  # noqa: F401
import src.models.heads.retrieval  # noqa: F401
import src.models.heads.vqa  # noqa: F401
from src.utils.logging import get_logger
from src.utils.registry import FUSION_REGISTRY, HEAD_REGISTRY

logger = get_logger(__name__)


class _CLIPVisionAdapter(nn.Module):
    """Adapter to use CLIP's vision encoder as a standalone backbone.

    Uses CLIPVisionModel (vision-only, no text weights) for memory efficiency,
    then projects pooler_output to the desired hidden_size.
    """

    def __init__(self, model_id: str, hidden_size: int, freeze: bool = False) -> None:
        super().__init__()
        self.model_id = model_id
        self.hidden_size = hidden_size
        self._freeze = freeze
        self.encoder: nn.Module | None = None
        self._vision_hidden_size: int | None = None
        self.projection: nn.Module | None = None

    def _load(self, device: torch.device) -> None:
        from transformers import CLIPVisionModel

        self.encoder = CLIPVisionModel.from_pretrained(self.model_id)
        self._vision_hidden_size = self.encoder.config.hidden_size

        # Add projection if hidden sizes don't match
        if self._vision_hidden_size != self.hidden_size:
            self.projection = nn.Linear(self._vision_hidden_size, self.hidden_size)
            self.projection.to(device)

        if self._freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.to(device)
        logger.info(
            f"Loaded CLIP vision encoder: {self.model_id} "
            f"(hidden={self._vision_hidden_size} → {self.hidden_size})"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            self._load(pixel_values.device)
        outputs = self.encoder(pixel_values=pixel_values)
        features = outputs.pooler_output  # [B, vision_hidden_size]
        if self.projection is not None:
            features = self.projection(features)  # [B, hidden_size]
        return features


class MultiGuardModel(nn.Module):
    """Full multimodal model: vision backbone + text backbone + fusion + head."""

    def __init__(
        self,
        vision_backbone: nn.Module,
        text_backbone: nn.Module,
        fusion: nn.Module,
        head: nn.Module,
        vision_proj: nn.Module | None = None,
        text_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.fusion = fusion
        self.head = head
        self.vision_proj = vision_proj
        self.text_proj = text_proj

    def forward(self, batch: dict) -> dict:
        """Forward pass through the full pipeline.

        Args:
            batch: Dict with 'pixel_values', 'input_ids', 'attention_mask'.

        Returns:
            Dict with 'logits' and 'fused_features'.
        """
        vision_features = self.vision_backbone(batch["pixel_values"])
        text_features = self.text_backbone(batch["input_ids"], batch.get("attention_mask"))

        if self.vision_proj is not None:
            vision_features = self.vision_proj(vision_features)
        if self.text_proj is not None:
            text_features = self.text_proj(text_features)

        fused = self.fusion(vision_features, text_features)
        logits = self.head(fused)
        return {"logits": logits, "fused_features": fused}


def build_model(config: DictConfig) -> MultiGuardModel:
    """Build a MultiGuard model from config.

    Args:
        config: Full config with 'model' section containing:
            - vision_backbone: HuggingFace model ID
            - vision_dim: Vision feature dimension
            - text_backbone: HuggingFace model ID
            - text_dim: Text feature dimension
            - fusion: Fusion strategy name (late_fusion, cross_attention, tensor_fusion)
            - num_labels: Number of output classes
            - dropout: Dropout rate

    Returns:
        Assembled MultiGuardModel.
    """
    model_cfg = config.model
    vision_id = model_cfg.vision_backbone
    text_id = model_cfg.text_backbone
    vision_dim = model_cfg.get("vision_dim", 768)
    text_dim = model_cfg.get("text_dim", 768)
    fusion_name = model_cfg.fusion
    num_labels = model_cfg.get("num_labels", 2)
    dropout = model_cfg.get("dropout", 0.1)
    freeze_backbones = model_cfg.get("freeze_backbones", False)

    logger.info(f"Building model: vision={vision_id}, text={text_id}, fusion={fusion_name}")

    # --- Vision backbone ---
    if "clip" in vision_id.lower():
        vision_backbone = _CLIPVisionAdapter(
            model_id=vision_id, hidden_size=vision_dim, freeze=freeze_backbones
        )
    else:
        from src.models.backbones.vision import VisionBackbone

        vision_backbone = VisionBackbone(
            model_id=vision_id, hidden_size=vision_dim, freeze=freeze_backbones
        )

    # --- Text backbone ---
    from src.models.backbones.language import LanguageBackbone

    text_backbone = LanguageBackbone(
        model_id=text_id,
        hidden_size=text_dim,
        freeze=freeze_backbones,
    )

    # --- Projection layers (align dims for fusion if needed) ---
    vision_proj = None
    text_proj = None
    fusion_cls = FUSION_REGISTRY.get(fusion_name)

    if fusion_name == "cross_attention":
        # Cross-attention requires same dimension for both modalities
        ca_cfg = model_cfg.get("cross_attention", {})
        hidden_dim = ca_cfg.get("hidden_dim", 768)
        if vision_dim != hidden_dim:
            vision_proj = nn.Linear(vision_dim, hidden_dim)
        if text_dim != hidden_dim:
            text_proj = nn.Linear(text_dim, hidden_dim)
        fusion = fusion_cls(
            hidden_dim=hidden_dim,
            num_heads=ca_cfg.get("num_heads", 8),
            num_layers=ca_cfg.get("num_layers", 4),
            feedforward_dim=ca_cfg.get("feedforward_dim", hidden_dim * 4),
            dropout=dropout,
        )
    elif fusion_name == "tensor_fusion":
        fusion = fusion_cls(
            vision_dim=vision_dim,
            text_dim=text_dim,
            dropout=dropout,
        )
    else:
        # late_fusion (default)
        fusion = fusion_cls(
            vision_dim=vision_dim,
            text_dim=text_dim,
            dropout=dropout,
        )

    # --- Classification head ---
    head_cls = HEAD_REGISTRY.get("classifier")
    head = head_cls(
        input_dim=fusion.output_dim,
        num_labels=num_labels,
        dropout=dropout,
    )

    model = MultiGuardModel(
        vision_backbone=vision_backbone,
        text_backbone=text_backbone,
        fusion=fusion,
        head=head,
        vision_proj=vision_proj,
        text_proj=text_proj,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built: {total_params:,} total params, {trainable_params:,} trainable")

    return model
