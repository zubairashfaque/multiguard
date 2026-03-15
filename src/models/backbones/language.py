"""Language backbone wrappers: BERT, RoBERTa."""

import torch
import torch.nn as nn

from src.utils.logging import get_logger
from src.utils.registry import BACKBONE_REGISTRY

logger = get_logger(__name__)


class LanguageBackbone(nn.Module):
    """Base language encoder using HuggingFace transformers.

    Extracts text features from pretrained language models.
    Supports BERT and RoBERTa architectures via model_id.
    """

    def __init__(
        self,
        model_id: str = "roberta-base",
        hidden_size: int = 768,
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self._freeze = freeze
        self.encoder: nn.Module | None = None

        logger.info(f"LanguageBackbone initialized: {model_id}, hidden_size={hidden_size}")

    def load_encoder(self) -> None:
        """Lazy-load the language encoder from HuggingFace."""
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(self.model_id)
        if self._freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Loaded and froze language encoder: {self.model_id}")
        else:
            logger.info(f"Loaded language encoder: {self.model_id}")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract text features.

        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Attention mask [B, seq_len].

        Returns:
            Text embeddings [B, hidden_size].
        """
        if self.encoder is None:
            self.load_encoder()
            self.encoder.to(input_ids.device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS token


@BACKBONE_REGISTRY.register("roberta")
class RoBERTaBackbone(LanguageBackbone):
    """RoBERTa backbone."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "roberta-base")
        kwargs.setdefault("hidden_size", 768)
        super().__init__(**kwargs)


@BACKBONE_REGISTRY.register("bert")
class BERTBackbone(LanguageBackbone):
    """BERT backbone."""

    def __init__(self, **kwargs: dict) -> None:
        kwargs.setdefault("model_id", "bert-base-uncased")
        kwargs.setdefault("hidden_size", 768)
        super().__init__(**kwargs)
