"""Tokenizer wrappers for text encoding."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextTokenizer:
    """Wrapper around HuggingFace tokenizers for consistent text encoding."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self._tokenizer: Any = None

    def load(self) -> None:
        """Lazy-load the tokenizer."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Loaded tokenizer: {self.model_name}")

    def encode(self, text: str) -> dict[str, Any]:
        """Tokenize a single text string."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

    def batch_encode(self, texts: list[str]) -> dict[str, Any]:
        """Tokenize a batch of text strings."""
        if self._tokenizer is None:
            self.load()
        return self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
