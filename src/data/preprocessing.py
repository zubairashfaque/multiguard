"""Preprocessing pipelines for text and image inputs."""

from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Clean and normalize text inputs before tokenization."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def clean(self, text: str) -> str:
        """Apply text cleaning: strip whitespace, normalize unicode."""
        text = text.strip()
        return text

    def __call__(self, text: str) -> str:
        return self.clean(text)


class ImagePreprocessor:
    """Validate and normalize image inputs before transforms."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def validate(self, image_path: str | Path) -> bool:
        """Check that image path exists and has valid extension."""
        path = Path(image_path)
        return path.exists() and path.suffix.lower() in self.VALID_EXTENSIONS

    def __call__(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        if not self.validate(path):
            raise ValueError(f"Invalid image: {path}")
        return path
