"""Visualization utilities for explanations and attributions."""

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a GradCAM heatmap on an image.

    Args:
        image: Original image [H, W, 3], values in [0, 255].
        heatmap: Heatmap [H, W], values in [0, 1].
        alpha: Blending factor.
        colormap: Matplotlib colormap name.

    Returns:
        Blended image [H, W, 3].
    """
    import cv2

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return blended


def save_explanation(
    image: np.ndarray,
    text: str,
    heatmap: np.ndarray | None = None,
    text_attributions: np.ndarray | None = None,
    output_path: str | Path = "reports/figures/explanation.png",
) -> None:
    """Save a visual explanation to disk.

    Args:
        image: Original image.
        text: Input text.
        heatmap: Optional GradCAM heatmap.
        text_attributions: Optional token-level attribution scores.
        output_path: Where to save the figure.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    logger.info(f"Explanation saved to {output_path}")
    raise NotImplementedError("Implement matplotlib visualization")
