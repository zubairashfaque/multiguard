"""Visualization utilities for explanations and attributions."""

from pathlib import Path

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
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
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

    Creates a figure with:
    - Left: original image with optional GradCAM overlay
    - Right: text with token-level attribution highlighting

    Args:
        image: Original image [H, W, 3], values in [0, 255].
        text: Input text.
        heatmap: Optional GradCAM heatmap [H, W], values in [0, 1].
        text_attributions: Optional token-level attribution scores.
        output_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    has_heatmap = heatmap is not None
    has_text_attr = text_attributions is not None
    ncols = 1 + int(has_heatmap)

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    # Show image (with or without heatmap overlay)
    if has_heatmap:
        overlaid = overlay_heatmap(image, heatmap)
        axes[0].imshow(overlaid)
        axes[0].set_title("GradCAM Overlay")
    else:
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Show text with attributions if we have a second panel
    if has_heatmap:
        ax_text = axes[1]
    else:
        ax_text = axes[0]

    if has_text_attr and has_heatmap:
        tokens = text.split()
        attr_normalized = text_attributions[: len(tokens)]
        if len(attr_normalized) > 0:
            attr_min = attr_normalized.min()
            attr_max = attr_normalized.max()
            if attr_max > attr_min:
                attr_normalized = (attr_normalized - attr_min) / (attr_max - attr_min)
            else:
                attr_normalized = np.zeros_like(attr_normalized)

        colored_text = ""
        for i, token in enumerate(tokens):
            if i < len(attr_normalized):
                score = attr_normalized[i]
                colored_text += f"{token} ({score:.2f})  "
            else:
                colored_text += f"{token}  "

        ax_text.text(
            0.05,
            0.5,
            colored_text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment="center",
            wrap=True,
        )
        ax_text.set_title("Text Attributions")
        ax_text.axis("off")
    elif has_heatmap:
        ax_text.text(
            0.05,
            0.5,
            text,
            transform=ax_text.transAxes,
            fontsize=12,
            verticalalignment="center",
            wrap=True,
        )
        ax_text.set_title("Input Text")
        ax_text.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Explanation saved to {output_path}")
