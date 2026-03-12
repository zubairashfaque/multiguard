"""GradCAM for visual attention explanation."""

from typing import Any

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for vision models.

    Generates heatmaps showing which image regions contributed to predictions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor.
            target_class: Target class index (None = predicted class).

        Returns:
            Heatmap tensor [H, W].
        """
        self.model.eval()
        output = self.model(input_tensor)
        logits = output if isinstance(output, torch.Tensor) else output.get("logits", output)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
