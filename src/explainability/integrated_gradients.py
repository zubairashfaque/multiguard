"""Integrated Gradients for feature attribution."""

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class IntegratedGradients:
    """Integrated Gradients attribution for text and image inputs.

    Computes feature importance by integrating gradients along a path
    from a baseline to the actual input.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50) -> None:
        self.model = model
        self.n_steps = n_steps

    def attribute(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor | None = None,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """Compute integrated gradients attribution.

        Args:
            input_tensor: Input tensor [B, ...].
            baseline: Baseline tensor (same shape), defaults to zeros.
            target_class: Target class for attribution.

        Returns:
            Attribution tensor (same shape as input).
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=input_tensor.device)
        scaled_inputs = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]

        # Compute gradients at each step
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            output = self.model(scaled_input)
            logits = output if isinstance(output, torch.Tensor) else output.get("logits", output)

            if target_class is None:
                target_class = logits.argmax(dim=-1).item()

            self.model.zero_grad()
            logits[0, target_class].backward()
            gradients.append(scaled_input.grad.detach())

        # Integrate
        avg_gradients = torch.stack(gradients).mean(dim=0)
        attributions = (input_tensor - baseline) * avg_gradients

        return attributions
