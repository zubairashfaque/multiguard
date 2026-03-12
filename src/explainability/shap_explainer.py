"""SHAP-based explainability for multimodal models."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """SHAP explanations for multimodal predictions.

    Uses kernel SHAP or deep SHAP to explain model predictions
    in terms of input feature contributions.
    """

    def __init__(self, model: Any, background_data: Any = None) -> None:
        self.model = model
        self.background_data = background_data
        self._explainer: Any = None

    def setup(self) -> None:
        """Initialize the SHAP explainer."""
        try:
            import shap

            self._explainer = shap.DeepExplainer(self.model, self.background_data)
            logger.info("SHAP DeepExplainer initialized")
        except Exception as e:
            logger.warning(f"SHAP setup failed: {e}")

    def explain(self, inputs: Any) -> Any:
        """Generate SHAP explanations.

        Args:
            inputs: Model inputs to explain.

        Returns:
            SHAP values.
        """
        if self._explainer is None:
            self.setup()
        return self._explainer.shap_values(inputs)
