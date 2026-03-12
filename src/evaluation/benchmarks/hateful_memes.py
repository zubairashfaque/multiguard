"""Hateful Memes Challenge benchmark evaluation (AUROC / F1)."""

from typing import Any

from src.utils.logging import get_logger
from src.utils.registry import BENCHMARK_REGISTRY

logger = get_logger(__name__)


@BENCHMARK_REGISTRY.register("hateful_memes")
class HatefulMemesBenchmark:
    """Facebook Hateful Memes Challenge benchmark.

    Evaluates multimodal hate speech detection on ~10K memes.
    Primary metric: AUROC. Secondary: F1, accuracy.
    """

    def __init__(self, data_dir: str = "data/processed/test", **kwargs: Any) -> None:
        self.data_dir = data_dir
        self.primary_metric = "auroc"

    def run(self, model: Any) -> dict[str, float]:
        """Run the Hateful Memes benchmark.

        Args:
            model: Trained multimodal model.

        Returns:
            Dict of benchmark metrics.
        """
        raise NotImplementedError("Implement after data pipeline and model are functional")
