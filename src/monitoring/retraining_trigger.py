"""Automated retraining trigger based on monitoring signals."""

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrainingTrigger:
    """Decide when to trigger model retraining based on drift and performance.

    Evaluates monitoring signals and triggers retraining pipeline
    when thresholds are exceeded.
    """

    def __init__(
        self,
        performance_threshold: float = 0.05,
        drift_threshold: float = 0.1,
        min_samples: int = 1000,
    ) -> None:
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples

    def should_retrain(
        self,
        current_performance: float,
        baseline_performance: float,
        drift_score: float,
        num_samples: int,
    ) -> bool:
        """Determine if retraining is needed.

        Args:
            current_performance: Current model AUROC.
            baseline_performance: Baseline AUROC from training.
            drift_score: Data drift magnitude.
            num_samples: Number of new samples since last training.

        Returns:
            True if retraining should be triggered.
        """
        performance_drop = baseline_performance - current_performance
        should = (
            performance_drop > self.performance_threshold
            or drift_score > self.drift_threshold
        ) and num_samples >= self.min_samples

        if should:
            logger.info(
                f"Retraining triggered: perf_drop={performance_drop:.4f}, "
                f"drift={drift_score:.4f}, samples={num_samples}"
            )
        return should
