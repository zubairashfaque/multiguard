"""Model comparison utilities for experiment tracking."""

from typing import Any

from src.utils.io import load_json, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compare_experiments(
    experiment_results: list[dict[str, Any]],
    primary_metric: str = "auroc",
) -> dict[str, Any]:
    """Compare multiple experiment results.

    Args:
        experiment_results: List of dicts with 'name' and metric values.
        primary_metric: Metric to rank by.

    Returns:
        Comparison summary with ranking.
    """
    sorted_results = sorted(
        experiment_results,
        key=lambda x: x.get(primary_metric, 0.0),
        reverse=True,
    )

    comparison = {
        "ranking": [
            {"rank": i + 1, "name": r["name"], primary_metric: r.get(primary_metric, 0.0)}
            for i, r in enumerate(sorted_results)
        ],
        "best": sorted_results[0] if sorted_results else None,
        "primary_metric": primary_metric,
    }

    logger.info(f"Best model: {comparison['best']['name'] if comparison['best'] else 'N/A'}")
    return comparison
