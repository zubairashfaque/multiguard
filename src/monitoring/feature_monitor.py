"""Feature quality monitoring."""

from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureMonitor:
    """Monitor feature quality and distributions over time.

    Tracks statistics of text and image features to detect anomalies
    using z-score based detection on running statistics.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self.history: list[dict[str, Any]] = []
        self.z_threshold = z_threshold

    def log_batch(self, features: dict[str, Any]) -> None:
        """Log feature statistics for a batch.

        Args:
            features: Dict with feature names and their values (numpy arrays).
        """
        stats = {}
        for name, values in features.items():
            arr = np.asarray(values)
            stats[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "nan_count": int(np.isnan(arr).sum()),
                "inf_count": int(np.isinf(arr).sum()),
            }
        self.history.append(stats)
        logger.debug(f"Logged feature batch: {len(self.history)} total")

    def check_anomalies(self) -> list[str]:
        """Check for anomalous feature values using z-score detection.

        Compares the latest batch statistics against the running history.

        Returns:
            List of anomaly descriptions.
        """
        if len(self.history) < 3:
            return []

        anomalies = []
        latest = self.history[-1]

        for feature_name, latest_stats in latest.items():
            # Check for NaN/Inf
            if latest_stats["nan_count"] > 0:
                anomalies.append(f"{feature_name}: {latest_stats['nan_count']} NaN values detected")
            if latest_stats["inf_count"] > 0:
                anomalies.append(f"{feature_name}: {latest_stats['inf_count']} Inf values detected")

            # Z-score check on mean against history
            historical_means = [
                h[feature_name]["mean"] for h in self.history[:-1] if feature_name in h
            ]
            if len(historical_means) >= 2:
                hist_mean = np.mean(historical_means)
                hist_std = np.std(historical_means)
                if hist_std > 1e-8:
                    z_score = abs(latest_stats["mean"] - hist_mean) / hist_std
                    if z_score > self.z_threshold:
                        anomalies.append(
                            f"{feature_name}: mean={latest_stats['mean']:.4f} is {z_score:.1f} "
                            f"std deviations from historical mean={hist_mean:.4f}"
                        )

        if anomalies:
            logger.warning(f"Feature anomalies detected: {anomalies}")

        return anomalies
