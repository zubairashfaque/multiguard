"""Data and model drift detection."""

from typing import Any

import numpy as np
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detect data and prediction drift using statistical tests.

    Monitors incoming data distributions against reference data
    and triggers alerts when drift is detected.
    """

    def __init__(
        self,
        reference_data: np.ndarray | None = None,
        drift_threshold: float = 0.05,
    ) -> None:
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold

    def check_data_drift(self, current_data: np.ndarray) -> dict[str, Any]:
        """Check for data drift using Kolmogorov-Smirnov test.

        Compares each feature dimension between reference and current data.

        Args:
            current_data: Current incoming data batch [N, D].

        Returns:
            Drift report with detected features and p-values.
        """
        if self.reference_data is None:
            return {"drift_detected": False, "message": "No reference data set"}

        ref = np.asarray(self.reference_data)
        cur = np.asarray(current_data)

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
            cur = cur.reshape(-1, 1)

        n_features = min(ref.shape[1], cur.shape[1])
        drifted_features = []
        p_values = []

        for i in range(n_features):
            statistic, p_value = stats.ks_2samp(ref[:, i], cur[:, i])
            p_values.append(float(p_value))
            if p_value < self.drift_threshold:
                drifted_features.append(
                    {"feature_idx": i, "p_value": float(p_value), "ks_stat": float(statistic)}
                )

        drift_detected = len(drifted_features) > 0
        drift_score = 1.0 - np.mean(p_values) if p_values else 0.0

        if drift_detected:
            logger.warning(f"Data drift detected in {len(drifted_features)}/{n_features} features")

        return {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "drifted_features": drifted_features,
            "total_features": n_features,
            "mean_p_value": float(np.mean(p_values)) if p_values else 1.0,
        }

    def check_prediction_drift(self, predictions: np.ndarray) -> dict[str, Any]:
        """Check for prediction distribution drift using chi-squared test.

        Compares the distribution of predicted classes against reference predictions.

        Args:
            predictions: Recent model predictions (class labels) [N].

        Returns:
            Drift report with distribution comparison.
        """
        if self.reference_data is None:
            return {"drift_detected": False, "message": "No reference predictions set"}

        ref_preds = np.asarray(self.reference_data).flatten()
        cur_preds = np.asarray(predictions).flatten()

        # Compute class distributions
        all_classes = np.union1d(np.unique(ref_preds), np.unique(cur_preds))
        ref_counts = np.array([np.sum(ref_preds == c) for c in all_classes], dtype=float)
        cur_counts = np.array([np.sum(cur_preds == c) for c in all_classes], dtype=float)

        # Normalize to proportions
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()

        # Chi-squared test
        expected = ref_props * cur_counts.sum()
        expected = np.maximum(expected, 1e-8)
        statistic, p_value = stats.chisquare(cur_counts, f_exp=expected)

        drift_detected = p_value < self.drift_threshold

        if drift_detected:
            logger.warning(f"Prediction drift detected: p={p_value:.4f}")

        return {
            "drift_detected": drift_detected,
            "p_value": float(p_value),
            "chi2_stat": float(statistic),
            "reference_distribution": {str(c): float(p) for c, p in zip(all_classes, ref_props)},
            "current_distribution": {str(c): float(p) for c, p in zip(all_classes, cur_props)},
        }
