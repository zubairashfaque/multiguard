"""Dataset splitting utilities with stratification support."""

from typing import Any

from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_dataset(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str | None = "label",
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into train/val/test with optional stratification.

    Args:
        samples: List of sample dicts.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        stratify_by: Key to stratify on (e.g., 'label'), or None.
        seed: Random seed.

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    labels = [s[stratify_by] for s in samples] if stratify_by else None

    train, temp = train_test_split(
        samples,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels,
    )

    val_fraction = val_ratio / (val_ratio + test_ratio)
    temp_labels = [s[stratify_by] for s in temp] if stratify_by else None
    val, test = train_test_split(
        temp,
        test_size=(1 - val_fraction),
        random_state=seed,
        stratify=temp_labels,
    )

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
