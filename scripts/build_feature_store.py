#!/usr/bin/env python3
"""Build Feast feature store for inference features.

Usage:
    python scripts/build_feature_store.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    setup_logging(level="INFO")
    logger.info("Building feature store...")

    raise NotImplementedError("Implement Feast feature store integration")


if __name__ == "__main__":
    main()
