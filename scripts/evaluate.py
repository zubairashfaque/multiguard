#!/usr/bin/env python3
"""Evaluation entry point for MultiGuard.

Usage:
    python scripts/evaluate.py --config configs/experiment/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    set_seed(config.get("project", {}).get("seed", 42))

    logger.info("Starting evaluation...")
    raise NotImplementedError("Implement after model and data pipelines are ready")


if __name__ == "__main__":
    main()
