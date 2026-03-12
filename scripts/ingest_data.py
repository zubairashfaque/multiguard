#!/usr/bin/env python3
"""Data ingestion script for Hateful Memes and other datasets.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --dataset hateful_memes
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard data ingestion")
    parser.add_argument(
        "--dataset",
        default="hateful_memes",
        choices=["hateful_memes"],
        help="Dataset to ingest",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info(f"Ingesting dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")

    raise NotImplementedError(
        "Implement data download and preprocessing. "
        "Hateful Memes requires access approval from Facebook Research."
    )


if __name__ == "__main__":
    main()
