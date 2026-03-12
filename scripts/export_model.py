#!/usr/bin/env python3
"""Model export script (ONNX, TorchScript).

Usage:
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format torchscript
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard model export")
    parser.add_argument(
        "--format",
        required=True,
        choices=["onnx", "torchscript"],
        help="Export format",
    )
    parser.add_argument("--checkpoint", default="models/final/best_fusion", help="Model checkpoint")
    parser.add_argument("--output-dir", default="models/exported", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info(f"Exporting model to {args.format}: {args.checkpoint}")

    raise NotImplementedError("Implement after model training pipeline")


if __name__ == "__main__":
    main()
