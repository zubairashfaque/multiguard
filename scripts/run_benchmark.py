#!/usr/bin/env python3
"""Run benchmarks across all checkpoints.

Usage:
    python scripts/run_benchmark.py --all-checkpoints
    python scripts/run_benchmark.py --checkpoint models/final/best_fusion
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard benchmarking")
    parser.add_argument("--all-checkpoints", action="store_true", help="Benchmark all checkpoints")
    parser.add_argument("--checkpoint", help="Specific checkpoint to benchmark")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info("Starting benchmark run...")

    raise NotImplementedError("Implement after evaluation pipeline is ready")


if __name__ == "__main__":
    main()
