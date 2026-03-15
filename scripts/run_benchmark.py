#!/usr/bin/env python3
"""Run benchmarks across all checkpoints.

Usage:
    python scripts/run_benchmark.py --all-checkpoints
    python scripts/run_benchmark.py --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from src.evaluation.benchmarks.hateful_memes import HatefulMemesBenchmark  # noqa: E402
from src.evaluation.comparison import compare_experiments  # noqa: E402
from src.serving.model_loader import load_model  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# Checkpoint directories to scan
CHECKPOINT_CONFIGS = {
    "baseline": {
        "checkpoint": "models/checkpoints/baseline/best_model.pt",
        "config": "configs/train/baseline.yaml",
    },
    "fusion_ablation": {
        "checkpoint": "models/checkpoints/fusion_ablation/best_model.pt",
        "config": "configs/train/fusion_ablation.yaml",
    },
    "distillation": {
        "checkpoint": "models/checkpoints/distillation/best_model.pt",
        "config": "configs/train/distillation.yaml",
    },
}


def benchmark_checkpoint(name: str, checkpoint_path: str, config_path: str) -> dict:
    """Run benchmark on a single checkpoint."""
    from src.utils.config import load_config

    logger.info(f"Benchmarking: {name} ({checkpoint_path})")

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )

    benchmark = HatefulMemesBenchmark(
        data_dir="data/raw/hateful_memes",
        split="val",
        tokenizer_name=config.model.text_backbone,
        batch_size=config.training.get("batch_size", 8),
        num_workers=config.device.get("num_workers", 4),
    )

    metrics = benchmark.run(model)
    metrics["name"] = name
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard benchmarking")
    parser.add_argument("--all-checkpoints", action="store_true", help="Benchmark all checkpoints")
    parser.add_argument("--checkpoint", help="Specific checkpoint path to benchmark")
    parser.add_argument("--config", help="Config path (required with --checkpoint)")
    parser.add_argument("--name", default="custom", help="Name for the benchmark run")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info("Starting benchmark run...")

    all_results = []

    if args.all_checkpoints:
        for name, cfg in CHECKPOINT_CONFIGS.items():
            ckpt = Path(cfg["checkpoint"])
            if ckpt.exists():
                result = benchmark_checkpoint(name, str(ckpt), cfg["config"])
                all_results.append(result)
            else:
                logger.warning(f"Skipping {name}: checkpoint not found at {ckpt}")
    elif args.checkpoint:
        if not args.config:
            raise ValueError("--config is required when using --checkpoint")
        result = benchmark_checkpoint(args.name, args.checkpoint, args.config)
        all_results.append(result)
    else:
        raise ValueError("Provide --all-checkpoints or --checkpoint")

    if not all_results:
        logger.warning("No checkpoints found to benchmark")
        return

    # Compare results
    comparison = compare_experiments(all_results, primary_metric="auroc")

    # Save results
    output_dir = Path("reports/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(all_results, output_dir / "benchmark_results.json")
    save_json(comparison, output_dir / "comparison.json")

    logger.info(f"Benchmark results saved to {output_dir}")
    logger.info(f"Best model: {comparison['best']['name'] if comparison['best'] else 'N/A'}")

    # Print summary
    print("\n=== Benchmark Results ===")
    for r in comparison["ranking"]:
        print(f"  #{r['rank']} {r['name']}: AUROC={r['auroc']:.4f}")


if __name__ == "__main__":
    main()
