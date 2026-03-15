#!/usr/bin/env python3
"""Evaluation entry point for MultiGuard.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml
    python scripts/evaluate.py --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml --split test
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import ensure_dir, save_json  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard evaluation")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--split", default="val", choices=["val", "dev", "test"], help="Evaluation split"
    )
    parser.add_argument(
        "--output-dir", default="reports/evaluation", help="Output directory for results"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    set_seed(config.get("project", {}).get("seed", 42))

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")

    import torch
    from transformers import AutoTokenizer

    from src.data.augmentation import build_image_transforms
    from src.data.loaders import HatefulMemesDataset, build_dataloader
    from src.evaluation.evaluator import Evaluator
    from src.serving.model_loader import load_model

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build evaluation dataset
    data_cfg = config.data
    model_cfg = config.model
    data_dir = Path(config.paths.data_dir) / "raw" / "hateful_memes"

    val_transforms = build_image_transforms(
        {"image_size": data_cfg.get("image_size", 224)}, split="val"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.text_backbone)

    dataset = HatefulMemesDataset(
        data_dir=data_dir,
        split=args.split,
        image_transform=val_transforms,
        tokenizer=tokenizer,
        max_text_length=data_cfg.get("max_text_length", 77),
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=config.training.get("batch_size", 8),
        shuffle=False,
        num_workers=config.device.get("num_workers", 4),
    )

    # Run evaluation
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate(dataloader)

    # Save results
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "metrics": metrics,
        "config": str(args.config),
    }

    results_path = output_dir / f"eval_{args.split}_{Path(args.checkpoint).stem}.json"
    save_json(results, results_path)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
