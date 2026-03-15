#!/usr/bin/env python3
"""Build feature store by pre-computing embeddings for all dataset samples.

Pre-computes and saves vision and text embeddings for fast retrieval
during inference and analysis.

Usage:
    python scripts/build_feature_store.py
    python scripts/build_feature_store.py --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.data.augmentation import build_image_transforms  # noqa: E402
from src.data.loaders import HatefulMemesDataset, build_dataloader  # noqa: E402
from src.serving.model_loader import load_model  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


@torch.no_grad()
def build_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract fused features for all samples.

    Returns:
        Dict with 'embeddings' [N, D], 'labels' [N], 'ids' [N].
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_ids = []

    for batch in dataloader:
        batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch_device)
        all_embeddings.append(outputs["fused_features"].cpu())
        if "labels" in batch:
            all_labels.append(batch["labels"])
        if "id" in batch:
            all_ids.extend(batch["id"] if isinstance(batch["id"], list) else batch["id"].tolist())

    result = {"embeddings": torch.cat(all_embeddings)}
    if all_labels:
        result["labels"] = torch.cat(all_labels)
    if all_ids:
        result["ids"] = all_ids

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard feature store builder")
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/baseline/best_model.pt",
        help="Model checkpoint",
    )
    parser.add_argument("--config", default="configs/train/baseline.yaml", help="Training config")
    parser.add_argument("--output-dir", default="data/features", help="Feature output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info("Building feature store...")

    from src.utils.config import load_config

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        device=str(device),
    )

    data_cfg = config.data
    model_cfg = config.model
    data_dir = Path(config.paths.data_dir) / "raw" / "hateful_memes"

    transforms = build_image_transforms(
        {"image_size": data_cfg.get("image_size", 224)}, split="val"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.text_backbone)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    for split in args.splits:
        logger.info(f"Processing split: {split}")

        dataset = HatefulMemesDataset(
            data_dir=data_dir,
            split=split,
            image_transform=transforms,
            tokenizer=tokenizer,
            max_text_length=data_cfg.get("max_text_length", 77),
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=config.training.get("batch_size", 8),
            shuffle=False,
            num_workers=config.device.get("num_workers", 4),
        )

        features = build_features(model, dataloader, device)

        # Save features
        split_path = output_dir / f"{split}_features.pt"
        torch.save(features, split_path)
        logger.info(
            f"  Saved {features['embeddings'].shape[0]} embeddings "
            f"(dim={features['embeddings'].shape[1]}) to {split_path}"
        )

    logger.info("Feature store build complete!")


if __name__ == "__main__":
    main()
