#!/usr/bin/env python3
"""Data ingestion script for Hateful Memes and other datasets.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --dataset hateful_memes
    python scripts/ingest_data.py --dataset hateful_memes --raw-dir data/raw/hateful_memes
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def ingest_hateful_memes(raw_dir: Path, output_dir: Path) -> None:
    """Process Hateful Memes dataset from raw JSONL + images into train/val/test splits.

    Expected raw structure:
        raw_dir/
        ├── train.jsonl
        ├── dev.jsonl
        ├── test.jsonl
        └── img/

    Output structure:
        output_dir/
        ├── train/
        │   ├── annotations.jsonl
        │   └── img/  (symlinked)
        ├── val/
        │   ├── annotations.jsonl
        │   └── img/  (symlinked)
        └── test/
            ├── annotations.jsonl
            └── img/  (symlinked)
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Place the Hateful Memes dataset in data/raw/hateful_memes/\n"
            "Expected files: train.jsonl, dev.jsonl, test.jsonl, img/"
        )

    split_mapping = {
        "train": "train.jsonl",
        "val": "dev.jsonl",
        "test": "test.jsonl",
    }

    img_dir = raw_dir / "img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    stats = {}

    for split_name, jsonl_file in split_mapping.items():
        jsonl_path = raw_dir / jsonl_file
        if not jsonl_path.exists():
            logger.warning(f"Skipping {split_name}: {jsonl_path} not found")
            continue

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Read and validate annotations
        samples = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                img_path = raw_dir / sample["img"]
                if img_path.exists():
                    samples.append(sample)
                else:
                    skipped += 1

        # Write processed annotations
        out_jsonl = split_dir / "annotations.jsonl"
        with open(out_jsonl, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Symlink image directory to avoid copying
        img_link = split_dir / "img"
        if img_link.exists() or img_link.is_symlink():
            img_link.unlink() if img_link.is_symlink() else shutil.rmtree(img_link)
        img_link.symlink_to(img_dir.resolve())

        label_dist = {}
        for s in samples:
            label = s.get("label", "unlabeled")
            label_dist[label] = label_dist.get(label, 0) + 1

        stats[split_name] = {
            "total": len(samples),
            "skipped": skipped,
            "label_distribution": label_dist,
        }

        logger.info(
            f"  {split_name}: {len(samples)} samples "
            f"(skipped {skipped} missing images), labels={label_dist}"
        )

    # Save dataset stats
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats saved to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard data ingestion")
    parser.add_argument(
        "--dataset",
        default="hateful_memes",
        choices=["hateful_memes"],
        help="Dataset to ingest",
    )
    parser.add_argument("--raw-dir", default=None, help="Path to raw dataset directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info(f"Ingesting dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "hateful_memes":
        raw_dir = Path(args.raw_dir) if args.raw_dir else Path("data/raw/hateful_memes")
        ingest_hateful_memes(raw_dir, output_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
