#!/usr/bin/env python3
"""Model export script (ONNX, TorchScript).

Usage:
    python scripts/export_model.py --format onnx --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml
    python scripts/export_model.py --format torchscript --checkpoint models/checkpoints/baseline/best_model.pt --config configs/train/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from src.serving.model_loader import load_model  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    """Export model to ONNX format."""
    model.eval()
    device = next(model.parameters()).device

    dummy_batch = {
        "pixel_values": torch.randn(1, 3, 224, 224, device=device),
        "input_ids": torch.randint(0, 30000, (1, 77), device=device),
        "attention_mask": torch.ones(1, 77, dtype=torch.long, device=device),
    }

    class WrapperModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values, input_ids, attention_mask):
            batch = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            outputs = self.model(batch)
            return outputs["logits"]

    wrapper = WrapperModel(model)

    torch.onnx.export(
        wrapper,
        (dummy_batch["pixel_values"], dummy_batch["input_ids"], dummy_batch["attention_mask"]),
        str(output_path),
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )
    logger.info(f"ONNX model exported to {output_path}")


def export_torchscript(model: torch.nn.Module, output_path: Path) -> None:
    """Export model to TorchScript format via tracing."""
    model.eval()
    device = next(model.parameters()).device

    dummy_batch = {
        "pixel_values": torch.randn(1, 3, 224, 224, device=device),
        "input_ids": torch.randint(0, 30000, (1, 77), device=device),
        "attention_mask": torch.ones(1, 77, dtype=torch.long, device=device),
    }

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            example_kwarg_inputs=dummy_batch,
            strict=False,
        )

    traced.save(str(output_path))
    logger.info(f"TorchScript model exported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiGuard model export")
    parser.add_argument(
        "--format",
        required=True,
        choices=["onnx", "torchscript"],
        help="Export format",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/baseline/best_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--config", default="configs/train/baseline.yaml", help="Training config path"
    )
    parser.add_argument("--output-dir", default="models/exported", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info(f"Exporting model to {args.format}: {args.checkpoint}")

    from src.utils.config import load_config

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.format == "onnx":
        output_path = output_dir / "multiguard.onnx"
        export_onnx(model, output_path)
    elif args.format == "torchscript":
        output_path = output_dir / "multiguard.pt"
        export_torchscript(model, output_path)

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
