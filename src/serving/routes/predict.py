"""Prediction endpoint for multimodal content classification."""

import io

import torch
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from src.data.augmentation import build_image_transforms
from src.serving.schemas import PredictionResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Lazy-loaded at first request
_model = None
_tokenizer = None
_transforms = None


def _get_model():
    """Get or load the serving model."""
    global _model
    if _model is None:
        from src.serving.model_loader import load_model

        _model = load_model(
            checkpoint_path="models/checkpoints/baseline/best_model.pt",
            config_path="configs/train/baseline.yaml",
        )
    return _model


def _get_tokenizer():
    """Get or load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return _tokenizer


def _get_transforms():
    """Get or load image transforms."""
    global _transforms
    if _transforms is None:
        _transforms = build_image_transforms({"image_size": 224}, split="val")
    return _transforms


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    text: str = Form(...),
    image: UploadFile | None = File(None),
) -> PredictionResponse:
    """Classify multimodal content as safe/unsafe.

    Args:
        text: Text content to analyze.
        image: Optional image file.

    Returns:
        Classification result with confidence scores.
    """
    model = _get_model()
    tokenizer = _get_tokenizer()
    transforms = _get_transforms()
    device = next(model.parameters()).device

    # Tokenize text
    encoding = tokenizer(
        text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )

    # Process image
    if image is not None:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = transforms(pil_image).unsqueeze(0)
    else:
        pixel_values = torch.zeros(1, 3, 224, 224)

    batch = {
        "pixel_values": pixel_values.to(device),
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }

    with torch.no_grad():
        outputs = model(batch)

    probs = torch.softmax(outputs["logits"], dim=-1)[0].cpu().tolist()
    label_names = ["safe", "unsafe"]
    predicted_idx = int(torch.argmax(outputs["logits"], dim=-1).item())

    return PredictionResponse(
        label=label_names[predicted_idx],
        confidence=probs[predicted_idx],
        probabilities={name: prob for name, prob in zip(label_names, probs)},
    )
