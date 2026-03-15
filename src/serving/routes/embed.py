"""Embedding endpoint for multimodal retrieval."""

import io

import torch
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from src.serving.schemas import EmbeddingResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed(
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> EmbeddingResponse:
    """Generate fused embeddings for text and/or image content.

    Args:
        text: Optional text to embed.
        image: Optional image to embed.

    Returns:
        Fused embedding vector.
    """
    from src.serving.routes.predict import _get_model, _get_tokenizer, _get_transforms

    if text is None and image is None:
        raise ValueError("At least one of text or image must be provided")

    model = _get_model()
    tokenizer = _get_tokenizer()
    transforms = _get_transforms()
    device = next(model.parameters()).device

    # Tokenize text
    if text is not None:
        encoding = tokenizer(
            text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
    else:
        input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
        attention_mask = torch.zeros(1, 77, dtype=torch.long, device=device)

    # Process image
    if image is not None:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = transforms(pil_image).unsqueeze(0).to(device)
    else:
        pixel_values = torch.zeros(1, 3, 224, 224, device=device)

    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    with torch.no_grad():
        outputs = model(batch)

    embedding = outputs["fused_features"][0].cpu().tolist()

    return EmbeddingResponse(
        embedding=embedding,
        dimension=len(embedding),
    )
