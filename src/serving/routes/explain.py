"""Explainability endpoint for prediction explanations."""

import io

import torch
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from src.serving.schemas import ExplanationResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/explain", response_model=ExplanationResponse)
async def explain(
    text: str = Form(...),
    image: UploadFile | None = File(None),
) -> ExplanationResponse:
    """Generate explainability output for a prediction.

    Returns prediction with token-level attribution scores computed
    via input gradient analysis.

    Args:
        text: Text content.
        image: Optional image file.

    Returns:
        Explanation with attribution scores.
    """
    from src.serving.routes.predict import _get_model, _get_tokenizer, _get_transforms

    model = _get_model()
    tokenizer = _get_tokenizer()
    transforms = _get_transforms()
    device = next(model.parameters()).device

    # Tokenize text
    encoding = tokenizer(
        text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    # Process image
    if image is not None:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = transforms(pil_image).unsqueeze(0).to(device)
    else:
        pixel_values = torch.zeros(1, 3, 224, 224, device=device)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get prediction
    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    with torch.no_grad():
        outputs = model(batch)

    probs = torch.softmax(outputs["logits"], dim=-1)[0].cpu()
    label_names = ["safe", "unsafe"]
    predicted_idx = int(probs.argmax().item())

    # Compute simple token attributions via embedding gradient
    text_attributions = None
    try:
        model.eval()
        embeddings = model.text_backbone.encoder.embeddings(input_ids)
        embeddings.retain_grad()

        # Forward through text backbone manually
        text_features = model.text_backbone.encoder(inputs_embeds=embeddings).last_hidden_state[
            :, 0
        ]
        if model.text_proj is not None:
            text_features = model.text_proj(text_features)

        vision_features = model.vision_backbone(pixel_values)
        if model.vision_proj is not None:
            vision_features = model.vision_proj(vision_features)

        fused = model.fusion(vision_features, text_features)
        logits = model.head(fused)
        logits[0, predicted_idx].backward()

        if embeddings.grad is not None:
            token_importance = embeddings.grad.norm(dim=-1)[0].cpu().tolist()
            active_tokens = [
                {"token": tok, "score": round(score, 4)}
                for tok, score, mask in zip(
                    tokens, token_importance, attention_mask[0].cpu().tolist()
                )
                if mask == 1 and tok not in ("<s>", "</s>", "<pad>")
            ]
            text_attributions = active_tokens
    except Exception as e:
        logger.warning(f"Attribution computation failed: {e}")

    model.zero_grad()

    return ExplanationResponse(
        label=label_names[predicted_idx],
        confidence=float(probs[predicted_idx]),
        text_attributions=text_attributions,
        image_heatmap_url=None,
    )
