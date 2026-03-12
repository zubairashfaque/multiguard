"""Prediction endpoint for multimodal content classification."""

from fastapi import APIRouter, UploadFile

from src.serving.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(text: str, image: UploadFile | None = None) -> PredictionResponse:
    """Classify multimodal content as safe/unsafe.

    Args:
        text: Text content to analyze.
        image: Optional image file.

    Returns:
        Classification result with confidence scores.
    """
    raise NotImplementedError("Implement after model loading pipeline")
