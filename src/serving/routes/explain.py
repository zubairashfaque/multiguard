"""Explainability endpoint for prediction explanations."""

from fastapi import APIRouter, UploadFile

from src.serving.schemas import ExplanationResponse

router = APIRouter()


@router.post("/explain", response_model=ExplanationResponse)
async def explain(text: str, image: UploadFile | None = None) -> ExplanationResponse:
    """Generate explainability output for a prediction.

    Returns GradCAM heatmap and token attributions.

    Args:
        text: Text content.
        image: Optional image file.

    Returns:
        Explanation with attribution maps.
    """
    raise NotImplementedError("Implement after explainability pipeline")
