"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    text: str = Field(..., description="Text content to analyze")
    image_url: str | None = Field(None, description="URL to image")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    label: str = Field(..., description="Predicted label (safe/unsafe)")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(..., description="Per-class probabilities")


class EmbeddingResponse(BaseModel):
    """Response schema for embedding endpoint."""

    embedding: list[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class ExplanationResponse(BaseModel):
    """Response schema for explainability endpoint."""

    label: str = Field(..., description="Predicted label")
    confidence: float = Field(..., description="Prediction confidence")
    text_attributions: list[dict[str, float]] | None = Field(
        None, description="Token-level attribution scores"
    )
    image_heatmap_url: str | None = Field(None, description="URL to GradCAM heatmap image")
