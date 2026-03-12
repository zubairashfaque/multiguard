"""Embedding endpoint for multimodal retrieval."""

from fastapi import APIRouter, UploadFile

from src.serving.schemas import EmbeddingResponse

router = APIRouter()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed(text: str | None = None, image: UploadFile | None = None) -> EmbeddingResponse:
    """Generate embeddings for text and/or image content.

    Args:
        text: Optional text to embed.
        image: Optional image to embed.

    Returns:
        Embedding vector(s).
    """
    raise NotImplementedError("Implement after model loading pipeline")
