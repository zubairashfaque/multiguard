"""Health check endpoint."""

from fastapi import APIRouter

from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "multiguard"}


@router.get("/ready")
async def readiness_check() -> dict[str, str | bool]:
    """Readiness check — verifies model is loaded."""
    try:
        from src.serving.routes.predict import _model

        model_loaded = _model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "ready" if model_loaded else "not_ready",
        "service": "multiguard",
        "model_loaded": model_loaded,
    }
