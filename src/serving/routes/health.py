"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "multiguard"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check — verifies model is loaded."""
    # TODO: Check if model is actually loaded
    return {"status": "ready", "service": "multiguard"}
