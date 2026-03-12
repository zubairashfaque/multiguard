"""FastAPI application entry point."""

from fastapi import FastAPI

from src.serving.routes import embed, explain, health, predict

app = FastAPI(
    title="MultiGuard API",
    description="Multimodal Content Intelligence Pipeline",
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(embed.router, prefix="/api/v1", tags=["embed"])
app.include_router(explain.router, prefix="/api/v1", tags=["explain"])
