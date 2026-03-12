"""FastAPI middleware for rate limiting, logging, and metrics."""

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing information."""

    async def dispatch(self, request: Request, call_next: callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)"
        )
        return response
