"""
Router: GET /api/health
Returns server liveness status.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns server status, environment, and timestamp. Use this to verify the API is reachable.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        timestamp=datetime.now(tz=timezone.utc),
        version="0.1.0",
    )
