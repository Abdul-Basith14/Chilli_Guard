"""
Pydantic response schemas for health-check endpoints.
"""

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""

    status: str
    """'ok' when the server is running normally."""

    environment: str
    """The current APP_ENV value (e.g. 'development')."""

    timestamp: datetime
    """UTC timestamp of when the response was generated."""

    version: str
    """API version string."""
