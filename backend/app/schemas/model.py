"""
Pydantic response schemas for model-status endpoints.
"""

from typing import Any, Literal, Optional


from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    """Response body for GET /api/model/status."""

    status: Literal["ready", "not_trained", "loading_failed"]
    """
    - 'ready'          — Model weights are loaded and inference is available.
    - 'not_trained'    — No model checkpoint found; training has not completed.
    - 'loading_failed' — A checkpoint was found but failed to load.
    """

    model_name: Optional[str] = None
    """Short name of the loaded model architecture, if available."""

    checkpoint_path: Optional[str] = None
    """Filesystem path of the checkpoint that was attempted, if applicable."""

    num_classes: Optional[int] = None
    """Number of output classes the model was trained for."""

    modes: Optional[dict[str, Any]] = None
    """Detailed load status per classification mode ('binary', 'multiclass', 'ftir')."""

    message: str

    """Human-readable explanation of the current status."""
