"""
Pydantic request/response schemas for the prediction endpoint.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Top5Prediction(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response body for POST /api/predict."""

    status: Literal["prediction_ready", "model_not_ready", "error"]
    """
    - 'prediction_ready' — A model is loaded and a prediction was produced.
    - 'model_not_ready'  — No trained model is available; no prediction made.
    - 'error'            — An unexpected error occurred during inference.
    """

    mode: Optional[Literal["binary", "multiclass", "ftir"]] = Field(
        default="binary",
        description="Prediction mode ('binary', 'multiclass', or 'ftir').",
    )

    # ── Fields populated when status == 'prediction_ready' ──────────────
    predicted_class: Optional[str] = Field(
        default=None,
        description="Predicted class label (e.g. 'Pure', 'Adulterated', or 'KTGM_15').",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence score in range [0, 1].",
    )
    class_probabilities: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-class softmax or classifier probabilities.",
    )
    top5_predictions: Optional[list[Top5Prediction]] = Field(
        default=None,
        description="Top-5 predicted classes and probabilities (multiclass mode only).",
    )
    estimated_concentration_percent: Optional[float] = Field(
        default=None,
        description="Estimated adulterant concentration (% w/w Sudan IV) for FTIR mode.",
    )

    # ── Always present ────────────────────────────────────────────────────────
    message: str = Field(
        description="Human-readable explanation of the result or reason no prediction was made."
    )
    filename: Optional[str] = Field(
        default=None,
        description="Original filename of the uploaded sample.",
    )
    file_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of the uploaded sample in bytes.",
    )


class PredictionError(BaseModel):
    """Error response for validation failures on the predict endpoint."""

    detail: str
    field: Optional[str] = None
