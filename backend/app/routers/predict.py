"""
Router: Prediction endpoints supporting binary, multiclass, and FTIR classification modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.config import settings
from app.schemas.predict import PredictionResponse
from app.services.model_service import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

_MAX_BYTES = settings.max_upload_bytes



def _validate_file_size(file_bytes: bytes, filename: str) -> None:
    file_size = len(file_bytes)
    max_mb = _MAX_BYTES / (1024 * 1024)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty.",
        )

    if file_size > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size {file_size / (1024 * 1024):.1f} MB exceeds the {max_mb:.0f} MB limit.",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Uploaded Sample (Unified Mode)",
    description=(
        "Upload a sample file and specify the prediction mode ('binary', 'multiclass', or 'ftir').\n"
        "- **binary**: Image input -> Pure vs. Adulterated.\n"
        "- **multiclass**: Image input -> 19-class prediction with top-5 breakdown.\n"
        "- **ftir**: Spectral CSV/Excel input -> Pure vs. Sudan IV Adulterated + concentration estimation."
    ),
    responses={
        400: {"description": "Invalid file type, unsupported mode, or empty/oversized upload."},
    },
)
async def predict_unified(
    file: UploadFile = File(..., description="Sample file (Image or FTIR CSV/Excel)."),
    mode: str = Form("binary", description="Prediction mode ('binary', 'multiclass', or 'ftir')."),
) -> PredictionResponse:
    mode_clean = (mode or "binary").lower().strip()
    filename = file.filename or "uploaded_sample"
    content_type = (file.content_type or "").lower()

    if mode_clean not in ["binary", "multiclass", "ftir"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported prediction mode '{mode}'. Allowed modes: 'binary', 'multiclass', 'ftir'.",
        )

    # ── Read file bytes & validate size ──────────────────────────────────────
    try:
        file_bytes: bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded file.",
        ) from exc

    _validate_file_size(file_bytes, filename)

    # ── Validate file format per mode ────────────────────────────────────────
    ext = Path(filename).suffix.lower()

    if mode_clean in ["binary", "multiclass"]:
        is_allowed_mime = content_type in settings.allowed_image_types_set
        is_allowed_ext = ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
        if not (is_allowed_mime or is_allowed_ext):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported image file '{filename}' (MIME: '{content_type}'). "
                    f"Allowed types: {', '.join(sorted(settings.allowed_image_types_set))}."
                ),
            )
    elif mode_clean == "ftir":
        if ext not in settings.allowed_spectral_extensions_set and ext != ".npy":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported spectral file '{filename}' with extension '{ext}'. "
                    f"Allowed extensions: {', '.join(sorted(settings.allowed_spectral_extensions_set))}, .npy."
                ),
            )

    logger.info("Prediction request: mode='%s', filename='%s', size=%d bytes", mode_clean, filename, len(file_bytes))

    # ── Execute Prediction via ModelService ──────────────────────────────────
    result = model_service.predict(file_bytes, filename, mode=mode_clean)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Inference failed due to invalid input or processing error."),
        )

    return PredictionResponse(
        status=result["status"],
        mode=result.get("mode", mode_clean),
        predicted_class=result.get("predicted_class"),
        confidence=result.get("confidence"),
        class_probabilities=result.get("class_probabilities"),
        top5_predictions=result.get("top5_predictions"),
        estimated_concentration_percent=result.get("estimated_concentration_percent"),
        message=result.get("message", "Prediction complete."),
        filename=filename,
        file_size_bytes=len(file_bytes),
    )


