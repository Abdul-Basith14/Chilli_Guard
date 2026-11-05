"""
FastAPI router for dataset inspection, reports, and visualization artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.dataset import (
    DatasetStatusResponse,
    DatasetInspectResponse,
    DatasetReportResponse,
)
from app.services.dataset_service import dataset_service
from ml.config import cfg

router = APIRouter(prefix="/dataset", tags=["Dataset Inspection"])


@router.get(
    "/status",
    response_model=DatasetStatusResponse,
    summary="Get dataset availability and inspection status",
)
async def get_dataset_status() -> DatasetStatusResponse:
    """Return dataset path validation, folder structure type, and report status."""
    return DatasetStatusResponse(**dataset_service.get_status())


@router.post(
    "/inspect",
    response_model=DatasetInspectResponse,
    summary="Run dataset inspection pipeline",
)
async def run_dataset_inspection() -> DatasetInspectResponse:
    """
    Trigger full exploratory data analysis (EDA) on Dataset I.
    Generates CSV inventories, JSON summary, visualizations, and Markdown report.
    """
    if not cfg.dataset_i_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset path does not exist: {cfg.dataset_i_path}",
        )

    try:
        summary = dataset_service.run_inspection()
        return DatasetInspectResponse(
            status="success",
            message="Dataset inspection completed successfully.",
            summary=summary,
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset inspection failed: {str(err)}",
        )


@router.get(
    "/report",
    response_model=DatasetReportResponse,
    summary="Get executive dataset inspection report",
)
async def get_dataset_report() -> DatasetReportResponse:
    """Return the cached summary JSON and Markdown report content."""
    report_data = dataset_service.get_full_report()
    return DatasetReportResponse(**report_data)


@router.get(
    "/charts/{chart_name}",
    summary="Fetch dataset visualization chart image",
)
async def get_dataset_chart(chart_name: str):
    """
    Serve generated chart PNG files from reports/figures directory.
    Allowed filenames: class_distribution.png, sample_images.png
    """
    allowed_charts = {"class_distribution.png", "sample_images.png"}
    if chart_name not in allowed_charts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chart name requested. Allowed: {list(allowed_charts)}",
        )

    chart_file = cfg.reports_dir / "figures" / chart_name
    if not chart_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chart figure '{chart_name}' not generated yet. Please run dataset inspection first.",
        )

    return FileResponse(chart_file, media_type="image/png")
