"""
Router: GET /api/model/status & GET /api/model/metrics
Reports model status and performance metrics from Phase 2.
"""

import json
from pathlib import Path

from fastapi import APIRouter
from app.config import settings
from app.schemas.model import ModelStatusResponse
from app.services.model_service import model_service

router = APIRouter(tags=["Model"])


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Model Status",
    description="Returns the current load status of the classification model.",
)
async def get_model_status() -> ModelStatusResponse:
    # Try reloading model if not loaded yet
    if not model_service.is_ready:
        model_service.try_load()

    return ModelStatusResponse(
        status=model_service.status,
        model_name=model_service.model_name,
        checkpoint_path=model_service.checkpoint_path,
        num_classes=model_service.num_classes,
        modes=model_service.status_details,
        message=model_service.status_message,
    )



@router.get(
    "/model/metrics",
    summary="Model Performance Metrics",
    description="Returns Phase 2 experiment results (Exp A and B) and leakage audit summary.",
)
async def get_model_metrics() -> dict:
    exp_dir = settings.project_root / "experiments"
    exp_a_path = exp_dir / "phase2_exp_A.json"
    exp_b_path = exp_dir / "phase2_exp_B.json"
    audit_path = exp_dir / "phase2_leakage_audit.json"
    test_path = exp_dir / "phase2_exp_B_test_results.json"

    res = {
        "status": "ready" if (exp_a_path.exists() or exp_b_path.exists()) else "not_trained",
        "exp_a": json.loads(exp_a_path.read_text()) if exp_a_path.exists() else None,
        "exp_b": json.loads(exp_b_path.read_text()) if exp_b_path.exists() else None,
        "audit": json.loads(audit_path.read_text()) if audit_path.exists() else None,
        "test_results": json.loads(test_path.read_text()) if test_path.exists() else None,
    }
    return res
