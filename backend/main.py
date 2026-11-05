"""
ChilliGuard Backend — FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import sys
from pathlib import Path

# Ensure backend directory and project root are in sys.path for uvicorn execution
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import dataset, health, model, predict
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model checkpoints on server startup
    model_service.try_load()
    yield


# ── Application instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="ChilliGuard API",
    description=(
        "REST API for the ChilliGuard red chilli powder adulteration detection system. "
        "Provides health checks, model status, dataset inspection, and classification endpoints."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(predict.router, prefix="/api")


# ── Root redirect to docs ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "ChilliGuard API is running.",
        "docs": "/docs",
        "health": "/api/health",
    }
