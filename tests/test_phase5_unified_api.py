"""
Tests for Phase 5: Unified ChilliGuard Application Integration.

Covers:
  - ModelService lifecycle, startup loading, and status properties across all 3 modes
  - Binary image prediction (valid & corrupted image byte streams)
  - 19-class image prediction (valid image & top-5 predictions structure)
  - ATR-FTIR spectral prediction (valid CSV/Excel/NPY & malformed/non-numeric input)
  - Preprocessing consistency and transform reproducibility
  - FastAPI endpoints (GET /api/health, GET /api/model/status, POST /api/predict)
  - Error validation (empty files, invalid mode, corrupt files, unsupported extensions)
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from backend.main import app

from backend.app.services.model_service import model_service
from ml.utils.ftir_transforms import SNVTransformer, SavgolSNVTransformer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """FastAPI test client instance."""
    # Ensure models are loaded
    model_service.try_load()
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def valid_image_bytes():
    """Generate a valid RGB JPEG image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(180, 40, 30))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def corrupt_image_bytes():
    """Generate invalid non-image header bytes."""
    return b"CORRUPTED_NON_IMAGE_HEADER_BYTES_1234567890"


@pytest.fixture()
def valid_spectral_csv_bytes():
    """Generate a valid 1610-point spectral CSV in memory."""
    wavenumbers = np.linspace(700.74, 3699.38, 1610)
    intensities = np.sin(wavenumbers / 100.0) * 0.5 + 0.8
    df = pd.DataFrame({"wavenumber": wavenumbers, "intensity": intensities})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def invalid_spectral_text_bytes():
    """Generate a CSV containing non-numeric text data."""
    csv_str = "wavenumber,intensity\nfoo,bar\nhello,world\n"
    return csv_str.encode("utf-8")


@pytest.fixture()
def short_spectral_csv_bytes():
    """Generate a CSV with fewer than 10 data points."""
    csv_str = "wavenumber,intensity\n700,0.5\n800,0.6\n900,0.7\n"
    return csv_str.encode("utf-8")


# ── 1. ModelService Unit Tests ────────────────────────────────────────────────

class TestModelServiceUnit:
    def test_model_service_load(self):
        model_service.try_load()
        assert model_service.is_ready, "ModelService should load at least one trained checkpoint"

    def test_model_service_status_properties(self):
        assert model_service.status in ["ready", "not_trained", "loading_failed"]
        assert model_service.model_name is not None
        assert model_service.checkpoint_path is not None
        assert model_service.num_classes is not None
        assert isinstance(model_service.status_details, dict)
        assert "binary" in model_service.status_details
        assert "multiclass" in model_service.status_details
        assert "ftir" in model_service.status_details

    def test_predict_binary_valid(self, valid_image_bytes):
        res = model_service.predict_binary(valid_image_bytes)
        assert res["status"] == "prediction_ready"
        assert res["mode"] == "binary"
        assert res["predicted_class"] in ["Pure", "Adulterated"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert "Pure" in res["class_probabilities"]
        assert "Adulterated" in res["class_probabilities"]

    def test_predict_binary_corrupted(self, corrupt_image_bytes):
        res = model_service.predict_binary(corrupt_image_bytes)
        assert res["status"] == "error"
        assert "Invalid image file" in res["message"]

    def test_predict_multiclass_valid(self, valid_image_bytes):
        res = model_service.predict_multiclass(valid_image_bytes)
        assert res["status"] == "prediction_ready"
        assert res["mode"] == "multiclass"
        assert res["predicted_class"] is not None
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top5_predictions"]) == 5

    def test_predict_multiclass_corrupted(self, corrupt_image_bytes):
        res = model_service.predict_multiclass(corrupt_image_bytes)
        assert res["status"] == "error"
        assert "Invalid image file" in res["message"]

    def test_predict_ftir_valid(self, valid_spectral_csv_bytes):
        res = model_service.predict_ftir(valid_spectral_csv_bytes, "sample_spectrum.csv")
        assert res["status"] == "prediction_ready"
        assert res["mode"] == "ftir"
        assert res["predicted_class"] in ["Pure", "Adulterated"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert "Pure" in res["class_probabilities"]
        assert "Adulterated" in res["class_probabilities"]

    def test_predict_ftir_non_numeric(self, invalid_spectral_text_bytes):
        res = model_service.predict_ftir(invalid_spectral_text_bytes, "text_data.csv")
        assert res["status"] == "error"
        assert "Invalid spectral content" in res["message"] or "non-numeric" in res["message"]

    def test_predict_ftir_unsupported_extension(self, valid_spectral_csv_bytes):
        res = model_service.predict_ftir(valid_spectral_csv_bytes, "sample.txt")
        assert res["status"] == "error"
        assert "Unsupported file extension" in res["message"]

    def test_predict_ftir_insufficient_points(self, short_spectral_csv_bytes):
        res = model_service.predict_ftir(short_spectral_csv_bytes, "short.csv")
        assert res["status"] == "error"
        assert "insufficient numeric data points" in res["message"]


# ── 2. Preprocessing Consistency Tests ────────────────────────────────────────

class TestPreprocessingConsistency:
    def test_snv_transform_reproducibility(self):
        raw_spectrum = np.random.RandomState(42).randn(1, 1610) * 0.5 + 2.0
        snv = SNVTransformer()
        out1 = snv.fit_transform(raw_spectrum)
        out2 = snv.transform(raw_spectrum)
        assert np.allclose(out1, out2), "SNV Transformer must produce deterministic, identical output"

    def test_savgol_snv_transform_reproducibility(self):
        raw_spectrum = np.random.RandomState(42).randn(1, 1610) * 0.5 + 2.0
        savgol_snv = SavgolSNVTransformer(deriv=1)
        out1 = savgol_snv.fit_transform(raw_spectrum)
        out2 = savgol_snv.transform(raw_spectrum)
        assert np.allclose(out1, out2), "SavgolSNVTransformer must produce deterministic, identical output"


# ── 3. FastAPI Endpoint Tests ─────────────────────────────────────────────────

class TestFastAPIEndpoints:
    def test_health_endpoint(self, api_client):
        res = api_client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"

    def test_model_status_endpoint(self, api_client):
        res = api_client.get("/api/model/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["ready", "not_trained", "loading_failed"]
        assert "model_name" in data
        assert "modes" in data

    def test_predict_endpoint_binary(self, api_client, valid_image_bytes):
        files = {"file": ("test_sample.jpg", valid_image_bytes, "image/jpeg")}
        data = {"mode": "binary"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "prediction_ready"
        assert body["mode"] == "binary"
        assert body["predicted_class"] in ["Pure", "Adulterated"]
        assert body["confidence"] is not None

    def test_predict_endpoint_multiclass(self, api_client, valid_image_bytes):
        files = {"file": ("test_sample.jpg", valid_image_bytes, "image/jpeg")}
        data = {"mode": "multiclass"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "prediction_ready"
        assert body["mode"] == "multiclass"
        assert body["predicted_class"] is not None
        assert body["top5_predictions"] is not None
        assert len(body["top5_predictions"]) == 5

    def test_predict_endpoint_ftir(self, api_client, valid_spectral_csv_bytes):
        files = {"file": ("test_spectrum.csv", valid_spectral_csv_bytes, "text/csv")}
        data = {"mode": "ftir"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "prediction_ready"
        assert body["mode"] == "ftir"
        assert body["predicted_class"] in ["Pure", "Adulterated"]

    def test_predict_invalid_mode_returns_400(self, api_client, valid_image_bytes):
        files = {"file": ("sample.jpg", valid_image_bytes, "image/jpeg")}
        data = {"mode": "unknown_mode"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 400
        assert "Unsupported prediction mode" in res.json()["detail"]

    def test_predict_empty_file_returns_400(self, api_client):
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        data = {"mode": "binary"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()

    def test_predict_corrupt_image_returns_400(self, api_client, corrupt_image_bytes):
        files = {"file": ("corrupt.jpg", corrupt_image_bytes, "image/jpeg")}
        data = {"mode": "binary"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 400
        assert "invalid image file" in res.json()["detail"].lower()

    def test_predict_corrupt_ftir_returns_400(self, api_client, invalid_spectral_text_bytes):
        files = {"file": ("text_data.csv", invalid_spectral_text_bytes, "text/csv")}
        data = {"mode": "ftir"}
        res = api_client.post("/api/predict", files=files, data=data)
        assert res.status_code == 400
        assert "invalid spectral content" in res.json()["detail"].lower() or "non-numeric" in res.json()["detail"].lower()
