"""
Application configuration — reads values from the .env file via pydantic-settings.
All settings have safe defaults so the server can start without a .env file
during development.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object.  Values are loaded (in order of priority):
    1. Actual environment variables
    2. .env file at the project root
    3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        # Look for .env in the project root (two levels above this file)
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    app_env: str = "development"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated origins, e.g. "http://localhost:5173,http://127.0.0.1:5173"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ── Data / Model paths ────────────────────────────────────────────────────
    data_root: str = "./data"
    dataset_i_path: str = "./data/dataset_I"
    dataset_ii_path: str = "./data/dataset_II"
    model_dir: str = "./ml/models"
    binary_model_name: str = "binary_exp_b_best.pth"
    multiclass_model_name: str = "multiclass_exp_c_best.pth"
    ftir_model_name: str = "ftir_classifier_best.joblib"

    # ── Upload constraints ────────────────────────────────────────────────────
    # Maximum upload file size in bytes (default: 10 MB)
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: str = "image/jpeg,image/png,image/webp,image/bmp,image/tiff"
    allowed_spectral_extensions: str = ".csv,.xlsx,.xls"

    # ── Derived helpers (not read from env) ───────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list, stripped of whitespace."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_image_types_set(self) -> set:
        return {t.strip() for t in self.allowed_image_types.split(",") if t.strip()}

    @property
    def allowed_spectral_extensions_set(self) -> set:
        return {ext.strip().lower() for ext in self.allowed_spectral_extensions.split(",") if ext.strip()}

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def binary_model_path(self) -> Path:
        return Path(self.model_dir) / self.binary_model_name


    @property
    def multiclass_model_path(self) -> Path:
        return Path(self.model_dir) / self.multiclass_model_name

    @property
    def ftir_model_path(self) -> Path:
        return Path(self.model_dir) / self.ftir_model_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()
