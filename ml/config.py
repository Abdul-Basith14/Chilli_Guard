"""
ChilliGuard ML — Central configuration.

Dataset paths and hyperparameters are read from environment variables
(via the .env file at the project root) so they never need to be hardcoded.

Usage:
    from ml.config import cfg
    print(cfg.dataset_i_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels above this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass
class MLConfig:
    """All configurable parameters for the ML pipeline."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data"))
    )
    dataset_i_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATASET_I_PATH", "./data/dataset_I"))
    )
    dataset_ii_path: Path = field(
        default_factory=lambda: (
            Path(os.getenv("DATASET_II_PATH", "./data/dataset_II"))
            if (Path(os.getenv("DATASET_II_PATH", "./data/dataset_II")).exists() and
                any(d.is_dir() for d in Path(os.getenv("DATASET_II_PATH", "./data/dataset_II")).iterdir()))
            else Path(r"c:\01.Myuse\Major_Project\dataset\images\MultiClass")
        )
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", "./ml/models"))
    )
    reports_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "reports"
    )
    experiments_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "experiments"
    )
    split_manifest_path: Path = field(
        default_factory=lambda: Path(os.getenv("SPLIT_MANIFEST", "./data/dataset_I_split.csv"))
    )

    # ── Dataset I — Binary classification ────────────────────────────────────
    # Class folder names as they appear in the filesystem
    binary_classes: list[str] = field(default_factory=lambda: ["C1_PKT", "C2_AKT"])
    # Human-readable labels (aligned with binary_classes order)
    binary_labels: list[str] = field(default_factory=lambda: ["Pure", "Adulterated"])
    # Folder → label mapping
    binary_class_to_label: dict[str, str] = field(
        default_factory=lambda: {"C1_PKT": "Pure", "C2_AKT": "Adulterated"}
    )

    # ── Dataset II — 19-class (future) ────────────────────────────────────────
    # Class names will be read directly from directory names during Phase 3.
    # Do NOT manually list them here until the dataset is inspected.
    num_multiclasses: int = 19  # expected, to be verified in Phase 1

    # ── Image preprocessing ───────────────────────────────────────────────────
    image_size: tuple[int, int] = (224, 224)  # (height, width)
    image_channels: int = 3                   # RGB

    # ── Training hyperparameters ──────────────────────────────────────────────
    batch_size: int = 16           # CPU-friendly default
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_val_test_split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42

    # ── Phase 2 — Validation split from training set only ────────────────────
    phase2_val_split_ratio: float = 0.15  # fraction of train set → val

    # ── Model architecture ────────────────────────────────────────────────────
    # Phase 2: lightweight pretrained ResNet18 with frozen backbone
    model_arch: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True   # Phase 2: only train the classifier head
    num_classes: int = 2           # Pure=0, Adulterated=1

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"  # "auto" → CUDA if available else CPU

    def __post_init__(self) -> None:
        """Resolve relative paths against the project root."""
        for attr in ("data_root", "dataset_i_path", "dataset_ii_path",
                     "model_dir", "split_manifest_path"):
            p = getattr(self, attr)
            if not p.is_absolute():
                setattr(self, attr, (_PROJECT_ROOT / p).resolve())

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "figures").mkdir(exist_ok=True)
        (self.reports_dir / "metrics").mkdir(exist_ok=True)


# Module-level singleton
cfg = MLConfig()
