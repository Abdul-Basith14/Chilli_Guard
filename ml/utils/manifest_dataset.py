"""
ChilliGuard ML -- Manifest-based PyTorch Dataset (Phase 2 & 3).

Loads images from an explicit CSV manifest rather than scanning folders.
Supports in-memory tensor/PIL caching for ultra-fast CPU training loops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("chilliguard.manifest_dataset")


class ManifestDataset(Dataset):
    """
    Loads images referenced in a CSV manifest for a single split.

    Args:
        manifest_df:     DataFrame with columns [path, split, label_idx, label/class_name].
        split:           Which split to use: "train", "val", or "test".
        transform:       Torchvision transform pipeline.
        verify_paths:    If True, raise FileNotFoundError for missing images.
        cache_in_memory: If True, preload and cache RGB PIL images in RAM for fast CPU loops.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        transform: Optional[Callable] = None,
        verify_paths: bool = True,
        cache_in_memory: bool = False,
    ) -> None:
        self.split = split
        self.transform = transform

        sub = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            raise ValueError(f"No rows in manifest for split='{split}'")

        label_col = "label" if "label" in sub.columns else "class_name"

        self.paths: list[Path] = [Path(p) for p in sub["path"]]
        self.labels: list[int] = sub["label_idx"].astype(int).tolist()
        self.label_names: list[str] = sub[label_col].tolist()
        self.class_names: list[str] = sorted(sub[label_col].unique().tolist())

        if verify_paths:
            missing = [p for p in self.paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} image(s) in split '{split}' not found. "
                    f"First missing: {missing[0]}"
                )

        self._cache: dict[int, Image.Image] = {}
        if cache_in_memory:
            logger.info(f"Preloading {len(self.paths)} images for split '{split}' into RAM...")
            for idx, p in enumerate(self.paths):
                img = Image.open(p).convert("RGB")
                img.load()  # Ensure image data is loaded into RAM
                self._cache[idx] = img

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            img = self._cache[idx].copy()
        else:
            img = Image.open(self.paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.label_names:
            counts[name] = counts.get(name, 0) + 1
        return counts
