"""
PyTorch Dataset classes for ChilliGuard.

ChilliDataset:  Loads Dataset I (binary) or Dataset II (multi-class)
                from an ImageFolder-style directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class ChilliDataset(Dataset):
    """
    Generic image dataset that loads from a root directory where each
    subdirectory is a class label.

    Directory layout expected:
        root/
            ClassA/img001.jpg
            ClassA/img002.png
            ClassB/img001.jpg
            ...

    Args:
        root:        Path to the dataset root directory.
        transform:   Torchvision transform pipeline to apply to each image.
        class_map:   Optional dict mapping folder names to integer labels.
                     If None, labels are assigned alphabetically (0, 1, ...).
        extensions:  Set of valid image file extensions (including the dot).
    """

    DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable] = None,
        class_map: Optional[dict[str, int]] = None,
        extensions: Optional[set[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.DEFAULT_EXTENSIONS

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Discover class directories
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"No class subdirectories found in: {self.root}")

        # Build class → index mapping
        if class_map is not None:
            self.class_to_idx = class_map
        else:
            self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        self.classes: list[str] = list(self.class_to_idx.keys())

        # Collect all (image_path, label_idx) pairs
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label_name = class_dir.name
            if label_name not in self.class_to_idx:
                continue
            label_idx = self.class_to_idx[label_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, label_idx))

        if not self.samples:
            raise ValueError(
                f"No images found in: {self.root}. "
                f"Check that images have one of these extensions: {self.extensions}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def class_counts(self) -> dict[str, int]:
        """Return per-class image counts (useful for class-weight computation)."""
        counts: dict[str, int] = {c: 0 for c in self.classes}
        for _, label_idx in self.samples:
            class_name = self.classes[label_idx]
            counts[class_name] += 1
        return counts
