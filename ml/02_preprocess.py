"""
Script 02: Image Preprocessing & Augmentation Setup

Defines and validates the transform pipelines that will be used during training.
Saves sample augmented images to reports/figures/ for visual inspection.

No model training occurs here.

Usage:
    python ml/02_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.utils.transforms import get_train_transforms, get_eval_transforms


def preview_transforms() -> None:
    """
    Load a few sample images from Dataset I and save augmented versions
    to reports/figures/ so transforms can be visually inspected.
    """
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
    except ImportError:
        print("[ERROR] Pillow and torchvision are required. Run: pip install -r ml/requirements.txt")
        return

    dataset_root = cfg.dataset_i_path
    if not dataset_root.exists():
        print(f"[INFO] Dataset I not found at {dataset_root}. Skipping preview.")
        return

    out_dir = cfg.reports_dir / "figures" / "augmentation_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = get_train_transforms(cfg.image_size)
    eval_transform = get_eval_transforms(cfg.image_size)

    print(f"\nTransform pipelines defined:")
    print(f"  Train: {train_transform}")
    print(f"  Eval : {eval_transform}")

    samples_saved = 0
    for folder_name in cfg.binary_classes:
        class_dir = dataset_root / folder_name
        if not class_dir.exists():
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue

        # Preview the first image from each class
        sample_path = images[0]
        img = Image.open(sample_path).convert("RGB")
        augmented = train_transform(img)

        # Save the PIL version of the augmented tensor for visual check
        aug_pil = TF.to_pil_image(augmented)
        save_path = out_dir / f"{folder_name}_augmented_sample.png"
        aug_pil.save(save_path)
        samples_saved += 1
        print(f"  Saved augmented preview: {save_path}")

    if samples_saved == 0:
        print("[INFO] No sample images found for preview. Run after placing dataset images.")
    else:
        print(f"\n  ✔ {samples_saved} augmented previews saved to: {out_dir}")


def main() -> None:
    print("\nChilliGuard — Preprocessing & Augmentation Setup")
    preview_transforms()


if __name__ == "__main__":
    main()
