"""
Script 06: Single-Image Inference -- Phase 2 & 3

Loads a trained ChilliGuard model (Binary or Multiclass) and runs inference on a single image.

Usage:
    # Binary classification (Phase 2 default)
    python ml/06_inference.py --image path/to/sample.jpg --mode binary

    # 19-class multiclass classification (Phase 3)
    python ml/06_inference.py --image path/to/sample.jpg --mode multiclass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.models.binary_cnn import load_checkpoint as load_binary_checkpoint
from ml.models.multiclass_cnn import load_multiclass_checkpoint
from ml.utils.transforms import get_eval_transforms

BINARY_LABELS = ["Pure", "Adulterated"]


def resolve_binary_checkpoint(custom_ckpt: Path | None) -> Path | None:
    if custom_ckpt and custom_ckpt.exists():
        return custom_ckpt
    for c in [
        cfg.model_dir / "binary_exp_b_best.pth",
        cfg.model_dir / "binary_exp_a_best.pth",
        cfg.model_dir / "binary_classifier.pth",
    ]:
        if c.exists():
            return c
    return None


def resolve_multiclass_checkpoint(custom_ckpt: Path | None) -> Path | None:
    if custom_ckpt and custom_ckpt.exists():
        return custom_ckpt
    for c in [
        cfg.model_dir / "multiclass_exp_c_best.pth",
        cfg.model_dir / "multiclass_exp_d_best.pth",
        cfg.model_dir / "multiclass_exp_b_best.pth",
        cfg.model_dir / "multiclass_exp_a_best.pth",
    ]:
        if c.exists():
            return c
    return None


def run_binary_inference(image_path: Path, ckpt_path: Path | None = None) -> None:
    checkpoint = resolve_binary_checkpoint(ckpt_path)
    if not checkpoint:
        print("[ERROR] No trained binary checkpoint found in ml/models/")
        sys.exit(1)

    print("\nChilliGuard -- Single-Image Binary Inference (Phase 2)")
    print("=" * 60)
    print(f"Input image  : {image_path}")
    print(f"Checkpoint   : {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")

    model, meta = load_binary_checkpoint(checkpoint, num_classes=2, pretrained=False, device=device)
    img = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms(cfg.image_size)(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = BINARY_LABELS[pred_idx]

    print("\nPrediction Results:")
    print("-" * 60)
    print(f"  Classification : {pred_label.upper()}")
    print(f"  Confidence     : {probs[pred_idx] * 100:.2f}%")
    print(f"  Probabilities  :")
    print(f"    - Pure        (`C1_PKT`): {probs[0] * 100:.2f}%")
    print(f"    - Adulterated (`C2_AKT`): {probs[1] * 100:.2f}%")
    print("=" * 60 + "\n")


def run_multiclass_inference(image_path: Path, ckpt_path: Path | None = None) -> None:
    checkpoint = resolve_multiclass_checkpoint(ckpt_path)
    if not checkpoint:
        print("[ERROR] No trained multiclass checkpoint found in ml/models/")
        sys.exit(1)

    print("\nChilliGuard -- Single-Image Multiclass Inference (Phase 3 -- 19 Classes)")
    print("=" * 60)
    print(f"Input image  : {image_path}")
    print(f"Checkpoint   : {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")

    model, payload = load_multiclass_checkpoint(checkpoint, pretrained=False, device=device)
    class_to_idx = payload.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    img = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms(cfg.image_size)(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = class_names[pred_idx]
    top5_indices = torch.topk(torch.tensor(probs), k=min(5, len(class_names))).indices.tolist()

    print("\nPrediction Results:")
    print("-" * 60)
    print(f"  Predicted Class: {pred_label.upper()}")
    print(f"  Confidence     : {probs[pred_idx] * 100:.2f}%")
    print(f"\n  Top-5 Probabilities:")
    for rank, idx in enumerate(top5_indices, 1):
        print(f"    {rank}. {class_names[idx]:<12}: {probs[idx] * 100:.2f}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChilliGuard -- Single Image Inference")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary", help="Classifier mode.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional custom checkpoint path.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)

    if args.mode == "binary":
        run_binary_inference(args.image, args.ckpt)
    else:
        run_multiclass_inference(args.image, args.ckpt)


if __name__ == "__main__":
    main()
