"""
Script 11: Grad-CAM Visualisations — Phase 6

Generates Gradient-weighted Class Activation Maps (Grad-CAM) for both
the binary and 19-class ResNet18 models.

Grad-CAM overview:
  Grad-CAM (Selvaraju et al., 2017) computes the gradient of the score for a
  target class with respect to the feature maps of the last convolutional layer.
  The gradient signal is globally average-pooled to produce per-channel weights,
  which are then combined with the feature maps and passed through ReLU to obtain
  a coarse spatial activation map that highlights discriminative regions.

  Grad-CAM is an interpretability tool — it shows WHICH spatial regions the model
  attended to when predicting a class. It does NOT prove that the model has learned
  chemically meaningful features, nor does it guarantee that the identified regions
  are biologically or chemically significant. The maps are model-specific and
  depend on both the input image and the training data distribution.

Implementation:
  This script uses a manual hook-based Grad-CAM implementation compatible with
  PyTorch's standard autograd system. No external package (e.g., pytorch-grad-cam)
  is required.

Target layer:
  ResNet18's final convolutional block: model.model.layer4[-1]
  This layer produces 512-channel spatial feature maps of size 7×7 for 224×224 input.

Usage:
    # Single image — binary mode
    python ml/11_gradcam.py --image path/to/image.jpg --mode binary

    # Single image — multiclass mode with specific target class
    python ml/11_gradcam.py --image path/to/image.jpg --mode multiclass --target-class KTWS_5

    # Batch from test split (saves examples to reports/figures/gradcam/)
    python ml/11_gradcam.py --mode binary --batch-examples 5
    python ml/11_gradcam.py --mode multiclass --batch-examples 5
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors with Unicode chars
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_lib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.models.binary_cnn import load_checkpoint as load_binary_checkpoint
from ml.models.multiclass_cnn import load_multiclass_checkpoint
from ml.utils.transforms import get_eval_transforms

warnings.filterwarnings("ignore")

# ── Grad-CAM Core ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Manual hook-based Grad-CAM implementation for ResNet-style models.

    Parameters
    ----------
    model : nn.Module
        The full classification model.
    target_layer : nn.Module
        The convolutional layer to extract activations and gradients from.
        For ResNet18, this is typically ``model.model.layer4[-1]``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM saliency map.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W), already preprocessed and on the correct device.
        target_class : int, optional
            Class index to explain. Defaults to the argmax predicted class.

        Returns
        -------
        cam : np.ndarray
            Normalised heatmap in [0, 1], shape (H, W).
        pred_class : int
            Predicted class index.
        pred_score : float
            Softmax probability of the target class.
        """
        self.model.eval()

        # Forward pass
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        if target_class is None:
            target_class = pred_class

        pred_score = float(probs[0, target_class].item())

        # Zero all gradients and backward on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients → weights (α)
        gradients = self._gradients  # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)  # ReLU: keep only positive contributions

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, pred_score

    def remove_hooks(self) -> None:
        """Deregister the forward/backward hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Overlay helpers ───────────────────────────────────────────────────────────

def cam_to_heatmap(cam: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """Resize CAM to img_size and convert to RGB heatmap (0-255)."""
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray(np.uint8(cam * 255))
    cam_resized = cam_img.resize(img_size, PILImage.BILINEAR)
    cam_arr = np.array(cam_resized) / 255.0
    heatmap = cm_lib.jet(cam_arr)[:, :, :3]  # (H, W, 3) RGB
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend heatmap over original image."""
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return overlay


def save_gradcam_figure(
    original_img: np.ndarray,
    heatmap_img: np.ndarray,
    overlay_img: np.ndarray,
    cam: np.ndarray,
    pred_label: str,
    true_label: Optional[str],
    pred_score: float,
    target_label: str,
    save_path: Path,
    model_name: str = "ResNet18",
) -> None:
    """Generate a 4-panel figure: original | CAM heatmap | overlay | colour bar."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM Heatmap\nTarget: {target_label}", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_img)
    verdict = f"Predicted: {pred_label} ({pred_score*100:.1f}%)"
    if true_label is not None:
        verdict += f"\nTrue: {true_label}"
    axes[2].set_title(f"Overlay\n{verdict}", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f"Grad-CAM — {model_name}\n"
        "Note: Highlighted regions indicate where the model attended, "
        "not necessarily chemically meaningful features.",
        fontsize=8.5, style="italic", y=1.01,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Grad-CAM saved: {save_path}")


# ── Binary Model Grad-CAM ─────────────────────────────────────────────────────

def run_binary_gradcam(
    image_path: Optional[Path],
    target_class: Optional[int],
    batch_examples: int,
) -> None:
    ckpt_path = cfg.model_dir / "binary_exp_b_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    device = torch.device("cpu")
    model, _ = load_binary_checkpoint(ckpt_path, num_classes=2, pretrained=False, device=device)
    model.train()  # Required for gradients to flow through BN

    target_layer = model.model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    class_names = ["Pure", "Adulterated"]
    transform = get_eval_transforms(cfg.image_size)
    out_dir = cfg.reports_dir / "figures" / "gradcam"

    def _process_image(img_path: Path, true_label: Optional[str], out_name: str):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        cam, pred_cls, score = grad_cam(tensor, target_class=target_class)
        pred_label = class_names[pred_cls]
        tgt_label = class_names[target_class if target_class is not None else pred_cls]
        heatmap = cam_to_heatmap(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        overlay = overlay_heatmap(img_rgb, heatmap)
        save_gradcam_figure(
            img_rgb, heatmap, overlay, cam,
            pred_label=pred_label,
            true_label=true_label,
            pred_score=score,
            target_label=tgt_label,
            save_path=out_dir / out_name,
            model_name="ResNet18 Binary Classifier",
        )

    if image_path:
        _process_image(image_path, None, "binary_single.png")
    elif batch_examples > 0:
        import pandas as pd
        split_csv = cfg.data_root / "dataset_I_phase2_splits.csv"
        if not split_csv.exists():
            split_csv = cfg.data_root / "dataset_I_split.csv"
        df = pd.read_csv(split_csv)
        df_test = df[df["split"] == "test"].copy()
        # Sample roughly equal per class
        per_class_n = max(1, batch_examples // 2)
        samples = []
        for cls_idx in [0, 1]:
            sub = df_test[df_test["label_idx"] == cls_idx].sample(
                n=min(per_class_n, len(df_test[df_test["label_idx"] == cls_idx])),
                random_state=42,
            )
            samples.append(sub)
        import pandas as pd_inner
        selected = pd_inner.concat(samples)

        for i, (_, row) in enumerate(selected.iterrows()):
            img_path = Path(row["path"])
            if not img_path.exists():
                img_path = cfg.project_root / row["path"]
            if not img_path.exists():
                continue
            label_name = class_names[int(row["label_idx"])]
            out_name = f"binary_example_{i+1:02d}_{label_name.lower()}.png"
            _process_image(img_path, label_name, out_name)

    grad_cam.remove_hooks()
    print(f"\n  Binary Grad-CAM figures saved to: {out_dir}")


# ── Multiclass Model Grad-CAM ─────────────────────────────────────────────────

def run_multiclass_gradcam(
    image_path: Optional[Path],
    target_class_name: Optional[str],
    batch_examples: int,
) -> None:
    ckpt_path = cfg.model_dir / "multiclass_exp_c_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    device = torch.device("cpu")
    model, payload = load_multiclass_checkpoint(ckpt_path, pretrained=False, device=device)
    model.train()  # Required for gradients through BN

    class_to_idx = payload.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    target_layer = model.model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    transform = get_eval_transforms(cfg.image_size)
    out_dir = cfg.reports_dir / "figures" / "gradcam"

    # Resolve target class index from name if provided
    target_class_idx = None
    if target_class_name and target_class_name in class_to_idx:
        target_class_idx = class_to_idx[target_class_name]
        print(f"  Target class: {target_class_name} (index {target_class_idx})")
    elif target_class_name:
        print(f"  [WARN] Class '{target_class_name}' not found. Using predicted class.")

    def _process_image(img_path: Path, true_label: Optional[str], out_name: str):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        cam, pred_cls, score = grad_cam(tensor, target_class=target_class_idx)
        pred_label = class_names[pred_cls]
        tgt_label = class_names[target_class_idx if target_class_idx is not None else pred_cls]
        heatmap = cam_to_heatmap(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        overlay = overlay_heatmap(img_rgb, heatmap)
        save_gradcam_figure(
            img_rgb, heatmap, overlay, cam,
            pred_label=pred_label,
            true_label=true_label,
            pred_score=score,
            target_label=tgt_label,
            save_path=out_dir / out_name,
            model_name="ResNet18 19-Class Partial Fine-Tune",
        )

    if image_path:
        _process_image(image_path, None, "multiclass_single.png")
    elif batch_examples > 0:
        import pandas as pd
        split_csv = cfg.data_root / "dataset_II_phase3_splits.csv"
        df = pd.read_csv(split_csv)
        df_test = df[df["split"] == "test"].copy()
        # Sample a spread of classes
        sample_classes = class_names[:min(batch_examples, len(class_names))]
        selected_rows = []
        for cn in sample_classes:
            cls_idx = class_to_idx.get(cn, -1)
            sub = df_test[df_test["label_idx"] == cls_idx]
            if len(sub) > 0:
                selected_rows.append(sub.sample(n=1, random_state=42))
        if selected_rows:
            import pandas as pd_inner
            selected = pd_inner.concat(selected_rows)
            for i, (_, row) in enumerate(selected.iterrows()):
                img_path = Path(row["path"])
                if not img_path.exists():
                    img_path = cfg.project_root / row["path"]
                if not img_path.exists():
                    continue
                cls_name = idx_to_class.get(int(row["label_idx"]), "unknown")
                out_name = f"multiclass_example_{i+1:02d}_{cls_name}.png"
                _process_image(img_path, cls_name, out_name)

    grad_cam.remove_hooks()
    print(f"\n  Multiclass Grad-CAM figures saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChilliGuard Phase 6 — Grad-CAM Explainability"
    )
    parser.add_argument(
        "--mode", choices=["binary", "multiclass", "both"], default="both",
        help="Which model to run Grad-CAM on (default: both)",
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to a single input image. If omitted, uses batch-examples from test set.",
    )
    parser.add_argument(
        "--target-class", type=str, default=None,
        help="For multiclass mode: class name to explain (e.g. 'KTWS_5'). "
             "For binary mode: 0=Pure, 1=Adulterated as integer string.",
    )
    parser.add_argument(
        "--batch-examples", type=int, default=5,
        help="Number of test-set examples to generate when no --image is given (default: 5).",
    )
    args = parser.parse_args()

    # Validate image path
    if args.image and not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    # Binary target class
    binary_target = None
    if args.target_class and args.mode in ("binary", "both"):
        try:
            binary_target = int(args.target_class)
        except ValueError:
            pass

    out_base = cfg.reports_dir / "figures" / "gradcam"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ChilliGuard Phase 6 — Grad-CAM Visualisation")
    print("=" * 65)
    print(f"  Output directory: {out_base}")
    print(
        "\n  ACADEMIC DISCLAIMER:\n"
        "  Grad-CAM highlights spatial regions that influenced the model's prediction.\n"
        "  These regions do NOT necessarily correspond to chemically significant\n"
        "  features of the chilli powder sample. Model predictions should be\n"
        "  interpreted in the context of training data and task scope only.\n"
    )

    if args.mode in ("binary", "both"):
        print("\n--- Binary Grad-CAM ---")
        run_binary_gradcam(args.image, binary_target, args.batch_examples)

    if args.mode in ("multiclass", "both"):
        print("\n--- Multiclass Grad-CAM ---")
        run_multiclass_gradcam(args.image, args.target_class, args.batch_examples)

    print("\n  [Done] Grad-CAM visualisations complete.")


if __name__ == "__main__":
    main()
