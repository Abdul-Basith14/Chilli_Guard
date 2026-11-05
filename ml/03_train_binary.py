"""
Script 03: Binary Classifier Training -- Phase 2

Trains a ResNet18-based binary classifier (Pure vs Adulterated) on Dataset I.

Runs two controlled experiments:
  Exp A: CrossEntropyLoss without class weighting (baseline)
  Exp B: CrossEntropyLoss WITH class weighting (imbalance handling)

Architecture:
  - ResNet18 pretrained on ImageNet (torchvision)
  - Backbone FROZEN: only the FC head is trained
  - Two output classes: Pure (label 0) and Adulterated (label 1)
  - Input: 224x224 RGB, ImageNet normalised

Data:
  - Reads from dataset_I_split.csv (produced by Phase 1)
  - 8 cross-split duplicate groups excluded from training pool
  - Validation split carved from clean training pool only (15%)
  - Official 278-image test set held out until final evaluation

Device:
  - Automatically uses CUDA if available, otherwise CPU
  - CPU-optimised defaults: batch_size=16, image_size=224x224

Usage:
  python ml/03_train_binary.py [--exp A|B|both] [--epochs N]
                               [--batch-size N] [--lr FLOAT]
                               [--no-freeze]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.models.binary_cnn import BinaryCNN, load_checkpoint, save_checkpoint
from ml.utils.leakage_audit import LABEL_TO_IDX, IDX_TO_LABEL, create_phase2_splits, split_summary
from ml.utils.manifest_dataset import ManifestDataset
from ml.utils.metrics import (
    compute_binary_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_training_curves,
)
from ml.utils.transforms import get_eval_transforms, get_train_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chilliguard.train")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight computation (from training set only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_dataset: ManifestDataset, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training set labels.
    Returns a 1D tensor of shape [num_classes].
    """
    labels = train_dataset.labels
    counts = np.bincount(labels, minlength=2).astype(float)
    # weight_i = total / (num_classes * count_i)
    weights = len(labels) / (2.0 * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights_tensor


# ─────────────────────────────────────────────────────────────────────────────
# One-epoch train / eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """
    Run one epoch.  Returns (avg_loss, accuracy).
    Pass optimizer=None for validation/test epochs.
    """
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Full evaluation: returns metrics dict + raw y_true/y_pred/y_prob.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            # Probability for the positive class (Pure = index 0)
            y_prob.extend(probs[:, 0].cpu().tolist())

    metrics = compute_binary_metrics(
        y_true, y_pred,
        y_prob=y_prob,
        class_names=class_names,
        positive_label=0,
    )
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_id: str,
    use_class_weights: bool,
    df_phase2,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: tuple,
    freeze_backbone: bool,
    seed: int,
) -> dict:
    """
    Train one experiment (A or B) and return a results dict.
    """
    set_seed(seed)
    class_names = ["Pure", "Adulterated"]  # index 0 = Pure (positive), 1 = Adulterated

    print(f"\n{'='*65}")
    print(f"  Experiment {exp_id}: {'WITH' if use_class_weights else 'WITHOUT'} class weighting")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Backbone frozen: {freeze_backbone} | Seed: {seed}")
    print(f"{'='*65}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ManifestDataset(df_phase2, "train",
                               transform=get_train_transforms(image_size))
    val_ds   = ManifestDataset(df_phase2, "val",
                               transform=get_eval_transforms(image_size))

    print(f"\n  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")
    train_counts = train_ds.class_counts
    val_counts   = val_ds.class_counts
    print(f"  Train class counts: {train_counts}")
    print(f"  Val   class counts: {val_counts}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BinaryCNN(num_classes=2, pretrained=True, freeze_backbone=freeze_backbone)
    model.to(device)
    trainable, total = model.parameter_count()
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # ── Loss ──────────────────────────────────────────────────────────────────
    if use_class_weights:
        class_weights = compute_class_weights(train_ds, device)
        print(f"  Class weights: Pure={class_weights[0]:.3f}, "
              f"Adulterated={class_weights[1]:.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=lr, weight_decay=weight_decay,
    )
    # StepLR scheduler: decay LR by 0.5 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = -1.0
    best_epoch  = -1
    ckpt_path = cfg.model_dir / f"binary_exp_{exp_id.lower()}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Quick val F1 for checkpoint selection
        val_metrics_quick = evaluate_full(model, val_loader, device, class_names)
        val_f1 = val_metrics_quick["f1_macro"]

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_f1:.4f}  ({elapsed:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            save_checkpoint(
                model, optimizer, epoch, val_f1, ckpt_path,
                extra_meta={"exp_id": exp_id, "use_class_weights": use_class_weights},
            )
            print(f"    [BEST] Checkpoint saved (val_macro_f1={val_f1:.4f})")

    print(f"\n  Best checkpoint: epoch {best_epoch}, val_macro_f1={best_val_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    # ── Final val evaluation from best checkpoint ─────────────────────────────
    best_model, _ = load_checkpoint(ckpt_path, device=device)
    best_model.to(device)
    val_metrics = evaluate_full(best_model, val_loader, device, class_names)

    print(f"\n  [VAL] Results from best checkpoint (Exp {exp_id}):")
    print(f"  Accuracy         : {val_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {val_metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC          : {val_metrics.get('auc_roc', 'N/A')}")
    print("\n" + val_metrics["classification_report"])

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_dir = cfg.reports_dir / "figures"
    save_training_curves(
        history,
        fig_dir / f"exp_{exp_id.lower()}_training_curves.png",
        title=f"Exp {exp_id} Training Curves",
    )
    save_confusion_matrix(
        val_metrics["y_true"], val_metrics["y_pred"],
        class_names=class_names,
        save_path=fig_dir / f"exp_{exp_id.lower()}_val_confusion_matrix.png",
        title=f"Exp {exp_id} - Validation Confusion Matrix",
    )
    if val_metrics.get("auc_roc") is not None:
        save_roc_curve(
            val_metrics["y_true"], val_metrics["y_prob"],
            save_path=fig_dir / f"exp_{exp_id.lower()}_val_roc.png",
            title=f"Exp {exp_id} - Validation ROC Curve",
        )

    results = {
        "exp_id": exp_id,
        "use_class_weights": use_class_weights,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["f1_macro"],
        "val_auc_roc": val_metrics.get("auc_roc"),
        "val_per_class": val_metrics["per_class"],
        "val_classification_report": val_metrics["classification_report"],
        "history": history,
        "config": {
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "image_size": list(image_size),
            "freeze_backbone": freeze_backbone,
            "seed": seed,
            "model_arch": "resnet18",
        },
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final test-set evaluation (only called once, after experiment selection)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    checkpoint_path: Path,
    df_phase2,
    device: torch.device,
    image_size: tuple,
    exp_id: str,
) -> dict:
    """Load best checkpoint and evaluate on the held-out test set."""
    class_names = ["Pure", "Adulterated"]
    test_ds = ManifestDataset(df_phase2, "test",
                              transform=get_eval_transforms(image_size))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=0, pin_memory=False)

    print(f"\n{'='*65}")
    print(f"  FINAL TEST EVALUATION (Exp {exp_id})")
    print(f"  Test set: {len(test_ds)} images (Pure={test_ds.class_counts.get('Pure',0)}, "
          f"Adulterated={test_ds.class_counts.get('Adulterated',0)})")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  NOTE: This evaluation is performed ONCE on the untouched test set.")
    print(f"{'='*65}")

    model, meta = load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    test_metrics = evaluate_full(model, test_loader, device, class_names)

    print(f"\n  [TEST] Final Results:")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {test_metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC          : {test_metrics.get('auc_roc', 'N/A')}")
    print("\n" + test_metrics["classification_report"])

    fig_dir = cfg.reports_dir / "figures"
    save_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        class_names=class_names,
        save_path=fig_dir / f"exp_{exp_id.lower()}_test_confusion_matrix.png",
        title=f"Exp {exp_id} - Final Test Confusion Matrix",
    )
    if test_metrics.get("auc_roc") is not None:
        save_roc_curve(
            test_metrics["y_true"], test_metrics["y_prob"],
            save_path=fig_dir / f"exp_{exp_id.lower()}_test_roc.png",
            title=f"Exp {exp_id} - Final Test ROC Curve",
        )

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ChilliGuard Phase 2 -- Binary Classifier Training"
    )
    parser.add_argument("--exp", choices=["A", "B", "both"], default="both",
                        help="Which experiment to run (default: both)")
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--no-freeze", action="store_true",
                        help="Unfreeze backbone (trains all layers; slower on CPU)")
    parser.add_argument("--run-test", action="store_true",
                        help="After training, evaluate the selected model on the test set")
    parser.add_argument("--test-exp", choices=["A", "B"], default="B",
                        help="Which experiment's checkpoint to use for test evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(cfg.random_seed)
    device = resolve_device(cfg.device)

    print("\n" + "=" * 65)
    print("  ChilliGuard Phase 2 -- Binary Classification Training")
    print("=" * 65)
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Backbone frozen: {not args.no_freeze}")
    print(f"  Manifest: {cfg.split_manifest_path}")

    # ── Build Phase 2 splits ──────────────────────────────────────────────────
    inventory_path = cfg.reports_dir / "metrics" / "image_inventory.csv"
    phase2_manifest_path = cfg.data_root / "dataset_I_phase2_splits.csv"

    df_phase2, audit = create_phase2_splits(
        manifest_path=cfg.split_manifest_path,
        inventory_path=inventory_path,
        val_ratio=cfg.phase2_val_split_ratio,
        random_seed=cfg.random_seed,
        output_path=phase2_manifest_path,
    )

    print(f"\n  [LEAKAGE AUDIT]")
    print(f"  Exact cross-split duplicate groups: {audit['exact_cross_split_groups']}")
    print(f"  Train-side images excluded:         {len(audit['train_side_leaky_paths'])}")
    print(f"  Perceptual cross-split groups:      {audit['perceptual_cross_split_groups']}")

    counts = split_summary(df_phase2)
    for split, info in counts.items():
        print(f"  {split:10s}: {info['total']:>4} total  "
              f"(Pure={info['Pure']}, Adulterated={info['Adulterated']})")

    # Save audit to experiments dir
    audit_out = cfg.experiments_dir / "phase2_leakage_audit.json"
    audit_save = {k: v for k, v in audit.items() if k != "train_side_leaky_paths"}
    audit_save["train_side_excluded_count"] = len(audit["train_side_leaky_paths"])
    with open(audit_out, "w") as f:
        json.dump(audit_save, f, indent=2)
    print(f"\n  Leakage audit saved: {audit_out}")

    # ── Run experiments ───────────────────────────────────────────────────────
    freeze = not args.no_freeze
    image_size = cfg.image_size
    results = {}

    exp_kwargs = dict(
        df_phase2=df_phase2,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=cfg.weight_decay,
        image_size=image_size,
        freeze_backbone=freeze,
        seed=cfg.random_seed,
    )

    if args.exp in ("A", "both"):
        results["A"] = run_experiment("A", use_class_weights=False, **exp_kwargs)
        exp_path = cfg.experiments_dir / "phase2_exp_A.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["A"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Exp A results saved: {exp_path}")

    if args.exp in ("B", "both"):
        results["B"] = run_experiment("B", use_class_weights=True, **exp_kwargs)
        exp_path = cfg.experiments_dir / "phase2_exp_B.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["B"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Exp B results saved: {exp_path}")

    # ── Comparison summary ────────────────────────────────────────────────────
    if len(results) == 2:
        print(f"\n{'='*65}")
        print("  EXPERIMENT COMPARISON (Validation Set)")
        print(f"{'='*65}")
        for exp_id, r in results.items():
            print(
                f"  Exp {exp_id}: "
                f"Acc={r['val_accuracy']:.4f}  "
                f"BalAcc={r['val_balanced_accuracy']:.4f}  "
                f"MacroF1={r['val_macro_f1']:.4f}  "
                f"AUC={r.get('val_auc_roc', 'N/A')}"
            )

    # ── Optional: final test evaluation ──────────────────────────────────────
    if args.run_test:
        test_exp = args.test_exp
        ckpt = cfg.model_dir / f"binary_exp_{test_exp.lower()}_best.pth"
        if not ckpt.exists():
            print(f"\n  [WARN] Checkpoint not found for Exp {test_exp}: {ckpt}")
        else:
            test_results = evaluate_on_test(ckpt, df_phase2, device, image_size, test_exp)
            test_out = cfg.experiments_dir / f"phase2_exp_{test_exp}_test_results.json"
            with open(test_out, "w") as f:
                safe = {k: v for k, v in test_results.items()
                        if k not in ("y_true", "y_pred", "y_prob")}
                json.dump(safe, f, indent=2)
            print(f"\n  Test results saved: {test_out}")

    print("\n" + "=" * 65)
    print("  Phase 2 training complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
