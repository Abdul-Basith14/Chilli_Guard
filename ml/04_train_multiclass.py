"""
Script 04: Multiclass Classifier Training -- Phase 3

Trains a ResNet18-based 19-class classifier (Pure + Adulterant subclasses) on Dataset II.

Runs two controlled experiments:
  Exp A: CrossEntropyLoss without class weighting (baseline)
  Exp B: CrossEntropyLoss WITH class weighting (imbalance handling)

Architecture:
  - ResNet18 pretrained on ImageNet (torchvision)
  - Backbone FROZEN: only the FC head is trained
  - 19 output classes corresponding to Dataset II subfolders
  - Input: 224x224 RGB, ImageNet normalised

Data:
  - Reads from reports/metrics/dataset_II_inventory.csv
  - Audits cross-split duplicates and excludes train-side duplicate copies
  - Stratified validation split carved from clean training pool (15%)
  - Official test set held out until final evaluation

Device:
  - Automatically uses CUDA if available, otherwise CPU
  - CPU-optimised defaults: batch_size=32, image_size=224x224

Usage:
  python -u ml/04_train_multiclass.py [--exp A|B|both] [--epochs N]
                                      [--batch-size N] [--lr FLOAT]
                                      [--no-freeze] [--run-test]
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
from ml.models.multiclass_cnn import MulticlassCNN, load_multiclass_checkpoint, save_multiclass_checkpoint
from ml.utils.leakage_audit_multiclass import create_phase3_splits
from ml.utils.manifest_dataset import ManifestDataset
from ml.utils.metrics import (
    compute_multiclass_metrics,
    save_confusion_matrix,
    save_training_curves,
)
from ml.utils.transforms import get_eval_transforms, get_train_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chilliguard.train_multiclass")


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


def compute_class_weights(train_dataset: ManifestDataset, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = train_dataset.labels
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Avoid zero division
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
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
    model.eval()
    y_true, y_pred, top_k_preds = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # Top 5
            top5 = torch.topk(probs, k=min(5, len(class_names)), dim=1).indices.cpu().tolist()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            top_k_preds.extend(top5)

    metrics = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        top_k_preds=top_k_preds,
    )
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def run_experiment(
    exp_id: str,
    use_class_weights: bool,
    df_phase3: pd.DataFrame,
    class_names: list[str],
    class_to_idx: dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: tuple,
    freeze_backbone: bool,
    seed: int,
) -> dict:
    set_seed(seed)
    num_classes = len(class_names)

    print(f"\n{'='*65}")
    print(f"  Experiment {exp_id}: {'WITH' if use_class_weights else 'WITHOUT'} class weighting (19 Classes)")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Backbone frozen: {freeze_backbone} | Seed: {seed}")
    print(f"{'='*65}")

    train_ds = ManifestDataset(df_phase3, "train", transform=get_train_transforms(image_size), cache_in_memory=True)
    val_ds   = ManifestDataset(df_phase3, "val",   transform=get_eval_transforms(image_size), cache_in_memory=True)

    print(f"\n  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images across {num_classes} classes")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = MulticlassCNN(num_classes=num_classes, pretrained=True, freeze_backbone=freeze_backbone)
    model.to(device)
    trainable, total = model.parameter_count()
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    if use_class_weights:
        weights = compute_class_weights(train_ds, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = -1.0
    best_epoch  = -1
    ckpt_path = cfg.model_dir / f"multiclass_exp_{exp_id.lower()}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        val_eval = evaluate_full(model, val_loader, device, class_names)
        val_f1 = val_eval["f1_macro"]

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
            save_multiclass_checkpoint(
                model, optimizer, epoch, val_f1, class_to_idx, ckpt_path,
                extra_meta={"exp_id": exp_id, "use_class_weights": use_class_weights},
            )
            print(f"    [BEST] Checkpoint saved (val_macro_f1={val_f1:.4f})")

    print(f"\n  Best checkpoint: epoch {best_epoch}, val_macro_f1={best_val_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    # Load best checkpoint for evaluation
    best_model, _ = load_multiclass_checkpoint(ckpt_path, device=device)
    val_metrics = evaluate_full(best_model, val_loader, device, class_names)

    print(f"\n  [VAL] Results from best checkpoint (Exp {exp_id}):")
    print(f"  Accuracy         : {val_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {val_metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1       : {val_metrics['f1_weighted']:.4f}")
    print(f"  Top-5 Accuracy   : {val_metrics.get('top5_accuracy', 'N/A')}")
    print("\n" + val_metrics["classification_report"])

    fig_dir = cfg.reports_dir / "figures"
    save_training_curves(
        history,
        fig_dir / f"phase3_exp_{exp_id.lower()}_training_curves.png",
        title=f"Phase 3 Exp {exp_id} Training Curves",
    )
    save_confusion_matrix(
        val_metrics["y_true"], val_metrics["y_pred"],
        class_names=class_names,
        save_path=fig_dir / f"phase3_exp_{exp_id.lower()}_val_confusion_matrix.png",
        title=f"Phase 3 Exp {exp_id} - Validation Confusion Matrix",
        figsize=(12, 10),
    )

    return {
        "exp_id": exp_id,
        "use_class_weights": use_class_weights,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path),
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["f1_macro"],
        "val_weighted_f1": val_metrics["f1_weighted"],
        "val_top5_accuracy": val_metrics.get("top5_accuracy"),
        "val_per_class": val_metrics["per_class"],
        "val_classification_report": val_metrics["classification_report"],
        "history": history,
    }


def evaluate_on_test(
    checkpoint_path: Path,
    df_phase3: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    image_size: tuple,
    exp_id: str,
) -> dict:
    test_ds = ManifestDataset(df_phase3, "test", transform=get_eval_transforms(image_size))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n{'='*65}")
    print(f"  FINAL HELD-OUT TEST EVALUATION (Phase 3 Exp {exp_id})")
    print(f"  Test set: {len(test_ds)} images across {len(class_names)} classes")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*65}")

    model, _ = load_multiclass_checkpoint(checkpoint_path, device=device)
    test_metrics = evaluate_full(model, test_loader, device, class_names)

    print(f"\n  [TEST] Final Results:")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {test_metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1       : {test_metrics['f1_weighted']:.4f}")
    print(f"  Top-5 Accuracy   : {test_metrics.get('top5_accuracy', 'N/A')}")
    print("\n" + test_metrics["classification_report"])

    fig_dir = cfg.reports_dir / "figures"
    save_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        class_names=class_names,
        save_path=fig_dir / f"phase3_exp_{exp_id.lower()}_test_confusion_matrix.png",
        title=f"Phase 3 Exp {exp_id} - Final Test Confusion Matrix",
        figsize=(12, 10),
    )

    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="ChilliGuard Phase 3 -- Multiclass Classifier Training")
    parser.add_argument("--exp", choices=["A", "B", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--test-exp", choices=["A", "B"], default="B")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(cfg.random_seed)
    device = resolve_device(cfg.device)

    print("\n" + "=" * 65)
    print("  ChilliGuard Phase 3 -- Multiclass Classification Training (19 Classes)")
    print("=" * 65)
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    inventory_path = cfg.reports_dir / "metrics" / "dataset_II_inventory.csv"
    if not inventory_path.exists():
        print(f"[ERROR] Inventory not found: {inventory_path}")
        print("  --> Run python ml/01_dataset_II_inspect.py first.")
        sys.exit(1)

    phase3_manifest_path = cfg.data_root / "dataset_II_phase3_splits.csv"
    df_phase3, audit, class_to_idx = create_phase3_splits(
        inventory_path=inventory_path,
        val_ratio=cfg.phase2_val_split_ratio,
        random_seed=cfg.random_seed,
        output_path=phase3_manifest_path,
    )

    class_names = sorted(list(class_to_idx.keys()))
    print(f"\n  Discovered {len(class_names)} classes: {class_names}")

    results = {}
    exp_kwargs = dict(
        df_phase3=df_phase3,
        class_names=class_names,
        class_to_idx=class_to_idx,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=cfg.weight_decay,
        image_size=cfg.image_size,
        freeze_backbone=not args.no_freeze,
        seed=cfg.random_seed,
    )

    if args.exp in ("A", "both"):
        results["A"] = run_experiment("A", use_class_weights=False, **exp_kwargs)
        exp_path = cfg.experiments_dir / "phase3_exp_A.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["A"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Phase 3 Exp A results saved: {exp_path}")

    if args.exp in ("B", "both"):
        results["B"] = run_experiment("B", use_class_weights=True, **exp_kwargs)
        exp_path = cfg.experiments_dir / "phase3_exp_B.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["B"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Phase 3 Exp B results saved: {exp_path}")

    if args.run_test:
        test_exp = args.test_exp
        ckpt = cfg.model_dir / f"multiclass_exp_{test_exp.lower()}_best.pth"
        if ckpt.exists():
            test_results = evaluate_on_test(ckpt, df_phase3, class_names, device, cfg.image_size, test_exp)
            test_out = cfg.experiments_dir / f"phase3_exp_{test_exp}_test_results.json"
            with open(test_out, "w") as f:
                safe = {k: v for k, v in test_results.items() if k not in ("y_true", "y_pred")}
                json.dump(safe, f, indent=2)
            print(f"\n  Phase 3 Test results saved: {test_out}")

    print("\n" + "=" * 65)
    print("  Phase 3 Multiclass Training complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
