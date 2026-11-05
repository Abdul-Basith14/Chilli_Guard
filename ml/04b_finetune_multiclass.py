"""
Script 04b: Fine-Tuning Experiments C & D -- Phase 3

Experiment C: Partial fine-tuning — ResNet18 with layer4 + FC head unfrozen.
Experiment D: Full fine-tuning   — Entire ResNet18 backbone + FC head trained.

Both experiments use class-weighted CrossEntropyLoss (same as Exp B baseline)
and the same fixed train/val split, preprocessing, seed, and evaluation methodology.

Selection criterion: validation macro-F1 (NOT test metrics).

Architecture changes from baseline (Exp A/B):
  Exp C: Unfreezes backbone.layer4 (residual block 4) in addition to the FC head.
         Uses differential LRs: layer4 LR=1e-4, FC head LR=1e-3.
  Exp D: Unfreezes entire backbone.
         Uses differential LRs: backbone LR=1e-5, FC head LR=1e-3.

Device: Intel Iris Xe integrated GPU (CPU fallback).
Usage:
  python -u ml/04b_finetune_multiclass.py [--exp C|D|both] [--epochs N]
                                          [--batch-size N] [--backbone-lr F]
                                          [--head-lr F] [--patience N]
                                          [--run-test] [--test-exp C|D]
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
import pandas as pd
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
logger = logging.getLogger("chilliguard.finetune_multiclass")


# ── Reproducibility ────────────────────────────────────────────────────────────

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


# ── Class weights (training set only) ─────────────────────────────────────────

def compute_class_weights(train_dataset: ManifestDataset, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = train_dataset.labels
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Training / Validation epoch ────────────────────────────────────────────────

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


# ── Full evaluation (macro-F1, top-5, per-class, etc.) ────────────────────────

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


# ── Core fine-tuning experiment ────────────────────────────────────────────────

def run_finetune_experiment(
    exp_id: str,
    fine_tune_mode: str,
    df_phase3: pd.DataFrame,
    class_names: list[str],
    class_to_idx: dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    image_size: tuple,
    patience: int,
    seed: int,
) -> dict:
    set_seed(seed)
    num_classes = len(class_names)
    t_start_total = time.time()

    print(f"\n{'='*65}")
    print(f"  Experiment {exp_id}: Partial fine-tuning (mode='{fine_tune_mode}') | WITH class weighting")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print(f"  Backbone LR: {backbone_lr}  |  Head LR: {head_lr}  |  Patience: {patience}")
    print(f"  Fine-tune mode: {fine_tune_mode}  |  Seed: {seed}")
    print(f"{'='*65}")

    # --- Datasets & Loaders ---
    train_ds = ManifestDataset(df_phase3, "train", transform=get_train_transforms(image_size), cache_in_memory=True)
    val_ds   = ManifestDataset(df_phase3, "val",   transform=get_eval_transforms(image_size),  cache_in_memory=True)
    print(f"\n  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images  |  Classes: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # --- Model ---
    model = MulticlassCNN(
        num_classes=num_classes,
        pretrained=True,
        fine_tune_mode=fine_tune_mode,
    )
    model.to(device)
    trainable, total = model.parameter_count()
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # --- Differential LRs ---
    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    print(f"  Optimizer param groups: {len(param_groups)}")
    for i, g in enumerate(param_groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"    Group {i+1}: {n_params:,} params, lr={g['lr']}")

    # --- Loss & Optimizer ---
    weights = compute_class_weights(train_ds, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(param_groups)
    # Cosine annealing LR schedule — suitable for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- Training loop with early stopping ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1  = -1.0
    best_epoch   = -1
    patience_ctr = 0
    ckpt_path = cfg.model_dir / f"multiclass_exp_{exp_id.lower()}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        val_eval = evaluate_full(model, val_loader, device, class_names)
        val_f1   = val_eval["f1_macro"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  ({elapsed:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            save_multiclass_checkpoint(
                model, optimizer, epoch, val_f1, class_to_idx, ckpt_path,
                extra_meta={
                    "exp_id": exp_id,
                    "fine_tune_mode": fine_tune_mode,
                    "use_class_weights": True,
                    "backbone_lr": backbone_lr,
                    "head_lr": head_lr,
                },
            )
            print(f"    [BEST] Checkpoint saved (val_macro_f1={val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  [EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    total_time = time.time() - t_start_total
    print(f"\n  Best checkpoint: epoch {best_epoch}, val_macro_f1={best_val_f1:.4f}")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Checkpoint: {ckpt_path}")

    # --- Evaluate best checkpoint on val ---
    best_model, _ = load_multiclass_checkpoint(ckpt_path, device=device)
    val_metrics = evaluate_full(best_model, val_loader, device, class_names)

    print(f"\n  [VAL] Results from best checkpoint (Exp {exp_id}):")
    print(f"  Accuracy         : {val_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {val_metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1      : {val_metrics['f1_weighted']:.4f}")
    print(f"  Top-5 Accuracy   : {val_metrics.get('top5_accuracy', 'N/A')}")
    print("\n" + val_metrics["classification_report"])

    # --- Save plots ---
    fig_dir = cfg.reports_dir / "figures"
    save_training_curves(
        history,
        fig_dir / f"phase3_exp_{exp_id.lower()}_training_curves.png",
        title=f"Phase 3 Exp {exp_id} Fine-Tuning Training Curves",
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
        "fine_tune_mode": fine_tune_mode,
        "use_class_weights": True,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "total_training_time_seconds": round(total_time, 1),
        "device": str(device),
        "trainable_params": trainable,
        "total_params": total,
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


# ── Final held-out test evaluation ────────────────────────────────────────────

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

    model, payload = load_multiclass_checkpoint(checkpoint_path, device=device)
    meta = payload.get("meta", {})
    print(f"  Loaded checkpoint meta: {meta}")
    test_metrics = evaluate_full(model, test_loader, device, class_names)

    print(f"\n  [TEST] Final Results:")
    print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1         : {test_metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1      : {test_metrics['f1_weighted']:.4f}")
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


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ChilliGuard Phase 3 -- Fine-Tuning Experiments C & D")
    parser.add_argument("--exp",        choices=["C", "D", "both"], default="C",
                        help="Which experiment(s) to run (default: C).")
    parser.add_argument("--epochs",     type=int,   default=15,
                        help="Maximum training epochs (default: 15).")
    parser.add_argument("--batch-size", type=int,   default=32,
                        help="Mini-batch size (default: 32).")
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="LR for unfrozen backbone layers (default: 1e-4 for Exp C, 1e-5 for Exp D).")
    parser.add_argument("--head-lr",    type=float, default=1e-3,
                        help="LR for the FC classification head (default: 1e-3).")
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Early-stopping patience (epochs without val F1 improvement).")
    parser.add_argument("--run-test",   action="store_true",
                        help="Evaluate the selected best model on the held-out test set.")
    parser.add_argument("--test-exp",   choices=["C", "D"], default="C",
                        help="Which experiment checkpoint to use for the final test evaluation.")
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(cfg.random_seed)
    device = resolve_device(cfg.device)

    print("\n" + "=" * 65)
    print("  ChilliGuard Phase 3 -- Fine-Tuning Experiments (19 Classes)")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  Experiments : {args.exp}")
    print(f"  Max Epochs  : {args.epochs}  |  Batch: {args.batch_size}")
    print(f"  Head LR     : {args.head_lr}  |  Patience: {args.patience}")

    # --- Load manifest ---
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

    # ── Experiment C: Partial fine-tuning (layer4 + FC head) ──────────────────
    if args.exp in ("C", "both"):
        exp_c_backbone_lr = args.backbone_lr   # default 1e-4
        results["C"] = run_finetune_experiment(
            exp_id="C",
            fine_tune_mode="partial",
            df_phase3=df_phase3,
            class_names=class_names,
            class_to_idx=class_to_idx,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_lr=exp_c_backbone_lr,
            head_lr=args.head_lr,
            weight_decay=cfg.weight_decay,
            image_size=cfg.image_size,
            patience=args.patience,
            seed=cfg.random_seed,
        )
        exp_path = cfg.experiments_dir / "phase3_exp_C.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["C"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Phase 3 Exp C results saved: {exp_path}")

    # ── Experiment D: Full fine-tuning (entire backbone) ──────────────────────
    if args.exp in ("D", "both"):
        # For full fine-tuning, backbone LR should be much smaller
        exp_d_backbone_lr = 1e-5
        print(f"\n  [NOTE] Exp D: Overriding backbone LR to {exp_d_backbone_lr} (full fine-tuning).")
        results["D"] = run_finetune_experiment(
            exp_id="D",
            fine_tune_mode="full",
            df_phase3=df_phase3,
            class_names=class_names,
            class_to_idx=class_to_idx,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_lr=exp_d_backbone_lr,
            head_lr=args.head_lr,
            weight_decay=cfg.weight_decay,
            image_size=cfg.image_size,
            patience=args.patience,
            seed=cfg.random_seed,
        )
        exp_path = cfg.experiments_dir / "phase3_exp_D.json"
        with open(exp_path, "w") as f:
            safe = {k: v for k, v in results["D"].items() if k != "history"}
            json.dump(safe, f, indent=2)
        print(f"\n  Phase 3 Exp D results saved: {exp_path}")

    # ── Final test evaluation (only if --run-test is passed) ──────────────────
    if args.run_test:
        test_exp = args.test_exp
        ckpt = cfg.model_dir / f"multiclass_exp_{test_exp.lower()}_best.pth"
        if ckpt.exists():
            test_metrics = evaluate_on_test(ckpt, df_phase3, class_names, device, cfg.image_size, test_exp)
            test_out = cfg.experiments_dir / f"phase3_exp_{test_exp}_test_results.json"
            with open(test_out, "w") as f:
                safe = {k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred")}
                json.dump(safe, f, indent=2)
            print(f"\n  Phase 3 Test results saved: {test_out}")
        else:
            print(f"[WARNING] Checkpoint not found for test evaluation: {ckpt}")
            print("  --> Run --exp C (or D) first, then re-run with --run-test.")

    print("\n" + "=" * 65)
    print("  Phase 3 Fine-Tuning experiments complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
