"""
Evaluation metric helpers for ChilliGuard (Binary & Multiclass).

Uses scikit-learn to compute standard classification metrics.
All functions are framework-agnostic -- pass plain Python lists or NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_binary_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: Optional[list | np.ndarray] = None,
    class_names: Optional[list[str]] = None,
    positive_label: int = 0,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = class_names or ["Class 0", "Class 1"]
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {},
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=labels,
            zero_division=0,
        ),
        "positive_class": labels[positive_label],
    }

    for idx, name in enumerate(labels):
        y_true_bin = [1 if t == idx else 0 for t in y_true]
        y_pred_bin = [1 if p == idx else 0 for p in y_pred]
        result["per_class"][name] = {
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            "recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            "f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            "support":   int(sum(1 for t in y_true if t == idx)),
        }

    if y_prob is not None:
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["auc_roc"] = None

    return result


def compute_multiclass_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list[str],
    top_k_preds: Optional[list[list[int]]] = None,
) -> dict:
    """
    Compute full multiclass metrics for 19 classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    per_class = {}
    prec_per_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_cls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)

    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(prec_per_cls[idx]),
            "recall": float(rec_per_cls[idx]),
            "f1": float(f1_per_cls[idx]),
            "support": int(sum(1 for t in y_true if t == idx)),
        }

    # Top-5 accuracy
    top5_acc = None
    if top_k_preds is not None:
        hits = 0
        for t, topk in zip(y_true, top_k_preds):
            if t in topk:
                hits += 1
        top5_acc = float(hits / len(y_true))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "top5_accuracy": top5_acc,
        "per_class": per_class,
        "classification_report": report,
    }


def save_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Generate and save a confusion matrix figure."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))),
                          normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f" if normalize else "d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved: {save_path}")


def save_roc_curve(
    y_true: list | np.ndarray,
    y_scores: list | np.ndarray,
    save_path: Path,
    title: str = "ROC Curve",
) -> float:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] ROC curve saved: {save_path}")
    return roc_auc


def save_training_curves(
    history: dict,
    save_path: Path,
    title: str = "Training Curves",
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Training curves saved: {save_path}")
