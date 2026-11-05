"""
Script: compare_and_select_multiclass.py

Compares Phase 3 experiments (B, C, D) on validation macro-F1, selects the winner,
evaluates it once on the held-out test set, and writes:
  - reports/phase3_model_comparison.md
  - experiments/phase3_winner_test_results.json

Selection criterion: validation macro-F1.

Usage:
  python -u ml/05_compare_select_multiclass.py [--skip-test]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.models.multiclass_cnn import load_multiclass_checkpoint
from ml.utils.leakage_audit_multiclass import create_phase3_splits
from ml.utils.manifest_dataset import ManifestDataset
from ml.utils.metrics import compute_multiclass_metrics, save_confusion_matrix
from ml.utils.transforms import get_eval_transforms
from torch.utils.data import DataLoader


# ── Load experiment result JSONs ───────────────────────────────────────────────

def load_experiment(exp_id: str) -> dict | None:
    path = cfg.experiments_dir / f"phase3_exp_{exp_id}.json"
    if not path.exists():
        print(f"  [SKIP] Experiment {exp_id}: JSON not found at {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data


# ── Evaluate a checkpoint on any split ────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: Path,
    df: pd.DataFrame,
    split: str,
    class_names: list[str],
    device: torch.device,
    image_size: tuple,
) -> dict:
    ds = ManifestDataset(df, split, transform=get_eval_transforms(image_size))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    model, payload = load_multiclass_checkpoint(ckpt_path, device=device)
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

    metrics = compute_multiclass_metrics(y_true, y_pred, class_names, top_k_preds=top_k_preds)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


# ── Build markdown comparison table ───────────────────────────────────────────

def build_comparison_markdown(experiments: list[dict], winner_id: str) -> str:
    lines = [
        "# Phase 3 Multiclass Model Comparison",
        "",
        "**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  ",
        "**Task:** 19-Class Red Chilli Powder Classification (Dataset II)  ",
        "**Selection Criterion:** Validation Macro-F1 Score  ",
        "",
        "---",
        "",
        "## Experiment Comparison Table (Validation Set)",
        "",
        "| Experiment | Strategy | Backbone LR | Head LR | Trainable Params | Best Epoch | Val Top-1 Acc | Val Top-5 Acc | Val Balanced Acc | **Val Macro-F1** | Val Weighted-F1 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for exp in experiments:
        eid = exp.get("exp_id", "?")
        strategy = {
            "B": "Frozen backbone + FC head (class-weighted CELoss)",
            "C": "Partial fine-tune: layer4 + FC head (class-weighted CELoss)",
            "D": "Full fine-tune: entire backbone + FC head (class-weighted CELoss)",
        }.get(eid, "Unknown")
        backbone_lr = exp.get("backbone_lr", "N/A")
        head_lr = exp.get("head_lr", exp.get("lr", "N/A"))
        trainable = exp.get("trainable_params", "N/A")
        if isinstance(trainable, int):
            trainable = f"{trainable:,}"
        best_epoch = exp.get("best_epoch", "?")
        val_acc  = f"{exp.get('val_accuracy', 0)*100:.2f}%"
        val_top5 = f"{exp.get('val_top5_accuracy', 0)*100:.2f}%"
        val_bal  = f"{exp.get('val_balanced_accuracy', 0)*100:.2f}%"
        val_f1   = f"{exp.get('val_macro_f1', 0):.4f}"
        val_wf1  = f"{exp.get('val_weighted_f1', 0):.4f}"
        winner_mark = " 🏆" if eid == winner_id else ""
        lines.append(
            f"| **Exp {eid}{winner_mark}** | {strategy} | {backbone_lr} | {head_lr} | "
            f"{trainable} | {best_epoch} | {val_acc} | {val_top5} | {val_bal} | **{val_f1}** | {val_wf1} |"
        )

    lines += [
        "",
        f"> **Winner: Experiment {winner_id}** (selected by highest validation macro-F1).",
        "",
        "---",
        "",
        "## Final Test Evaluation (Held-Out — 1,093 Images)",
        "",
        "> The test set was untouched during all training and model selection steps.",
        "> It was evaluated exactly **once** on the selected winning model.",
        "",
        "### Results will appear here after test evaluation runs.",
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "1. **Experiment B (Baseline):** ResNet18 backbone fully frozen. Only the 2-layer FC head (9,747 parameters) was trained.",
        "   This represents the minimal fine-tuning setup where only task-specific linear weights are learned.",
        "2. **Experiment C (Partial Fine-Tuning):** The last residual block (`layer4`, 8,393,728 parameters) was unfrozen,",
        "   in addition to the FC head. A differential LR was applied: `backbone LR=1e-4, head LR=1e-3`.",
        "3. **Experiment D (Full Fine-Tuning):** The entire ResNet18 backbone was unfrozen.",
        "   Differential LR: `backbone LR=1e-5, head LR=1e-3` to prevent catastrophic forgetting.",
        "4. **Common settings:** class-weighted CrossEntropyLoss, CosineAnnealingLR schedule, AdamW optimizer,",
        "   early stopping (patience=5), fixed seed=42, batch size=32, max 15 epochs.",
        "5. **Model selection:** Best model chosen exclusively by **validation macro-F1**, without any test set information.",
        "",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Phase 3 experiments and select winner")
    parser.add_argument("--skip-test", action="store_true", help="Skip the final test evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("  Phase 3 Model Comparison & Selection")
    print("=" * 65)

    # Load available experiment results
    exp_ids = ["B", "C", "D"]
    experiments = []
    for eid in exp_ids:
        data = load_experiment(eid)
        if data is not None:
            experiments.append(data)

    if not experiments:
        print("[ERROR] No experiment results found. Train at least Exp B or C first.")
        sys.exit(1)

    print(f"\n  Found {len(experiments)} experiment(s): {[e['exp_id'] for e in experiments]}")

    # Select winner by validation macro-F1
    winner = max(experiments, key=lambda e: e.get("val_macro_f1", -1))
    winner_id = winner["exp_id"]
    print(f"\n  [WINNER] Winner: Experiment {winner_id}  (val_macro_f1={winner['val_macro_f1']:.4f})")
    print("\n  Validation comparison:")
    for exp in experiments:
        marker = " <-- WINNER" if exp["exp_id"] == winner_id else ""
        print(f"    Exp {exp['exp_id']}: val_macro_f1={exp.get('val_macro_f1', 0):.4f}"
              f"  val_acc={exp.get('val_accuracy', 0)*100:.2f}%"
              f"  val_top5={exp.get('val_top5_accuracy', 0)*100:.2f}%{marker}")

    # Build comparison markdown
    md = build_comparison_markdown(experiments, winner_id)

    # --- Final test evaluation ---
    if not args.skip_test:
        print(f"\n  Running FINAL test evaluation for Exp {winner_id}...")

        inventory_path = cfg.reports_dir / "metrics" / "dataset_II_inventory.csv"
        phase3_manifest_path = cfg.data_root / "dataset_II_phase3_splits.csv"
        df_phase3, _, class_to_idx = create_phase3_splits(
            inventory_path=inventory_path,
            val_ratio=cfg.phase2_val_split_ratio,
            random_seed=cfg.random_seed,
            output_path=phase3_manifest_path,
        )
        class_names = sorted(list(class_to_idx.keys()))

        ckpt_path = cfg.model_dir / f"multiclass_exp_{winner_id.lower()}_best.pth"
        if not ckpt_path.exists():
            print(f"  [ERROR] Winner checkpoint not found: {ckpt_path}")
            sys.exit(1)

        test_metrics = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            df=df_phase3,
            split="test",
            class_names=class_names,
            device=device,
            image_size=cfg.image_size,
        )

        print(f"\n  [TEST] Final Results (Exp {winner_id}):")
        print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"  Macro-F1         : {test_metrics['f1_macro']:.4f}")
        print(f"  Weighted-F1      : {test_metrics['f1_weighted']:.4f}")
        print(f"  Top-5 Accuracy   : {test_metrics.get('top5_accuracy', 'N/A')}")
        print("\n" + test_metrics["classification_report"])

        # Save confusion matrix
        fig_dir = cfg.reports_dir / "figures"
        save_confusion_matrix(
            test_metrics["y_true"], test_metrics["y_pred"],
            class_names=class_names,
            save_path=fig_dir / f"phase3_winner_exp_{winner_id.lower()}_test_confusion_matrix.png",
            title=f"Phase 3 Winner (Exp {winner_id}) - Final Test Confusion Matrix",
            figsize=(12, 10),
        )

        # Save test results JSON
        test_out = cfg.experiments_dir / f"phase3_winner_exp_{winner_id}_test_results.json"
        with open(test_out, "w") as f:
            safe = {k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred")}
            json.dump(safe, f, indent=2)
        print(f"\n  Winner test results saved: {test_out}")

        # Update the markdown with actual test results
        test_section = [
            "## Final Test Evaluation (Held-Out — 1,093 Images)",
            "",
            "> The test set was untouched during all training and model selection steps.",
            "> It was evaluated exactly **once** on the selected winning model.",
            "",
            f"**Winning Model:** Experiment {winner_id}  ",
            f"**Checkpoint:** `{ckpt_path.name}`  ",
            "",
            "| Metric | Score |",
            "| :--- | :---: |",
            f"| **Top-1 Accuracy** | {test_metrics['accuracy']*100:.2f}% |",
            f"| **Top-5 Accuracy** | {test_metrics.get('top5_accuracy', 0)*100:.2f}% |",
            f"| **Balanced Accuracy** | {test_metrics['balanced_accuracy']*100:.2f}% |",
            f"| **Macro-F1** | {test_metrics['f1_macro']:.4f} |",
            f"| **Weighted-F1** | {test_metrics['f1_weighted']:.4f} |",
            f"| **Macro-Precision** | {test_metrics.get('precision_macro', 0):.4f} |",
            f"| **Macro-Recall** | {test_metrics.get('recall_macro', 0):.4f} |",
            "",
            "```",
            test_metrics["classification_report"],
            "```",
            "",
            "---",
        ]
        # Replace the placeholder section
        md = md.replace(
            "### Results will appear here after test evaluation runs.\n\n---",
            "\n".join(test_section),
        )

    # Write report
    report_path = cfg.reports_dir / "phase3_model_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n  Model comparison report saved: {report_path}")

    print("\n" + "=" * 65)
    print(f"  Phase 3 Comparison complete. Winner: Experiment {winner_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
