"""
Script 05: Model Evaluation

Loads a trained model checkpoint and evaluates it on the held-out test split.
Produces a confusion matrix, classification report, and ROC curve.

Status: PLACEHOLDER — Will be implemented after Phase 2 training is complete.

Usage (Phase 2+):
    python ml/05_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg


def main() -> None:
    print("\nChilliGuard — Model Evaluation (Phase 2+)")
    print("─" * 55)

    binary_ckpt = cfg.model_dir / "binary_classifier.pth"
    if not binary_ckpt.exists():
        print(f"[INFO] No binary model checkpoint found at: {binary_ckpt}")
        print("  → Train the binary classifier first (Phase 2).")
        return

    # ── Phase 2+ implementation outline ───────────────────────────────────────
    # 1. Load model checkpoint
    # 2. Build test DataLoader from held-out split
    # 3. Run inference on test set
    # 4. Compute: accuracy, precision, recall, F1 (macro + weighted)
    # 5. Generate confusion matrix → save to reports/figures/
    # 6. Generate ROC curve (binary) → save to reports/figures/
    # 7. Save classification report JSON → reports/metrics/
    # ─────────────────────────────────────────────────────────────────────────
    print("Evaluation logic will be implemented in Phase 2.")


if __name__ == "__main__":
    main()
