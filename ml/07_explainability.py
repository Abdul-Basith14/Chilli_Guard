"""
Script 07: Explainability — Grad-CAM & SHAP

Generates saliency maps and feature importance visualisations for trained models.

Status: PLACEHOLDER — Phase 5 implementation.
        Requires a trained model checkpoint and the grad-cam / shap packages.

Usage (Phase 5):
    python ml/07_explainability.py --image path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print("\nChilliGuard — Explainability (Phase 5)")
    print("─" * 55)
    print("STATUS: Not yet implemented.")
    print()
    print("Planned techniques:")
    print("  1. Grad-CAM    — Class Activation Maps (pytorch-grad-cam)")
    print("  2. SHAP        — SHapley Additive exPlanations (shap)")
    print("  3. LIME        — Local Interpretable Model-agnostic Explanations")
    print()
    print("  → Implement after Phase 2/3 training is validated.")


if __name__ == "__main__":
    main()
