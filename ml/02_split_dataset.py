"""
Script 02: Dataset Splitting Module (Phase 1)

Generates a stratified 70/15/15 train/val/test split manifest CSV.
Reads existing inspection inventory or runs inspector if missing.

Usage:
    python ml/02_split_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.utils.inspector import DatasetInspector
from ml.utils.splitter import DatasetSplitter


def parse_args():
    parser = argparse.ArgumentParser(description="ChilliGuard Phase 1 — Dataset Splitting")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing split manifest CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("ChilliGuard Phase 1 -- Dataset Splitting")
    print("=" * 70)

    inspector = DatasetInspector(dataset_path=cfg.dataset_i_path)
    records = inspector.discover_and_validate()

    if not records:
        print(f"[ERROR] No images found at {cfg.dataset_i_path}. Cannot generate split.")
        sys.exit(1)

    splitter = DatasetSplitter(
        split_ratios=cfg.train_val_test_split,
        random_seed=cfg.random_seed,
    )

    df_split = splitter.create_split(records, force_overwrite=args.force)

    print("\n[SPLIT BREAKDOWN] Dataset Split Breakdown:")
    split_counts = df_split.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(split_counts)

    print("\n[SUCCESS] Dataset split manifest ready at:")
    print(f"   {splitter.output_csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
