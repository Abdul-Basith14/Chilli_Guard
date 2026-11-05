"""
Script 01: Dataset Inspection & Exploratory Data Analysis (Phase 1)

Runs non-destructive inspection on Dataset I (and checks Dataset II),
verifies image integrity, calculates exact & perceptual duplicates,
generates distribution & preview plots, exports CSV inventory/summaries,
and outputs a complete Markdown report.

Usage:
    python ml/01_dataset_inspect.py [--dataset-path PATH] [--perceptual-threshold INT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.utils.inspector import DatasetInspector


def parse_args():
    parser = argparse.ArgumentParser(description="ChilliGuard Phase 1 — Dataset Inspection")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to Dataset I directory (defaults to DATASET_I_PATH in .env)",
    )
    parser.add_argument(
        "--perceptual-threshold",
        type=int,
        default=2,
        help="Hamming distance threshold for perceptual dHash duplicates (default: 2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path) if args.dataset_path else cfg.dataset_i_path

    print("\n" + "=" * 70)
    print("ChilliGuard Phase 1 -- Dataset Inspection & Exploratory Data Analysis")
    print("=" * 70)
    print(f"Target Dataset Path : {dataset_path.resolve()}")
    print(f"Reports Output Dir  : {cfg.reports_dir.resolve()}")
    print(f"Binary Class Map    : {cfg.binary_class_to_label}")
    print("-" * 70)

    inspector = DatasetInspector(
        dataset_path=dataset_path,
        class_mapping=cfg.binary_class_to_label,
        reports_dir=cfg.reports_dir,
        perceptual_threshold=args.perceptual_threshold,
    )

    structure = inspector.detect_structure()
    print(f"\n[INFO] Detected Folder Structure: [{structure.upper()}]")

    if structure in ("missing", "empty_or_unrecognized"):
        print(f"\n[ERROR] Dataset directory not found or empty at: {dataset_path}")
        print("  Please configure DATASET_I_PATH in your .env file or pass --dataset-path.")
        print("  Expected directory structure:")
        print("    data/dataset_I/")
        print("    ├── C1_PKT/   (Pure Khammam Teja images)")
        print("    └── C2_AKT/   (Adulterated images)")
        sys.exit(1)

    print("\n[INFO] Discovering images, inspecting file headers, and computing hashes...")
    records = inspector.discover_and_validate()

    valid_records = [r for r in records if r.is_valid]
    corrupted_records = [r for r in records if not r.is_valid]

    print("\n[SUMMARY] Statistics:")
    print(f"  Total Discovered Files : {len(records):,}")
    print(f"  Readable Images        : {len(valid_records):,}")
    print(f"  Corrupted / Invalid    : {len(corrupted_records):,}")

    class_counts = {}
    for r in valid_records:
        class_counts[r.class_folder] = class_counts.get(r.class_folder, 0) + 1

    print("\n[BREAKDOWN] Per-Class Image Breakdown:")
    for folder_name, label in cfg.binary_class_to_label.items():
        cnt = class_counts.get(folder_name, 0)
        print(f"  - {folder_name:12s} ({label:12s}): {cnt:>6,} images")

    print("\n[DUPLICATES] Summary:")
    print(f"  - Exact Duplicates (MD5)  : {len(inspector.exact_duplicates)} groups")
    print(f"  - Perceptual Duplicates   : {len(inspector.perceptual_duplicates)} groups")

    print("\n[VISUALIZATION] Generating Visualizations...")
    class_chart, sample_chart = inspector.generate_visualizations()
    print(f"  [OK] Class distribution chart : {class_chart}")
    print(f"  [OK] Sample image preview     : {sample_chart}")

    print("\n[REPORTS] Exporting Reports & Inventories...")
    report_paths = inspector.export_reports()
    for name, pth in report_paths.items():
        print(f"  [OK] {name:22s} : {pth}")

    print("\n" + "=" * 70)
    print("SUCCESS: Dataset inspection complete! Executive report generated at:")
    print(f"   {report_paths['markdown_report']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
