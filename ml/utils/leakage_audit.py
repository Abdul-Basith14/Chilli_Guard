"""
ChilliGuard ML -- Data Leakage Audit & Validation Split (Phase 2).

Reads the Phase 1 manifest (dataset_I_split.csv) and:

1.  Audits exact-duplicate (MD5) and perceptual-duplicate (dHash) images
    that cross the train/test boundary.

2.  Excludes the train-side copies of any cross-split duplicates from the
    training pool.  The official test set is NOT modified.

3.  Creates a stratified validation split from the *clean* training pool,
    ensuring the test set is never touched.

4.  Saves a phase2_splits.csv with columns:
        path, filename, class_folder, label, split, leakage_flag
    where split in {train, val, test} and leakage_flag marks images
    excluded from training due to cross-split duplication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("chilliguard.leakage_audit")


# Label to integer index  (Pure=0 is the minority / positive class for F1)
LABEL_TO_IDX: dict[str, int] = {"Pure": 0, "Adulterated": 1}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}


def _load_inventory(inventory_path: Path) -> pd.DataFrame:
    df = pd.read_csv(inventory_path)
    required = {"md5_hash", "dhash", "split_folder", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"inventory.csv is missing columns: {missing}")
    return df


def audit_cross_split_leakage(
    manifest_path: Path,
    inventory_path: Path,
) -> dict:
    """
    Return a leakage audit report:
      - exact_cross_split_groups: list of MD5 hashes shared across train & test
      - perceptual_cross_split_groups: list of dHash groups crossing the boundary
      - train_side_leaky_paths: paths in train that have a duplicate in test
    """
    manifest = pd.read_csv(manifest_path)
    inventory = _load_inventory(inventory_path)

    # Build a path -> hash lookup from inventory
    inv_lookup = inventory.set_index("path")[["md5_hash", "dhash", "split_folder"]]

    # -- Exact duplicates across splits --
    dup_groups = inventory.groupby("md5_hash")["split_folder"].agg(set)
    cross_exact = dup_groups[dup_groups.apply(lambda s: "train" in s and "test" in s)]
    leaky_hashes = set(cross_exact.index)

    # Train-side images whose hash also appears in test
    leaky_train_paths = set(
        inventory[(inventory["md5_hash"].isin(leaky_hashes)) &
                  (inventory["split_folder"] == "train")]["path"].tolist()
    )

    # -- Perceptual duplicates across splits --
    perc_groups = inventory.groupby("dhash")["split_folder"].agg(set)
    cross_perc = perc_groups[perc_groups.apply(lambda s: "train" in s and "test" in s and len(s) > 1)]

    report = {
        "exact_cross_split_groups": int(len(cross_exact)),
        "exact_cross_split_hashes": sorted(leaky_hashes),
        "train_side_leaky_paths": sorted(leaky_train_paths),
        "perceptual_cross_split_groups": int(len(cross_perc)),
        "note": (
            "Train-side copies of cross-split duplicates are excluded from the "
            "training pool. The official test set is NOT modified."
        ),
    }
    return report


def create_phase2_splits(
    manifest_path: Path,
    inventory_path: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build clean train / val / test split DataFrames for Phase 2.

    Returns:
        df_phase2: DataFrame with columns [path, filename, class_folder,
                   label, label_idx, split, leakage_flag]
        audit:     Leakage audit dict (for logging / saving)
    """
    manifest = pd.read_csv(manifest_path)
    audit = audit_cross_split_leakage(manifest_path, inventory_path)

    leaky_train_paths = set(audit["train_side_leaky_paths"])

    train_rows = manifest[manifest["split"] == "train"].copy()
    test_rows  = manifest[manifest["split"] == "test"].copy()

    # Mark leaky train images
    train_rows["leakage_flag"] = train_rows["path"].isin(leaky_train_paths)

    # Clean pool: exclude leaky images from training/validation selection
    clean_train = train_rows[~train_rows["leakage_flag"]].copy()
    leaky_excluded = train_rows[train_rows["leakage_flag"]].copy()

    logger.info(
        f"Train pool: {len(train_rows)} total | "
        f"{len(clean_train)} clean | {len(leaky_excluded)} excluded (cross-split dupes)"
    )

    # Stratified val split from clean training pool only
    if len(clean_train) > 0:
        stratify_col = clean_train["label"]
        # Check we have enough samples per class
        min_class_count = stratify_col.value_counts().min()
        actual_val_ratio = val_ratio if min_class_count * (1 - val_ratio) >= 1 else 0.15

        sub_train, sub_val = train_test_split(
            clean_train,
            test_size=actual_val_ratio,
            stratify=stratify_col,
            random_state=random_seed,
        )
        sub_train = sub_train.copy()
        sub_val   = sub_val.copy()
        sub_train["split"] = "train"
        sub_val["split"]   = "val"
    else:
        sub_train = clean_train.copy()
        sub_val   = pd.DataFrame(columns=clean_train.columns)

    # Bring excluded rows back as "excluded" (not used for training or val)
    leaky_excluded["split"] = "excluded"
    leaky_excluded["leakage_flag"] = True

    test_rows["leakage_flag"] = False

    df_phase2 = pd.concat([sub_train, sub_val, leaky_excluded, test_rows], ignore_index=True)
    df_phase2["label_idx"] = df_phase2["label"].map(LABEL_TO_IDX)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_phase2.to_csv(output_path, index=False)
        logger.info(f"Phase 2 split manifest saved to {output_path}")

    # Summary counts for logging
    split_counts = df_phase2.groupby(["split", "label"]).size().unstack(fill_value=0)
    logger.info(f"\nPhase 2 split counts:\n{split_counts}")

    return df_phase2, audit


def split_summary(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable split summary dict."""
    summary = {}
    for split in ["train", "val", "test", "excluded"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        summary[split] = {
            "total": int(len(sub)),
            "Pure": int((sub["label"] == "Pure").sum()),
            "Adulterated": int((sub["label"] == "Adulterated").sum()),
        }
    return summary
