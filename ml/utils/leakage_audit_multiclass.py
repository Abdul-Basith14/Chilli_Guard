"""
ChilliGuard ML -- Data Leakage Audit & Validation Split for Dataset II (Phase 3).

Reads dataset_II_inventory.csv and:
1. Audits exact-duplicate (MD5) images crossing the train/test boundary.
2. Excludes train-side duplicate copies from the training pool.
3. Creates a stratified validation split from the clean training pool only (15%).
4. Saves data/dataset_II_phase3_splits.csv with columns:
       path, filename, class_name, label_idx, split, leakage_flag
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("chilliguard.leakage_audit_multiclass")


def build_multiclass_label_mapping(classes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    sorted_classes = sorted(list(set(classes)))
    label_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    idx_to_label = {i: c for i, c in enumerate(sorted_classes)}
    return label_to_idx, idx_to_label


def audit_dataset_ii_leakage(inventory_df: pd.DataFrame) -> dict:
    """
    Identifies exact-duplicate hashes shared between train and test splits in Dataset II.
    """
    dup_groups = inventory_df.groupby("md5_hash")["split_folder"].agg(set)
    cross_exact = dup_groups[dup_groups.apply(lambda s: "train" in s and "test" in s)]
    leaky_hashes = set(cross_exact.index)

    leaky_train_paths = set(
        inventory_df[
            (inventory_df["md5_hash"].isin(leaky_hashes)) &
            (inventory_df["split_folder"] == "train")
        ]["path"].tolist()
    )

    return {
        "exact_cross_split_groups": int(len(cross_exact)),
        "exact_cross_split_hashes": sorted(leaky_hashes),
        "train_side_leaky_paths": sorted(leaky_train_paths),
        "note": "Train-side copies of cross-split duplicates are excluded from training.",
    }


def create_phase3_splits(
    inventory_path: Path,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict, dict[str, int]]:
    """
    Build clean train / val / test split DataFrame for Phase 3 (19 classes).
    """
    df = pd.read_csv(inventory_path)
    audit = audit_dataset_ii_leakage(df)
    leaky_train_paths = set(audit["train_side_leaky_paths"])

    classes = sorted(df["class_name"].unique().tolist())
    label_to_idx, _ = build_multiclass_label_mapping(classes)
    df["label_idx"] = df["class_name"].map(label_to_idx)
    df["split"] = df["split_folder"]

    train_rows = df[df["split"] == "train"].copy()
    test_rows  = df[df["split"] == "test"].copy()

    train_rows["leakage_flag"] = train_rows["path"].isin(leaky_train_paths)
    clean_train = train_rows[~train_rows["leakage_flag"]].copy()
    leaky_excluded = train_rows[train_rows["leakage_flag"]].copy()

    # Stratified validation split from clean train pool
    sub_train, sub_val = train_test_split(
        clean_train,
        test_size=val_ratio,
        stratify=clean_train["class_name"],
        random_state=random_seed,
    )
    sub_train = sub_train.copy()
    sub_val   = sub_val.copy()
    sub_train["split"] = "train"
    sub_val["split"]   = "val"

    leaky_excluded["split"] = "excluded"
    leaky_excluded["leakage_flag"] = True

    test_rows["leakage_flag"] = False

    df_phase3 = pd.concat([sub_train, sub_val, leaky_excluded, test_rows], ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_phase3.to_csv(output_path, index=False)
        logger.info(f"Phase 3 split manifest saved to {output_path}")

    return df_phase3, audit, label_to_idx
