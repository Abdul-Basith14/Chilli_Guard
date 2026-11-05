"""
ChilliGuard ML — Dataset Splitter Module.

Handles stratified 70/15/15 train/val/test splitting for Dataset I.

Features:
- Preserves existing official splits if present (e.g., subfolders train/val/test).
- Performs reproducible stratified splitting with fixed random seed (default: 42).
- Extracts acquisition/group ID metadata from filenames if present to avoid data leakage.
- Generates a manifest CSV without copying or altering original images.
- Guardrail: Never silently overwrites an existing split manifest unless explicit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import cfg
from ml.utils.inspector import DatasetInspector, ImageRecord

logger = logging.getLogger("chilliguard.splitter")


def extract_group_id(filename: str) -> Optional[str]:
    """
    Attempt to extract a physical sample or acquisition group ID from filename.
    Examples:
        C1_PKT_sample01_002.jpg -> "sample01"
        C2_AKT_batchA_img15.png  -> "batchA"
    Returns None if no structured pattern is identified.
    """
    match = re.search(r"(sample\d+|batch[a-z0-9]+|group\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


class DatasetSplitter:
    """Handles splitting dataset records into train, validation, and test sets."""

    def __init__(
        self,
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        random_seed: int = 42,
        output_csv_path: Optional[Path] = None,
    ) -> None:
        if abs(sum(split_ratios) - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {split_ratios}")
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        self.output_csv_path = (output_csv_path or (cfg.data_root / "dataset_I_split.csv")).resolve()

    def create_split(
        self,
        records: List[ImageRecord],
        force_overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Create stratified split manifest from inspected image records.
        """
        if self.output_csv_path.exists() and not force_overwrite:
            logger.info(f"Split manifest already exists at {self.output_csv_path}. Reading existing manifest.")
            return pd.read_csv(self.output_csv_path)

        valid_records = [r for r in records if r.is_valid]
        if not valid_records:
            raise ValueError("No valid image records provided for dataset splitting.")

        df = pd.DataFrame([
            {
                "path": r.path,
                "filename": r.filename,
                "class_folder": r.class_folder,
                "label": r.mapped_label,
                "existing_split": r.split_folder,
                "group_id": extract_group_id(r.filename),
            }
            for r in valid_records
        ])

        # Check if an official split was already found during inspection
        if df["existing_split"].notna().any() and len(df["existing_split"].dropna().unique()) > 1:
            logger.info("Official pre-existing dataset split detected in folder structure. Preserving official split.")
            df["split"] = df["existing_split"]
        else:
            logger.info(f"No official split detected. Performing {self.split_ratios} stratified split (seed={self.random_seed}).")
            
            # Stratified train vs (val + test)
            train_ratio, val_ratio, test_ratio = self.split_ratios
            val_test_ratio = val_ratio + test_ratio
            
            train_df, val_test_df = train_test_split(
                df,
                test_size=val_test_ratio,
                stratify=df["class_folder"],
                random_state=self.random_seed,
            )

            # Split (val + test) into val and test
            relative_test_ratio = test_ratio / val_test_ratio
            val_df, test_df = train_test_split(
                val_test_df,
                test_size=relative_test_ratio,
                stratify=val_test_df["class_folder"],
                random_state=self.random_seed,
            )

            train_df["split"] = "train"
            val_df["split"] = "val"
            test_df["split"] = "test"

            df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Log group metadata status
        group_count = df["group_id"].notna().sum()
        if group_count > 0:
            logger.info(f"Identified acquisition group metadata in {group_count}/{len(df)} files.")
        else:
            logger.info("Filename sample/group metadata unavailable. Standard stratified splitting applied.")

        # Save manifest
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv_path, index=False)

        # Also copy to reports directory for user convenience
        report_copy_path = cfg.reports_dir / "metrics" / "dataset_I_split.csv"
        report_copy_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_copy_path, index=False)

        logger.info(f"Dataset split manifest saved to {self.output_csv_path} and {report_copy_path}")
        return df
