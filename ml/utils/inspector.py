"""
ChilliGuard ML — Dataset Inspector Module.

Provides comprehensive, non-destructive exploratory data analysis (EDA) for
Dataset I (and extensible to Dataset II).

Features:
- Structure & split detection (flat class folders vs nested train/val/test)
- Format discovery & verification (.jpg, .jpeg, .png, .bmp, .tiff, .webp)
- Label mapping (C1_PKT -> Pure, C2_AKT -> Adulterated)
- Image integrity & corruption checks via PIL
- Dimension, format, and color mode statistics
- Exact duplicate detection (MD5 hash) & perceptual duplicate detection (dHash)
- Data visualization (class distribution plot, sample grid plot)
- Export of CSV inventories, JSON summaries, and Markdown report
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ml.config import cfg

logger = logging.getLogger("chilliguard.inspector")


@dataclass
class ImageRecord:
    """Metadata record for a single image file."""
    path: str
    filename: str
    class_folder: str
    mapped_label: str
    split_folder: Optional[str]  # e.g. "train", "val", "test", or None if flat
    format: str
    width: int
    height: int
    aspect_ratio: float
    color_mode: str
    file_size_bytes: int
    md5_hash: str
    dhash: str
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class DuplicateGroup:
    """Group of duplicate images."""
    group_id: int
    type: str  # "exact" or "perceptual"
    hash_value: str
    files: List[str]


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Compute difference hash (dHash) for perceptual duplicate detection using PIL.
    Resizes image to (hash_size + 1, hash_size) in grayscale, compares adjacent pixels.
    """
    try:
        gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray)
        # Difference between consecutive columns
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Convert boolean array to hex string
        decimal_val = 0
        hex_str = ""
        for i, val in enumerate(diff.flatten()):
            if val:
                decimal_val += 2 ** (i % 4)
            if (i % 4) == 3:
                hex_str += f"{decimal_val:x}"
                decimal_val = 0
        return hex_str
    except Exception:
        return ""


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex hash strings of equal length."""
    if len(h1) != len(h2) or not h1 or not h2:
        return 999
    try:
        n1 = int(h1, 16)
        n2 = int(h2, 16)
        x = n1 ^ n2
        return bin(x).count("1")
    except ValueError:
        return 999


class DatasetInspector:
    """
    Full-featured inspector for image datasets.
    Operates strictly read-only on input dataset directories.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        class_mapping: Optional[Dict[str, str]] = None,
        reports_dir: Optional[Path] = None,
        perceptual_threshold: int = 2,
    ) -> None:
        self.dataset_path = (dataset_path or cfg.dataset_i_path).resolve()
        self.class_mapping = class_mapping or cfg.binary_class_to_label
        self.reports_dir = (reports_dir or cfg.reports_dir).resolve()
        self.perceptual_threshold = perceptual_threshold

        self.records: List[ImageRecord] = []
        self.structure_type: str = "unknown"
        self.detected_splits: List[str] = []
        self.class_folders: List[str] = []
        self.exact_duplicates: List[DuplicateGroup] = []
        self.perceptual_duplicates: List[DuplicateGroup] = []

    def detect_structure(self) -> str:
        """
        Analyze dataset path structure:
        - "flat": root/class_folder/image.jpg
        - "nested_split": root/train/class_folder/image.jpg, root/val/..., root/test/...
        - "ambiguous": both or unrecognized patterns
        - "missing": path does not exist
        """
        if not self.dataset_path.exists():
            self.structure_type = "missing"
            return self.structure_type

        subdirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        subdir_names = {d.name.lower() for d in subdirs}

        standard_split_names = {"train", "val", "validation", "test"}
        matched_splits = subdir_names.intersection(standard_split_names)

        # Check for class folders directly under root
        expected_class_folders = set(self.class_mapping.keys())
        found_class_folders = subdir_names.intersection({c.lower() for c in expected_class_folders})

        if matched_splits and found_class_folders:
            self.structure_type = "ambiguous"
            logger.warning(
                f"Ambiguous dataset structure at {self.dataset_path}: "
                f"found split subfolders {matched_splits} AND class subfolders {found_class_folders}"
            )
        elif matched_splits:
            self.structure_type = "nested_split"
            self.detected_splits = [d.name for d in subdirs if d.name.lower() in standard_split_names]
        elif found_class_folders or len(subdirs) > 0:
            self.structure_type = "flat"
            self.class_folders = [d.name for d in subdirs]
        else:
            self.structure_type = "empty_or_unrecognized"

        return self.structure_type

    def discover_and_validate(self) -> List[ImageRecord]:
        """
        Recursively scan images, map labels, check image integrity,
        compute hashes, and record metadata.
        """
        self.records.clear()
        self.detect_structure()

        if self.structure_type in ("missing", "empty_or_unrecognized"):
            logger.error(f"Cannot scan dataset: path status is '{self.structure_type}' ({self.dataset_path})")
            return self.records

        image_files: List[Tuple[Path, str, Optional[str]]] = []

        if self.structure_type == "flat" or self.structure_type == "ambiguous":
            for class_folder in self.dataset_path.iterdir():
                if class_folder.is_dir():
                    folder_name = class_folder.name
                    for file_path in class_folder.rglob("*"):
                        if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                            rel_parts = file_path.relative_to(class_folder).parts
                            split_name = None
                            if len(rel_parts) > 1 and rel_parts[0].lower() in {"train", "val", "validation", "test"}:
                                split_name = rel_parts[0].lower()
                                if split_name == "validation":
                                    split_name = "val"
                            image_files.append((file_path, folder_name, split_name))

        elif self.structure_type == "nested_split":
            for split_dir in self.dataset_path.iterdir():
                if split_dir.is_dir():
                    split_name = split_dir.name
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            folder_name = class_dir.name
                            for file_path in class_dir.rglob("*"):
                                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                                    image_files.append((file_path, folder_name, split_name))

        logger.info(f"Discovered {len(image_files)} image files at {self.dataset_path}")

        for file_path, class_folder, split_name in image_files:
            mapped_label = self.class_mapping.get(class_folder, f"Unmapped ({class_folder})")
            file_size = file_path.stat().st_size

            # Compute MD5 Hash
            hasher = hashlib.md5()
            try:
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                md5_hash = hasher.hexdigest()
            except Exception:
                md5_hash = ""

            # Validate Image with PIL
            is_valid = False
            fmt = ""
            width = 0
            height = 0
            aspect_ratio = 0.0
            color_mode = ""
            dhash = ""
            error_msg = None

            try:
                with Image.open(file_path) as img:
                    img.verify()
                
                with Image.open(file_path) as img:
                    fmt = img.format or file_path.suffix.lstrip(".").upper()
                    width, height = img.size
                    aspect_ratio = round(width / max(height, 1), 4)
                    color_mode = img.mode
                    dhash = compute_dhash(img)
                    is_valid = True
            except Exception as err:
                is_valid = False
                error_msg = str(err)
                logger.warning(f"Corrupted or invalid image: {file_path} — {error_msg}")

            record = ImageRecord(
                path=str(file_path),
                filename=file_path.name,
                class_folder=class_folder,
                mapped_label=mapped_label,
                split_folder=split_name,
                format=fmt,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                color_mode=color_mode,
                file_size_bytes=file_size,
                md5_hash=md5_hash,
                dhash=dhash,
                is_valid=is_valid,
                error_message=error_msg,
            )
            self.records.append(record)

        self._find_duplicates()
        return self.records

    def _find_duplicates(self) -> None:
        """Categorize exact and perceptual duplicates from recorded files."""
        self.exact_duplicates.clear()
        self.perceptual_duplicates.clear()

        # 1. Exact Duplicates via MD5
        md5_groups: Dict[str, List[str]] = {}
        for r in self.records:
            if r.is_valid and r.md5_hash:
                md5_groups.setdefault(r.md5_hash, []).append(r.path)

        group_id = 1
        for md5, paths in md5_groups.items():
            if len(paths) > 1:
                self.exact_duplicates.append(
                    DuplicateGroup(
                        group_id=group_id,
                        type="exact",
                        hash_value=md5,
                        files=paths,
                    )
                )
                group_id += 1

        # 2. Perceptual Duplicates via dHash (excluding exact duplicate groups)
        exact_duplicate_files = {p for g in self.exact_duplicates for p in g.files}
        valid_unique_records = [
            r for r in self.records if r.is_valid and r.dhash and r.path not in exact_duplicate_files
        ]

        perc_group_id = 1
        visited = set()
        for i in range(len(valid_unique_records)):
            r1 = valid_unique_records[i]
            if r1.path in visited:
                continue

            matches = [r1.path]
            for j in range(i + 1, len(valid_unique_records)):
                r2 = valid_unique_records[j]
                if r2.path in visited:
                    continue
                dist = hamming_distance(r1.dhash, r2.dhash)
                if dist <= self.perceptual_threshold:
                    matches.append(r2.path)
                    visited.add(r2.path)

            if len(matches) > 1:
                visited.add(r1.path)
                self.perceptual_duplicates.append(
                    DuplicateGroup(
                        group_id=perc_group_id,
                        type="perceptual",
                        hash_value=r1.dhash,
                        files=matches,
                    )
                )
                perc_group_id += 1

    def generate_visualizations(self) -> Tuple[Path, Path]:
        """Generate class distribution bar chart and grid sample image sheet."""
        figures_dir = self.reports_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([asdict(r) for r in self.records if r.is_valid])

        # ── 1. Class Distribution Chart ───────────────────────────────────────
        class_dist_path = figures_dir / "class_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        if not df.empty:
            counts = df.groupby(["class_folder", "mapped_label"]).size().reset_index(name="count")
            colors = ["#4D5D18", "#C0392B", "#27AE60", "#2980B9", "#8E44AD"]
            
            bars = ax.bar(
                [f"{row['mapped_label']}\n({row['class_folder']})" for _, row in counts.iterrows()],
                counts["count"],
                color=[colors[i % len(colors)] for i in range(len(counts))],
                edgecolor="#1E230B",
                linewidth=1.2,
            )

            total = counts["count"].sum()
            for bar in bars:
                h = bar.get_height()
                pct = (h / total) * 100 if total > 0 else 0
                ax.annotate(
                    f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title("Dataset I — Class Distribution (Pure vs Adulterated)", fontsize=13, pad=15, fontweight="bold")
            ax.set_ylabel("Number of Images", fontsize=11)
            ax.set_ylim(0, max(counts["count"]) * 1.18 if not counts.empty else 10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No readable images found", ha="center", va="center", fontsize=14)

        plt.tight_layout()
        plt.savefig(class_dist_path, dpi=300)
        plt.close(fig)

        # ── 2. Sample Image Grid ──────────────────────────────────────────────
        sample_images_path = figures_dir / "sample_images.png"
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=300)
        fig.suptitle("Dataset I — Sample Images Preview", fontsize=14, fontweight="bold", y=0.98)

        if not df.empty:
            classes = df["class_folder"].unique()
            for row_idx, cf in enumerate(classes[:2]):  # up to 2 classes
                class_df = df[df["class_folder"] == cf]
                sample_files = class_df.sample(min(4, len(class_df)), random_state=42) if len(class_df) >= 4 else class_df
                
                for col_idx in range(4):
                    ax = axes[row_idx, col_idx]
                    if col_idx < len(sample_files):
                        rec = sample_files.iloc[col_idx]
                        try:
                            with Image.open(rec["path"]) as img:
                                img_rgb = img.convert("RGB")
                                ax.imshow(img_rgb)
                                label_str = rec["mapped_label"]
                                ax.set_title(
                                    f"{label_str} ({rec['class_folder']})\n{rec['width']}x{rec['height']} | {rec['format']}",
                                    fontsize=8,
                                )
                        except Exception:
                            ax.text(0.5, 0.5, "Error loading", ha="center", va="center")
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.axis("off")
        else:
            for ax in axes.flatten():
                ax.axis("off")
            fig.text(0.5, 0.5, "No images available for preview", ha="center", va="center", fontsize=12)

        plt.tight_layout()
        plt.savefig(sample_images_path, dpi=300)
        plt.close(fig)

        return class_dist_path, sample_images_path

    def export_reports(self) -> Dict[str, Path]:
        """Generate all required CSV files, JSON summary, and Markdown report."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = self.reports_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        # Convert records to DataFrame
        records_data = [asdict(r) for r in self.records]
        df_all = pd.DataFrame(records_data) if records_data else pd.DataFrame()

        # 1. image_inventory.csv
        inv_path = metrics_dir / "image_inventory.csv"
        if not df_all.empty:
            df_all.to_csv(inv_path, index=False)
        else:
            pd.DataFrame(columns=[field.name for field in ImageRecord.__dataclass_fields__.values()]).to_csv(inv_path, index=False)
        paths["image_inventory"] = inv_path

        # 2. class_distribution.csv
        dist_path = metrics_dir / "class_distribution.csv"
        if not df_all.empty and "is_valid" in df_all.columns:
            valid_df = df_all[df_all["is_valid"]]
            if not valid_df.empty:
                dist_df = valid_df.groupby(["class_folder", "mapped_label"]).size().reset_index(name="count")
                dist_df["percentage"] = (dist_df["count"] / dist_df["count"].sum() * 100).round(2)
                dist_df.to_csv(dist_path, index=False)
            else:
                pd.DataFrame(columns=["class_folder", "mapped_label", "count", "percentage"]).to_csv(dist_path, index=False)
        else:
            pd.DataFrame(columns=["class_folder", "mapped_label", "count", "percentage"]).to_csv(dist_path, index=False)
        paths["class_distribution"] = dist_path

        # 3. corrupted_images.csv
        corr_path = metrics_dir / "corrupted_images.csv"
        if not df_all.empty and "is_valid" in df_all.columns:
            corr_df = df_all[~df_all["is_valid"]][["path", "filename", "class_folder", "error_message"]]
            corr_df.to_csv(corr_path, index=False)
        else:
            pd.DataFrame(columns=["path", "filename", "class_folder", "error_message"]).to_csv(corr_path, index=False)
        paths["corrupted_images"] = corr_path

        # 4. duplicate_images.csv
        dup_path = metrics_dir / "duplicate_images.csv"
        dup_rows = []
        for g in self.exact_duplicates + self.perceptual_duplicates:
            for idx, p in enumerate(g.files):
                dup_rows.append({
                    "group_id": g.group_id,
                    "type": g.type,
                    "hash_value": g.hash_value,
                    "file_path": p,
                    "is_primary": idx == 0,
                })
        dup_df = pd.DataFrame(dup_rows) if dup_rows else pd.DataFrame(columns=["group_id", "type", "hash_value", "file_path", "is_primary"])
        dup_df.to_csv(dup_path, index=False)
        paths["duplicate_images"] = dup_path

        # 5. dataset_summary.json
        valid_records = [r for r in self.records if r.is_valid]
        corrupted_records = [r for r in self.records if not r.is_valid]
        widths = [r.width for r in valid_records]
        heights = [r.height for r in valid_records]
        formats = set(r.format for r in valid_records)
        color_modes = set(r.color_mode for r in valid_records)

        class_counts = {}
        for r in valid_records:
            class_counts[r.class_folder] = class_counts.get(r.class_folder, 0) + 1

        summary_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(self.dataset_path),
            "structure_type": self.structure_type,
            "detected_splits": self.detected_splits,
            "total_images_discovered": len(self.records),
            "readable_images": len(valid_records),
            "corrupted_images": len(corrupted_records),
            "class_counts": class_counts,
            "class_mapping": self.class_mapping,
            "dimension_stats": {
                "min_width": min(widths) if widths else 0,
                "max_width": max(widths) if widths else 0,
                "mean_width": float(np.mean(widths)) if widths else 0.0,
                "min_height": min(heights) if heights else 0,
                "max_height": max(heights) if heights else 0,
                "mean_height": float(np.mean(heights)) if heights else 0.0,
            },
            "formats": list(formats),
            "color_modes": list(color_modes),
            "exact_duplicate_groups": len(self.exact_duplicates),
            "exact_duplicate_files_count": sum(len(g.files) for g in self.exact_duplicates),
            "perceptual_duplicate_groups": len(self.perceptual_duplicates),
            "perceptual_duplicate_files_count": sum(len(g.files) for g in self.perceptual_duplicates),
        }

        json_path = metrics_dir / "dataset_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        paths["dataset_summary_json"] = json_path

        # 6. dataset_inspection_report.md
        md_path = self.reports_dir / "dataset_inspection_report.md"
        self._write_markdown_report(md_path, summary_data)
        paths["markdown_report"] = md_path

        return paths

    def _write_markdown_report(self, md_path: Path, summary: Dict[str, Any]) -> None:
        """Write executive Markdown inspection report."""
        dim = summary["dimension_stats"]
        class_table = "\n".join(
            [f"| `{k}` | {self.class_mapping.get(k, 'Unmapped')} | {v:,} |" for k, v in summary["class_counts"].items()]
        ) or "| None | None | 0 |"

        content = f"""# Dataset Inspection & Exploratory Data Analysis Report

**Project:** ChilliGuard — Red Chilli Powder Adulteration Detection  
**Generated:** {summary['timestamp']}  
**Dataset Path:** `{summary['dataset_path']}`  
**Structure Detected:** `{summary['structure_type']}`  

---

## 1. Executive Summary

- **Total Images Discovered:** `{summary['total_images_discovered']:,}`
- **Usable (Readable) Images:** `{summary['readable_images']:,}`
- **Corrupted / Unreadable Images:** `{summary['corrupted_images']:,}`
- **Exact Duplicate Groups (MD5):** `{summary['exact_duplicate_groups']}` ({summary['exact_duplicate_files_count']} total files)
- **Perceptual Duplicate Groups (dHash):** `{summary['perceptual_duplicate_groups']}` ({summary['perceptual_duplicate_files_count']} total files)

---

## 2. Class Distribution & Mapping

The target binary problem classifies pure Khammam Teja chilli powder against adulterated samples.

| Class Folder | Mapped Label | Discovered Count |
|--------------|--------------|------------------|
{class_table}

---

## 3. Image Integrity & Format Summary

- **Supported Formats Found:** `{', '.join(summary['formats']) if summary['formats'] else 'None'}`
- **Color Modes:** `{', '.join(summary['color_modes']) if summary['color_modes'] else 'None'}`
- **Dimensions Summary:**
  - Width: min `{dim['min_width']}` px, max `{dim['max_width']}` px, mean `{dim['mean_width']:.1f}` px
  - Height: min `{dim['min_height']}` px, max `{dim['max_height']}` px, mean `{dim['mean_height']:.1f}` px

---

## 4. Duplicate Findings & Data Quality Assessment

- **Exact Duplicates:** {summary['exact_duplicate_groups']} duplicate set(s) identified via byte hashing.
- **Perceptual Duplicates:** {summary['perceptual_duplicate_groups']} set(s) identified via difference hashing (dHash <= {self.perceptual_threshold}).
- **Corrupted Files:** {summary['corrupted_images']} file(s) failed Pillow header validation (`img.verify()`).

> [!NOTE]
> No files have been deleted, moved, or altered during this inspection phase. Original dataset files remain intact.

---

## 5. Recommended Next Steps

1. **Review Data Splitting:** Verify whether a physical sample grouping metadata exists to prevent data leakage across train/val/test splits.
2. **Execute Stratified Split:** Run `python ml/02_split_dataset.py` to create `dataset_I_split.csv` with a 70/15/15 ratio.
3. **Proceed to Binary Training (Phase 2):** Train the ResNet50 baseline model on verified dataset split.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)
