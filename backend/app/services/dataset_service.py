"""
Dataset inspection service connecting FastAPI routers to ML DatasetInspector logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import settings
from ml.config import cfg
from ml.utils.inspector import DatasetInspector

logger = logging.getLogger("chilliguard.backend.dataset_service")


class DatasetService:
    """Service wrapper for dataset status checks and inspection execution."""

    def __init__(self) -> None:
        self.reports_dir = cfg.reports_dir
        self.dataset_path = cfg.dataset_i_path

    def get_status(self) -> Dict[str, Any]:
        """Check dataset path status and existing cached inspection report."""
        inspector = DatasetInspector(dataset_path=self.dataset_path)
        structure_type = inspector.detect_structure()
        exists = self.dataset_path.exists()

        summary_json_path = self.reports_dir / "metrics" / "dataset_summary.json"
        has_report = summary_json_path.exists()

        last_inspected = None
        summary_data = {}

        if has_report:
            try:
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                last_inspected = summary_data.get("timestamp")
            except Exception as err:
                logger.error(f"Error reading summary JSON: {err}")

        return {
            "dataset_path": str(self.dataset_path),
            "exists": exists,
            "structure_type": structure_type,
            "num_total_images": summary_data.get("total_images_discovered", 0),
            "num_readable_images": summary_data.get("readable_images", 0),
            "num_corrupted_images": summary_data.get("corrupted_images", 0),
            "has_inspection_report": has_report,
            "last_inspected_at": last_inspected,
            "class_counts": summary_data.get("class_counts", {}),
        }

    def run_inspection(self) -> Dict[str, Any]:
        """Execute full dataset inspection synchronously."""
        inspector = DatasetInspector(dataset_path=self.dataset_path, reports_dir=self.reports_dir)
        inspector.discover_and_validate()
        inspector.generate_visualizations()
        report_paths = inspector.export_reports()

        summary_json_path = report_paths["dataset_summary_json"]
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        return summary_data

    def get_full_report(self) -> Dict[str, Any]:
        """Return cached summary JSON, Markdown report text, and key metrics."""
        summary_json_path = self.reports_dir / "metrics" / "dataset_summary.json"
        md_report_path = self.reports_dir / "dataset_inspection_report.md"

        if not summary_json_path.exists():
            return {
                "summary": {},
                "markdown_report": "Inspection report has not been generated yet. Please run inspection.",
                "inventory_count": 0,
                "corrupted_count": 0,
                "duplicate_groups_count": 0,
            }

        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        md_text = ""
        if md_report_path.exists():
            with open(md_report_path, "r", encoding="utf-8") as f:
                md_text = f.read()

        return {
            "summary": summary,
            "markdown_report": md_text,
            "inventory_count": summary.get("total_images_discovered", 0),
            "corrupted_count": summary.get("corrupted_images", 0),
            "duplicate_groups_count": summary.get("exact_duplicate_groups", 0) + summary.get("perceptual_duplicate_groups", 0),
        }


# Singleton service instance
dataset_service = DatasetService()
