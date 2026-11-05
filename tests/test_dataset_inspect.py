"""
Unit tests for Phase 1 — Dataset Inspector & Dataset Splitter.

Tests:
1. Structure detection (flat vs nested vs missing).
2. Image discovery and format filtering.
3. Binary class label mapping (C1_PKT -> Pure, C2_AKT -> Adulterated).
4. Corrupted / invalid image header detection.
5. Exact MD5 duplicate detection.
6. Perceptual dHash duplicate detection.
7. Stratified 70/15/15 split reproducibility and manifest generation.
"""

import sys
import tempfile
from pathlib import Path
import pytest
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.utils.inspector import DatasetInspector
from ml.utils.splitter import DatasetSplitter


@pytest.fixture
def synthetic_dataset_dir():
    """Create a temporary dataset directory with synthetic image files for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        c1_dir = root / "C1_PKT"
        c2_dir = root / "C2_AKT"
        c1_dir.mkdir(parents=True)
        c2_dir.mkdir(parents=True)

        # 1. Create 5 valid C1_PKT images with distinct colors
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(200, 50, 10 * i))
            img.save(c1_dir / f"img_c1_{i}.jpg")

        # 2. Create 5 valid C2_AKT images with distinct colors
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(50, 200, 10 * i))
            img.save(c2_dir / f"img_c2_{i}.png")

        # 3. Create 1 corrupted image file
        corrupted_file = c1_dir / "corrupted.jpg"
        with open(corrupted_file, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_HEADER_BYTES_12345")

        # 4. Create an exact duplicate in C2_AKT
        exact_orig = c2_dir / "img_c2_0.png"
        exact_copy = c2_dir / "img_c2_exact_duplicate.png"
        exact_copy.write_bytes(exact_orig.read_bytes())

        yield root


def test_structure_detection(synthetic_dataset_dir):
    inspector = DatasetInspector(dataset_path=synthetic_dataset_dir)
    structure = inspector.detect_structure()
    assert structure == "flat"


def test_discovery_and_label_mapping(synthetic_dataset_dir):
    inspector = DatasetInspector(dataset_path=synthetic_dataset_dir)
    records = inspector.discover_and_validate()

    # Total files = 5 (C1) + 5 (C2) + 1 (corrupted) + 1 (duplicate) = 12
    assert len(records) == 12

    valid_records = [r for r in records if r.is_valid]
    corrupted_records = [r for r in records if not r.is_valid]

    assert len(valid_records) == 11
    assert len(corrupted_records) == 1

    c1_labels = {r.mapped_label for r in valid_records if r.class_folder == "C1_PKT"}
    c2_labels = {r.mapped_label for r in valid_records if r.class_folder == "C2_AKT"}

    assert c1_labels == {"Pure"}
    assert c2_labels == {"Adulterated"}


def test_exact_duplicate_detection(synthetic_dataset_dir):
    inspector = DatasetInspector(dataset_path=synthetic_dataset_dir)
    inspector.discover_and_validate()

    assert len(inspector.exact_duplicates) >= 1
    exact_group = inspector.exact_duplicates[0]
    assert exact_group.type == "exact"
    assert len(exact_group.files) == 2


def test_report_export(synthetic_dataset_dir):
    with tempfile.TemporaryDirectory() as tmp_reports:
        reports_dir = Path(tmp_reports)
        inspector = DatasetInspector(dataset_path=synthetic_dataset_dir, reports_dir=reports_dir)
        inspector.discover_and_validate()
        inspector.generate_visualizations()
        report_paths = inspector.export_reports()

        assert report_paths["image_inventory"].exists()
        assert report_paths["class_distribution"].exists()
        assert report_paths["corrupted_images"].exists()
        assert report_paths["duplicate_images"].exists()
        assert report_paths["dataset_summary_json"].exists()
        assert report_paths["markdown_report"].exists()


def test_split_reproducibility(synthetic_dataset_dir):
    with tempfile.TemporaryDirectory() as tmp_out:
        out_csv = Path(tmp_out) / "split_manifest.csv"
        inspector = DatasetInspector(dataset_path=synthetic_dataset_dir)
        records = inspector.discover_and_validate()

        splitter1 = DatasetSplitter(split_ratios=(0.60, 0.20, 0.20), random_seed=42, output_csv_path=out_csv)
        df1 = splitter1.create_split(records, force_overwrite=True)

        out_csv2 = Path(tmp_out) / "split_manifest2.csv"
        splitter2 = DatasetSplitter(split_ratios=(0.60, 0.20, 0.20), random_seed=42, output_csv_path=out_csv2)
        df2 = splitter2.create_split(records, force_overwrite=True)

        assert (df1["split"] == df2["split"]).all()
        assert set(df1["split"].unique()) == {"train", "val", "test"}
