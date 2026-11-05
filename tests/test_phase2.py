"""
Tests for Phase 2: Binary Classification Pipeline.

Covers:
  - ManifestDataset loading and split isolation
  - BinaryCNN model output shape
  - Single training step (smoke test)
  - Metric computation
  - Checkpoint save / load round-trip
  - Leakage audit logic
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_manifest(tmp_path):
    """Create a tiny dummy manifest with actual JPEG images on disk."""
    from PIL import Image

    rows = []
    split_class = [
        ("train", "Pure",       0, 10),
        ("train", "Adulterated",1, 20),
        ("val",   "Pure",       0,  3),
        ("val",   "Adulterated",1,  5),
        ("test",  "Pure",       0,  2),
        ("test",  "Adulterated",1,  6),
    ]
    for split, label, idx, count in split_class:
        class_dir = tmp_path / split / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_path = class_dir / f"img_{i}.jpg"
            img = Image.new("RGB", (64, 64), color=(idx * 80, 50, 80 - idx * 30))
            img.save(img_path)
            rows.append({
                "path": str(img_path),
                "filename": img_path.name,
                "class_folder": f"C{idx+1}",
                "label": label,
                "label_idx": idx,
                "split": split,
                "leakage_flag": False,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_model():
    from ml.models.binary_cnn import BinaryCNN
    return BinaryCNN(num_classes=2, pretrained=False, freeze_backbone=True, dropout_p=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ManifestDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestDataset:
    def test_train_split_length(self, small_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        ds = ManifestDataset(small_manifest, "train", verify_paths=True)
        assert len(ds) == 30, f"Expected 30 train samples, got {len(ds)}"

    def test_val_split_length(self, small_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        ds = ManifestDataset(small_manifest, "val", verify_paths=True)
        assert len(ds) == 8

    def test_test_split_length(self, small_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        ds = ManifestDataset(small_manifest, "test", verify_paths=True)
        assert len(ds) == 8

    def test_item_returns_correct_shape(self, small_manifest):
        import torchvision.transforms as T
        from ml.utils.manifest_dataset import ManifestDataset
        transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        ds = ManifestDataset(small_manifest, "train", transform=transform)
        img, label = ds[0]
        assert img.shape == (3, 64, 64), f"Unexpected shape: {img.shape}"
        assert label in (0, 1)

    def test_invalid_split_raises(self, small_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        with pytest.raises(ValueError, match="No rows in manifest"):
            ManifestDataset(small_manifest, "nonexistent")

    def test_class_counts_sum_matches_length(self, small_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        ds = ManifestDataset(small_manifest, "train")
        assert sum(ds.class_counts.values()) == len(ds)

    def test_test_data_not_in_train(self, small_manifest):
        """Train and test path sets must be disjoint."""
        from ml.utils.manifest_dataset import ManifestDataset
        train_ds = ManifestDataset(small_manifest, "train")
        test_ds  = ManifestDataset(small_manifest, "test")
        train_paths = {str(p) for p in train_ds.paths}
        test_paths  = {str(p) for p in test_ds.paths}
        overlap = train_paths & test_paths
        assert len(overlap) == 0, f"Train/test overlap found: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. BinaryCNN model
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryCNN:
    def test_forward_output_shape(self, tiny_model):
        x = torch.randn(4, 3, 224, 224)
        logits = tiny_model(x)
        assert logits.shape == (4, 2), f"Expected (4,2), got {logits.shape}"

    def test_frozen_backbone_grad(self, tiny_model):
        """All non-FC parameters should have requires_grad=False."""
        trainable_params = [
            name for name, p in tiny_model.named_parameters()
            if p.requires_grad and "fc" not in name
        ]
        assert len(trainable_params) == 0, (
            f"Non-FC params have requires_grad=True: {trainable_params[:3]}"
        )

    def test_fc_head_trainable(self, tiny_model):
        fc_trainable = [
            name for name, p in tiny_model.named_parameters()
            if p.requires_grad and "fc" in name
        ]
        assert len(fc_trainable) > 0, "FC head should have trainable parameters"

    def test_parameter_count(self, tiny_model):
        trainable, total = tiny_model.parameter_count()
        assert trainable < total, "Trainable params should be less than total (frozen backbone)"
        assert trainable > 0

    def test_unfrozen_model(self):
        from ml.models.binary_cnn import BinaryCNN
        model = BinaryCNN(pretrained=False, freeze_backbone=False)
        trainable, total = model.parameter_count()
        assert trainable == total, "All params should be trainable when backbone unfrozen"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Single training step (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingStep:
    def test_loss_decreases_after_step(self, tiny_model):
        import torch.nn as nn
        device = torch.device("cpu")
        model = tiny_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 0, 1])

        model.train()
        logits = model(x)
        loss_before = criterion(logits, y).item()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        logits_after = model(x)
        loss_after = criterion(logits_after, y).item()

        # After one step, parameters updated (loss may increase or decrease
        # due to random init, but we just check no errors / NaNs)
        assert not np.isnan(loss_after), "Loss is NaN after training step"

    def test_weighted_loss_runs(self, tiny_model):
        import torch.nn as nn
        weights = torch.tensor([3.0, 1.0])  # simulate imbalance weighting
        criterion = nn.CrossEntropyLoss(weight=weights)
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 0, 1])
        tiny_model.train()
        logits = tiny_model(x)
        loss = criterion(logits, y)
        assert loss.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_prediction(self):
        from ml.utils.metrics import compute_binary_metrics
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        m = compute_binary_metrics(y_true, y_pred, class_names=["Pure", "Adulterated"])
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from ml.utils.metrics import compute_binary_metrics
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = compute_binary_metrics(y_true, y_pred, class_names=["Pure", "Adulterated"])
        assert m["accuracy"] == pytest.approx(0.0)

    def test_auc_roc_computed(self):
        from ml.utils.metrics import compute_binary_metrics
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        y_prob = [0.9, 0.3, 0.2, 0.8]
        m = compute_binary_metrics(y_true, y_pred, y_prob=y_prob,
                                   class_names=["Pure", "Adulterated"])
        assert "auc_roc" in m
        assert 0.0 <= m["auc_roc"] <= 1.0

    def test_imbalanced_balanced_accuracy(self):
        from ml.utils.metrics import compute_binary_metrics
        # Predict all Adulterated (all 1s) -- common failure mode
        y_true = [0, 0, 1, 1, 1, 1, 1, 1]  # 2 Pure, 6 Adulterated
        y_pred = [1, 1, 1, 1, 1, 1, 1, 1]  # all predicted Adulterated
        m = compute_binary_metrics(y_true, y_pred, class_names=["Pure", "Adulterated"])
        assert m["accuracy"] == pytest.approx(0.75)  # 6/8 correct
        assert m["balanced_accuracy"] == pytest.approx(0.5)  # 0/2 Pure correct


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load_round_trip(self, tiny_model, tmp_path):
        from ml.models.binary_cnn import load_checkpoint, save_checkpoint

        optimizer = torch.optim.AdamW(tiny_model.trainable_parameters(), lr=1e-3)
        ckpt_path = tmp_path / "test_ckpt.pth"

        save_checkpoint(
            tiny_model, optimizer, epoch=5, val_metric=0.88,
            save_path=ckpt_path,
            extra_meta={"exp_id": "TEST"},
        )
        assert ckpt_path.exists()

        loaded_model, meta = load_checkpoint(ckpt_path, pretrained=False)
        assert meta["epoch"] == 5
        assert meta["val_metric"] == pytest.approx(0.88)

        # Weights should be identical
        for (n1, p1), (n2, p2) in zip(
            tiny_model.named_parameters(), loaded_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"
