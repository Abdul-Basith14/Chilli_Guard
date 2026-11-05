"""
Tests for Phase 3: Multiclass Classification Pipeline (19 Classes).

Covers:
  - Dataset II Manifest loading and 19-class label mapping
  - MulticlassCNN model output shape (batch, 19)
  - Frozen backbone parameter freezing
  - Single training step smoke test (multiclass CrossEntropyLoss)
  - Multiclass metric computation (macro F1, weighted F1, top-5 accuracy)
  - Checkpoint save / load round-trip with class_to_idx dictionary
  - Fine-tuning layer configuration (partial / full mode)
  - Differential learning rate parameter groups
  - Metric reproducibility from saved predictions
  - Split integrity (no train/test overlap)
  - CPU training compatibility for fine-tuning experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Shared fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def small_multiclass_manifest(tmp_path):
    from PIL import Image

    classes = [f"Class_{i:02d}" for i in range(19)]
    rows = []

    for split in ["train", "val", "test"]:
        for idx, cname in enumerate(classes):
            class_dir = tmp_path / split / cname
            class_dir.mkdir(parents=True, exist_ok=True)
            for img_i in range(3):
                img_path = class_dir / f"img_{img_i}.jpg"
                img = Image.new("RGB", (64, 64), color=(idx * 10, 100, 150))
                img.save(img_path)
                rows.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "class_name": cname,
                    "label_idx": idx,
                    "split": split,
                    "leakage_flag": False,
                })

    return pd.DataFrame(rows)


@pytest.fixture()
def multiclass_model():
    from ml.models.multiclass_cnn import MulticlassCNN
    return MulticlassCNN(num_classes=19, pretrained=False, freeze_backbone=True, dropout_p=0.0)


# ── Phase 3 Dataset Tests ───────────────────────────────────────────────────────

class TestPhase3Dataset:
    def test_manifest_discovers_19_classes(self, small_multiclass_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        ds = ManifestDataset(small_multiclass_manifest, "train")
        assert len(ds.class_names) == 19

    def test_train_val_test_disjoint(self, small_multiclass_manifest):
        from ml.utils.manifest_dataset import ManifestDataset
        train_ds = ManifestDataset(small_multiclass_manifest, "train")
        test_ds  = ManifestDataset(small_multiclass_manifest, "test")
        train_paths = {str(p) for p in train_ds.paths}
        test_paths  = {str(p) for p in test_ds.paths}
        assert len(train_paths & test_paths) == 0


# ── MulticlassCNN Architecture Tests ───────────────────────────────────────────

class TestMulticlassCNN:
    def test_forward_output_shape(self, multiclass_model):
        x = torch.randn(4, 3, 224, 224)
        logits = multiclass_model(x)
        assert logits.shape == (4, 19), f"Expected shape (4, 19), got {logits.shape}"

    def test_frozen_backbone_params(self, multiclass_model):
        non_fc_trainable = [
            n for n, p in multiclass_model.named_parameters()
            if p.requires_grad and "fc" not in n
        ]
        assert len(non_fc_trainable) == 0, "Non-FC parameters must be frozen"

    def test_fc_head_trainable(self, multiclass_model):
        fc_trainable = [
            n for n, p in multiclass_model.named_parameters()
            if p.requires_grad and "fc" in n
        ]
        assert len(fc_trainable) > 0, "FC head parameters must be trainable"


# ── Training Step Tests ─────────────────────────────────────────────────────────

class TestMulticlassTrainingStep:
    def test_training_step_executes(self, multiclass_model):
        import torch.nn as nn
        model = multiclass_model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 5, 12, 18])

        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), "Loss should not be NaN"


# ── Metric Tests ────────────────────────────────────────────────────────────────

class TestMulticlassMetrics:
    def test_compute_multiclass_metrics(self):
        from ml.utils.metrics import compute_multiclass_metrics
        classes = [f"C_{i}" for i in range(19)]
        y_true = list(range(19)) + list(range(19))
        y_pred = list(range(19)) + list(range(19))
        top_k = [[i] for i in range(19)] + [[i] for i in range(19)]

        metrics = compute_multiclass_metrics(y_true, y_pred, classes, top_k_preds=top_k)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["top5_accuracy"] == 1.0

    def test_imbalanced_multiclass_metrics(self):
        from ml.utils.metrics import compute_multiclass_metrics
        classes = [f"C_{i}" for i in range(5)]
        y_true = [0, 0, 0, 1, 2, 3, 4]
        y_pred = [0, 0, 0, 0, 0, 0, 0]  # predict majority for all

        metrics = compute_multiclass_metrics(y_true, y_pred, classes)
        assert metrics["accuracy"] == pytest.approx(3 / 7)
        assert metrics["balanced_accuracy"] == pytest.approx(1 / 5)


# ── Checkpoint Tests ────────────────────────────────────────────────────────────

class TestMulticlassCheckpoint:
    def test_save_load_roundtrip(self, multiclass_model, tmp_path):
        from ml.models.multiclass_cnn import load_multiclass_checkpoint, save_multiclass_checkpoint

        optimizer = torch.optim.AdamW(multiclass_model.trainable_parameters(), lr=1e-3)
        ckpt_path = tmp_path / "multiclass_test.pth"
        class_to_idx = {f"Class_{i:02d}": i for i in range(19)}

        save_multiclass_checkpoint(
            model=multiclass_model,
            optimizer=optimizer,
            epoch=3,
            val_metric=0.75,
            class_to_idx=class_to_idx,
            save_path=ckpt_path,
        )
        assert ckpt_path.exists()

        loaded_model, payload = load_multiclass_checkpoint(ckpt_path, pretrained=False)
        assert payload["num_classes"] == 19
        assert payload["class_to_idx"] == class_to_idx
        assert loaded_model.num_classes == 19


# ── Fine-tuning layer configuration tests (Exp C & D) ──────────────────────────

class TestFineTuneModeConfiguration:
    """Tests for the fine_tune_mode parameter introduced for Experiments C & D."""

    def test_frozen_mode_trainable_params(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="frozen")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("fc" in n for n in trainable_names), \
            f"In 'frozen' mode only FC should be trainable, got: {trainable_names}"

    def test_partial_mode_unfreezes_layer4(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="partial")
        layer4_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "layer4" in n]
        assert len(layer4_trainable) > 0, "In 'partial' mode, layer4 params must be trainable"
        layer1_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "layer1" in n]
        assert len(layer1_trainable) == 0, "In 'partial' mode, layer1 params must remain frozen"

    def test_full_mode_all_params_trainable(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="full")
        trainable, total = model.parameter_count()
        assert trainable == total, "In 'full' mode all parameters should be trainable"

    def test_invalid_mode_raises(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        with pytest.raises(ValueError, match="Invalid fine_tune_mode"):
            MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="invalid")

    def test_partial_mode_more_trainable_than_frozen(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        frozen_model  = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="frozen")
        partial_model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="partial")
        frozen_count, _  = frozen_model.parameter_count()
        partial_count, _ = partial_model.parameter_count()
        assert partial_count > frozen_count, \
            "Partial fine-tuning should have more trainable params than frozen"

    def test_full_mode_more_trainable_than_partial(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        partial_model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="partial")
        full_model    = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="full")
        partial_count, _ = partial_model.parameter_count()
        full_count, _    = full_model.parameter_count()
        assert full_count > partial_count, \
            "Full fine-tuning should have more trainable params than partial"


# ── Differential LR parameter group tests ──────────────────────────────────────

class TestDifferentialLearningRates:
    """Tests for the differential LR parameter group builder."""

    def test_frozen_mode_single_group(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="frozen")
        groups = model.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
        assert len(groups) == 1, "Frozen mode should produce 1 param group"
        assert groups[0]["lr"] == 1e-3

    def test_partial_mode_two_groups(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="partial")
        groups = model.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
        assert len(groups) == 2, "Partial mode should produce 2 param groups"
        lrs = {g["lr"] for g in groups}
        assert 1e-4 in lrs, "Backbone LR must be 1e-4"
        assert 1e-3 in lrs, "Head LR must be 1e-3"

    def test_full_mode_two_groups(self):
        from ml.models.multiclass_cnn import MulticlassCNN
        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode="full")
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        assert len(groups) == 2, "Full mode should produce 2 param groups"
        lrs = {g["lr"] for g in groups}
        assert 1e-5 in lrs, "Backbone LR must be 1e-5"
        assert 1e-3 in lrs, "Head LR must be 1e-3"

    def test_no_parameter_duplication(self):
        """Each parameter must appear in exactly one group."""
        from ml.models.multiclass_cnn import MulticlassCNN
        for mode in ("frozen", "partial", "full"):
            model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode=mode)
            groups = model.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
            all_ids = []
            for g in groups:
                all_ids.extend([id(p) for p in g["params"]])
            assert len(all_ids) == len(set(all_ids)), \
                f"Duplicate parameters found in param groups for mode '{mode}'"


# ── CPU training compatibility tests ───────────────────────────────────────────

class TestCPUTrainingCompatibility:
    """Smoke-test that fine-tuning modes can run a forward/backward pass on CPU."""

    @pytest.mark.parametrize("fine_tune_mode", ["frozen", "partial", "full"])
    def test_cpu_forward_backward(self, fine_tune_mode):
        import torch.nn as nn
        from ml.models.multiclass_cnn import MulticlassCNN

        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode=fine_tune_mode)
        model.to("cpu")
        groups = model.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
        optimizer = torch.optim.AdamW(groups)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 18])

        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), f"Loss is NaN for fine_tune_mode='{fine_tune_mode}'"


# ── Metric reproducibility tests ───────────────────────────────────────────────

class TestMetricReproducibility:
    """Verify that metrics computed from saved predictions match re-computed metrics."""

    def test_metrics_reproducible_from_predictions(self):
        from ml.utils.metrics import compute_multiclass_metrics
        rng = np.random.default_rng(42)
        class_names = [f"C_{i}" for i in range(19)]
        y_true = rng.integers(0, 19, size=200).tolist()
        y_pred = rng.integers(0, 19, size=200).tolist()

        m1 = compute_multiclass_metrics(y_true, y_pred, class_names)
        m2 = compute_multiclass_metrics(y_true, y_pred, class_names)

        assert m1["f1_macro"]          == pytest.approx(m2["f1_macro"])
        assert m1["accuracy"]          == pytest.approx(m2["accuracy"])
        assert m1["balanced_accuracy"] == pytest.approx(m2["balanced_accuracy"])


# ── Fine-tuning checkpoint compatibility tests ──────────────────────────────────

class TestFineTuneCheckpointCompatibility:
    """Verify that fine-tuning checkpoints can be saved and loaded for inference."""

    @pytest.mark.parametrize("fine_tune_mode", ["partial", "full"])
    def test_finetune_checkpoint_load(self, fine_tune_mode, tmp_path):
        from ml.models.multiclass_cnn import MulticlassCNN, load_multiclass_checkpoint, save_multiclass_checkpoint

        model = MulticlassCNN(num_classes=19, pretrained=False, fine_tune_mode=fine_tune_mode)
        groups = model.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
        optimizer = torch.optim.AdamW(groups)
        class_to_idx = {f"Class_{i:02d}": i for i in range(19)}
        ckpt_path = tmp_path / f"exp_{fine_tune_mode}.pth"

        save_multiclass_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            val_metric=0.42,
            class_to_idx=class_to_idx,
            save_path=ckpt_path,
            extra_meta={"fine_tune_mode": fine_tune_mode},
        )

        loaded_model, payload = load_multiclass_checkpoint(ckpt_path, pretrained=False)
        assert payload["num_classes"] == 19
        assert payload["meta"]["fine_tune_mode"] == fine_tune_mode

        # Run inference
        loaded_model.eval()
        with torch.no_grad():
            logits = loaded_model(torch.randn(1, 3, 224, 224))
        assert logits.shape == (1, 19), f"Loaded {fine_tune_mode} model output shape wrong"
