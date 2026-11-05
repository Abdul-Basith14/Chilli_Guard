"""
ChilliGuard ML -- Binary CNN Model (Phase 2).

Wraps a pretrained ResNet18 backbone (torchvision) with a two-class
classification head.  The backbone is frozen by default so the network
trains efficiently on CPU.

Architecture choice:
  ResNet18 was selected over ResNet50 for CPU training:
  - 11M vs 25M parameters
  - ~3x faster per epoch on CPU
  - Still provides excellent transfer-learning representations
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class BinaryCNN(nn.Module):
    """
    ResNet18 backbone with a two-class (Pure / Adulterated) head.

    Args:
        num_classes:      Number of output classes (default: 2).
        pretrained:       Load ImageNet weights (default: True).
        freeze_backbone:  Freeze all backbone parameters, training only
                          the final fully-connected layer (default: True).
        dropout_p:        Dropout probability before the FC layer (0 = off).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features: int = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        # Unfreeze the newly added head regardless of freeze_backbone flag
        for param in backbone.fc.parameters():
            param.requires_grad = True

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> tuple[int, int]:
        """Returns (trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


def save_checkpoint(
    model: BinaryCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metric: float,
    save_path: Path,
    extra_meta: Optional[dict] = None,
) -> None:
    """Save model + optimiser state + metadata to a .pth checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_metric": val_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": extra_meta or {},
    }
    torch.save(payload, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    num_classes: int = 2,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> tuple[BinaryCNN, dict]:
    """
    Load a BinaryCNN from a checkpoint file.

    Returns:
        (model, meta_dict)  where meta_dict contains epoch, val_metric, meta.
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BinaryCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "epoch": payload.get("epoch"),
        "val_metric": payload.get("val_metric"),
        "meta": payload.get("meta", {}),
    }
    return model, meta
