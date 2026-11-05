"""
ChilliGuard ML -- Multiclass CNN Model for Dataset II (Phase 3).

Wraps a pretrained ResNet18 backbone (torchvision) with a 19-class
classification head. The backbone is frozen by default for efficient CPU training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MulticlassCNN(nn.Module):
    """
    ResNet18 backbone with a 19-class classification head.

    Args:
        num_classes:      Number of output classes (default: 19).
        pretrained:       Load ImageNet weights (default: True).
        freeze_backbone:  Legacy boolean flag (True -> fine_tune_mode="frozen", False -> "full").
        fine_tune_mode:   Fine-tuning strategy: "frozen", "partial" (unfreeze layer4), or "full".
        dropout_p:        Dropout probability before the FC layer (0.3).
    """

    def __init__(
        self,
        num_classes: int = 19,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        fine_tune_mode: Optional[str] = None,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if fine_tune_mode is None:
            fine_tune_mode = "frozen" if freeze_backbone else "full"
        fine_tune_mode = fine_tune_mode.lower()
        if fine_tune_mode not in ("frozen", "partial", "full"):
            raise ValueError(f"Invalid fine_tune_mode '{fine_tune_mode}'. Must be 'frozen', 'partial', or 'full'.")

        self.fine_tune_mode = fine_tune_mode

        # Set parameter trainable status according to strategy
        for param in backbone.parameters():
            param.requires_grad = False

        if fine_tune_mode == "full":
            for param in backbone.parameters():
                param.requires_grad = True
        elif fine_tune_mode == "partial":
            # Unfreeze layer4
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        in_features: int = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        for param in backbone.fc.parameters():
            param.requires_grad = True

        self.model = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def get_parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float = 1e-4) -> list[dict]:
        """Build differential learning rate parameter groups for PyTorch optimizers."""
        if self.fine_tune_mode == "partial":
            return [
                {"params": [p for p in self.model.layer4.parameters() if p.requires_grad], "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": [p for p in self.model.fc.parameters() if p.requires_grad], "lr": head_lr, "weight_decay": weight_decay},
            ]
        elif self.fine_tune_mode == "full":
            backbone_params = [p for name, p in self.model.named_parameters() if not name.startswith("fc") and p.requires_grad]
            head_params = [p for name, p in self.model.named_parameters() if name.startswith("fc") and p.requires_grad]
            return [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
            ]
        else:
            return [
                {"params": self.trainable_parameters(), "lr": head_lr, "weight_decay": weight_decay},
            ]

    def parameter_count(self) -> tuple[int, int]:
        """Returns (trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total



def save_multiclass_checkpoint(
    model: MulticlassCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metric: float,
    class_to_idx: dict[str, int],
    save_path: Path,
    extra_meta: Optional[dict] = None,
) -> None:
    """Save multiclass model + optimiser state + class mapping to a .pth checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "val_metric": val_metric,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_to_idx),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": extra_meta or {},
    }
    torch.save(payload, save_path)


def load_multiclass_checkpoint(
    checkpoint_path: Path,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> tuple[MulticlassCNN, dict]:
    """
    Load a MulticlassCNN from a checkpoint file.

    Returns:
        (model, payload_dict)
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = payload.get("num_classes", 19)

    model = MulticlassCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    return model, payload
