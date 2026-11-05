"""
Image transform pipelines for ChilliGuard.

Two pipelines are defined:
  - get_train_transforms: augmentation-heavy, used during training
  - get_eval_transforms:  deterministic, used for validation, testing, inference
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T


def get_train_transforms(image_size: tuple[int, int] = (224, 224)) -> Callable:
    """
    Return a transform pipeline with augmentation for training.

    Augmentations chosen to be realistic for food-powder imagery:
    - Random horizontal + vertical flips (powder appearance is orientation-agnostic)
    - Random rotation (±30°)
    - Colour jitter (brightness, contrast, saturation, hue)
    - Gaussian blur (simulates slight focus variation)
    - Normalise with ImageNet mean/std (standard for pre-trained CNNs)
    """
    h, w = image_size
    return T.Compose(
        [
            T.Resize((h, w)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
            ),
        ]
    )


def get_eval_transforms(image_size: tuple[int, int] = (224, 224)) -> Callable:
    """
    Return a deterministic transform pipeline for validation, test, and inference.

    No augmentation — only resize, tensor conversion, and normalisation.
    """
    h, w = image_size
    return T.Compose(
        [
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_inference_transform(image_size: tuple[int, int] = (224, 224)) -> Callable:
    """
    Alias for get_eval_transforms for use in the inference pipeline.
    Returned pipeline is identical to the evaluation pipeline.
    """
    return get_eval_transforms(image_size)
