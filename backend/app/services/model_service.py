"""
Model service — handles loading model weights and running inference.

Supports three distinct classification modes:
1. Binary Image Classification (Pure vs. Adulterated)
2. 19-Class Multiclass Image Classification (Adulterant type & concentration)
3. FTIR Spectral Classification (Pure vs. Sudan IV Adulterated + concentration estimation)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton service for model lifecycle management across all three ChilliGuard prediction modes.
    """

    BINARY_CLASSES: List[str] = ["Pure", "Adulterated"]

    def __init__(self) -> None:
        # Binary model state
        self._binary_model: Optional[Any] = None
        self._binary_loaded: bool = False
        self._binary_error: Optional[str] = None

        # Multiclass model state
        self._multiclass_model: Optional[Any] = None
        self._multiclass_loaded: bool = False
        self._multiclass_error: Optional[str] = None
        self._multiclass_class_to_idx: Dict[str, int] = {}
        self._multiclass_idx_to_class: Dict[int, str] = {}

        # FTIR model state
        self._ftir_payload: Optional[Dict[str, Any]] = None
        self._ftir_loaded: bool = False
        self._ftir_error: Optional[str] = None

    # ── Status Properties ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if at least one model is loaded and ready."""
        return self._binary_loaded or self._multiclass_loaded or self._ftir_loaded

    @property
    def status(self) -> str:
        """Return overall status: 'ready', 'not_trained', or 'loading_failed'."""
        if self.is_ready:
            return "ready"
        if self._binary_error or self._multiclass_error or self._ftir_error:
            return "loading_failed"
        return "not_trained"

    @property
    def status_details(self) -> Dict[str, Any]:
        """Detailed status for health/model endpoints."""
        return {
            "binary": {
                "ready": self._binary_loaded,
                "model_name": "ResNet18-BinaryClassifier" if self._binary_loaded else None,
                "error": self._binary_error,
            },
            "multiclass": {
                "ready": self._multiclass_loaded,
                "model_name": "ResNet18-PartialFineTune-19Class" if self._multiclass_loaded else None,
                "num_classes": len(self._multiclass_idx_to_class),
                "error": self._multiclass_error,
            },
            "ftir": {
                "ready": self._ftir_loaded,
                "model_name": self._ftir_payload.get("model_name", "SNV + SVM (Linear)") if self._ftir_loaded and self._ftir_payload else None,
                "error": self._ftir_error,
            },
        }

    @property
    def status_message(self) -> str:
        if self.is_ready:
            modes = []
            if self._binary_loaded: modes.append("binary")
            if self._multiclass_loaded: modes.append("multiclass")
            if self._ftir_loaded: modes.append("ftir")
            return f"Model service ready for modes: {', '.join(modes)}."
        return "No trained models loaded. Please complete training first."

    @property
    def model_name(self) -> Optional[str]:
        loaded = []
        if self._binary_loaded: loaded.append("ResNet18-BinaryClassifier")
        if self._multiclass_loaded: loaded.append("ResNet18-PartialFineTune-19Class")
        if self._ftir_loaded: loaded.append(self._ftir_payload.get("model_name", "SNV + SVM (Linear)") if self._ftir_payload else "FTIR-Classifier")
        return " + ".join(loaded) if loaded else None

    @property
    def checkpoint_path(self) -> Optional[str]:
        paths = []
        if self._binary_loaded: paths.append(str(settings.binary_model_path))
        if self._multiclass_loaded: paths.append(str(settings.multiclass_model_path))
        if self._ftir_loaded: paths.append(str(settings.ftir_model_path))
        return ", ".join(paths) if paths else None

    @property
    def num_classes(self) -> Optional[int]:
        if self._multiclass_loaded:
            return len(self._multiclass_idx_to_class)
        if self._binary_loaded or self._ftir_loaded:
            return 2
        return None

    # ── Model Loading Lifecycle ────────────────────────────────────────────────

    def try_load(self) -> None:
        """Attempt to load all available model checkpoints from disk."""
        self._load_binary_model()
        self._load_multiclass_model()
        self._load_ftir_model()

    def _load_binary_model(self) -> None:
        model_dir = settings.project_root / "ml" / "models"
        candidate_paths = [
            model_dir / "binary_exp_b_best.pth",
            model_dir / "binary_exp_a_best.pth",
            settings.project_root / settings.binary_model_path,
        ]
        ckpt = next((p for p in candidate_paths if p.exists()), None)
        if not ckpt:
            logger.info("No binary model checkpoint found in '%s'.", model_dir)
            return

        try:
            import sys
            import torch
            project_root = settings.project_root
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from ml.models.binary_cnn import load_checkpoint

            model, _ = load_checkpoint(
                checkpoint_path=ckpt,
                num_classes=2,
                pretrained=False,
                device=torch.device("cpu"),
            )
            model.eval()
            self._binary_model = model
            self._binary_loaded = True
            self._binary_error = None
            logger.info("Binary model loaded successfully from '%s'.", ckpt)
        except Exception as exc:
            self._binary_error = str(exc)
            logger.error("Failed to load binary model: %s", exc)

    def _load_multiclass_model(self) -> None:
        model_dir = settings.project_root / "ml" / "models"
        candidate_paths = [
            model_dir / "multiclass_exp_c_best.pth",
            model_dir / "multiclass_exp_a_best.pth",
            settings.project_root / settings.multiclass_model_path,
        ]
        ckpt = next((p for p in candidate_paths if p.exists()), None)
        if not ckpt:
            logger.info("No multiclass model checkpoint found in '%s'.", model_dir)
            return

        try:
            import sys
            import torch
            project_root = settings.project_root
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from ml.models.multiclass_cnn import load_multiclass_checkpoint

            model, payload = load_multiclass_checkpoint(
                checkpoint_path=ckpt,
                pretrained=False,
                freeze_backbone=False,
                device=torch.device("cpu"),
            )
            class_to_idx = payload.get("class_to_idx", {})
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            self._multiclass_model = model
            self._multiclass_class_to_idx = class_to_idx
            self._multiclass_idx_to_class = idx_to_class
            self._multiclass_loaded = True
            self._multiclass_error = None
            logger.info("Multiclass model loaded successfully from '%s' (%d classes).", ckpt, len(idx_to_class))
        except Exception as exc:
            self._multiclass_error = str(exc)
            logger.error("Failed to load multiclass model: %s", exc)

    def _load_ftir_model(self) -> None:
        model_dir = settings.project_root / "ml" / "models"
        ckpt = model_dir / "ftir_classifier_best.joblib"
        if not ckpt.exists():
            logger.info("No FTIR model artifact found in '%s'.", model_dir)
            return

        try:
            import sys
            project_root = settings.project_root
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))


            # Import required transform classes so joblib can deserialize
            from ml.utils.ftir_transforms import SNVTransformer, SavgolSNVTransformer  # noqa: F401

            payload = joblib.load(ckpt)
            self._ftir_payload = payload
            self._ftir_loaded = True
            self._ftir_error = None
            logger.info("FTIR model loaded successfully from '%s'.", ckpt)
        except Exception as exc:
            self._ftir_error = str(exc)
            logger.error("Failed to load FTIR model: %s", exc)

    # ── Inference Methods ──────────────────────────────────────────────────────

    def predict(self, file_bytes: bytes, filename: str, mode: str = "binary") -> Dict[str, Any]:
        """
        Unified entry point for running inference in 'binary', 'multiclass', or 'ftir' mode.
        """
        mode = (mode or "binary").lower()
        if mode == "binary":
            return self.predict_binary(file_bytes)
        elif mode == "multiclass":
            return self.predict_multiclass(file_bytes)
        elif mode == "ftir":
            return self.predict_ftir(file_bytes, filename)
        else:
            return {
                "status": "error",
                "mode": mode,
                "predicted_class": None,
                "confidence": None,
                "message": f"Unsupported prediction mode '{mode}'. Choose 'binary', 'multiclass', or 'ftir'.",
            }

    def predict_binary(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run binary image classification (Pure vs. Adulterated)."""
        if not self._binary_loaded or self._binary_model is None:
            return {
                "status": "model_not_ready",
                "mode": "binary",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "message": self._binary_error or "Binary classification model is not loaded.",
            }

        try:
            import torch
            from ml.utils.transforms import get_eval_transforms

            transform = get_eval_transforms((224, 224))
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except (Image.UnidentifiedImageError, OSError, ValueError) as img_err:
                logger.warning("Invalid image file: %s", img_err)
                return {
                    "status": "error",
                    "mode": "binary",
                    "predicted_class": None,
                    "confidence": None,
                    "class_probabilities": None,
                    "message": f"Invalid image file: Uploaded file could not be identified or decoded as a valid image ({str(img_err)}).",
                }

            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self._binary_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

            predicted_idx = int(torch.argmax(logits, dim=1).item())
            class_probs = {
                self.BINARY_CLASSES[0]: float(probs[0]),
                self.BINARY_CLASSES[1]: float(probs[1]),
            }

            return {
                "status": "prediction_ready",
                "mode": "binary",
                "predicted_class": self.BINARY_CLASSES[predicted_idx],
                "confidence": float(probs[predicted_idx]),
                "class_probabilities": class_probs,
                "top5_predictions": None,
                "estimated_concentration_percent": None,
                "message": "Binary prediction complete.",
            }
        except Exception as exc:
            logger.error("Binary inference error: %s", exc)
            return {
                "status": "error",
                "mode": "binary",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "message": f"Binary inference failed: {str(exc)}",
            }

    def predict_multiclass(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run 19-class image classification with top-5 predictions."""
        if not self._multiclass_loaded or self._multiclass_model is None:
            return {
                "status": "model_not_ready",
                "mode": "multiclass",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "top5_predictions": None,
                "message": self._multiclass_error or "Multiclass model is not loaded.",
            }

        try:
            import torch
            from ml.utils.transforms import get_eval_transforms

            transform = get_eval_transforms((224, 224))
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except (Image.UnidentifiedImageError, OSError, ValueError) as img_err:
                logger.warning("Invalid image file: %s", img_err)
                return {
                    "status": "error",
                    "mode": "multiclass",
                    "predicted_class": None,
                    "confidence": None,
                    "class_probabilities": None,
                    "top5_predictions": None,
                    "message": f"Invalid image file: Uploaded file could not be identified or decoded as a valid image ({str(img_err)}).",
                }

            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self._multiclass_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

            # Map class probabilities
            class_probs = {}
            top_pairs = []
            for idx, prob in enumerate(probs):
                cls_name = self._multiclass_idx_to_class.get(idx, f"Class_{idx}")
                class_probs[cls_name] = float(prob)
                top_pairs.append((cls_name, float(prob)))

            top_pairs.sort(key=lambda x: x[1], reverse=True)
            predicted_class = top_pairs[0][0]
            confidence = top_pairs[0][1]
            top5 = [{"class_name": cls, "probability": p} for cls, p in top_pairs[:5]]

            return {
                "status": "prediction_ready",
                "mode": "multiclass",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probs,
                "top5_predictions": top5,
                "estimated_concentration_percent": None,
                "message": "Multiclass prediction complete.",
            }
        except Exception as exc:
            logger.error("Multiclass inference error: %s", exc)
            return {
                "status": "error",
                "mode": "multiclass",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "top5_predictions": None,
                "message": f"Multiclass inference failed: {str(exc)}",
            }

    def predict_ftir(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Run ATR-FTIR spectral classification (Pure vs. Sudan IV Adulterated)."""
        if not self._ftir_loaded or self._ftir_payload is None:
            return {
                "status": "model_not_ready",
                "mode": "ftir",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "estimated_concentration_percent": None,
                "message": self._ftir_error or "FTIR spectral model is not loaded.",
            }

        try:
            # Parse spectral vector from file bytes
            X_input, parse_error = self._parse_spectral_bytes(file_bytes, filename)
            if parse_error or X_input is None:
                return {
                    "status": "error",
                    "mode": "ftir",
                    "predicted_class": None,
                    "confidence": None,
                    "class_probabilities": None,
                    "estimated_concentration_percent": None,
                    "message": parse_error or "Failed to parse spectral data.",
                }

            pipeline = self._ftir_payload["pipeline"]
            conc_pipeline = self._ftir_payload.get("conc_pipeline", None)

            preds = pipeline.predict(X_input)
            pred_idx = int(preds[0])
            pred_class = "Adulterated" if pred_idx == 1 else "Pure"

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_input)[0]
                prob_pure = float(probs[0])
                prob_adult = float(probs[1])
                confidence = float(probs[pred_idx])
            else:
                prob_pure = 0.0 if pred_idx == 1 else 1.0
                prob_adult = 1.0 if pred_idx == 1 else 0.0
                confidence = 1.0

            class_probs = {"Pure": prob_pure, "Adulterated": prob_adult}

            conc_est = None
            if pred_idx == 1 and conc_pipeline is not None:
                try:
                    c_pred = float(conc_pipeline.predict(X_input)[0])
                    conc_est = round(max(0.0, c_pred), 2)
                except Exception:
                    conc_est = None

            return {
                "status": "prediction_ready",
                "mode": "ftir",
                "predicted_class": pred_class,
                "confidence": confidence,
                "class_probabilities": class_probs,
                "top5_predictions": None,
                "estimated_concentration_percent": conc_est,
                "message": "FTIR spectral prediction complete.",
            }
        except Exception as exc:
            logger.error("FTIR inference error: %s", exc)
            return {
                "status": "error",
                "mode": "ftir",
                "predicted_class": None,
                "confidence": None,
                "class_probabilities": None,
                "estimated_concentration_percent": None,
                "message": f"FTIR inference failed: {str(exc)}",
            }

    def _parse_spectral_bytes(self, file_bytes: bytes, filename: str, expected_length: int = 1610) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Parse raw spectral file bytes (.csv, .xlsx, .xls, .npy) into a (1, 1610) float array.
        Returns (X_array, error_message).
        """
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".npy":
                arr = np.load(io.BytesIO(file_bytes))
            elif ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
                if df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    arr = df.iloc[:, 1].values
                else:
                    arr = df.values.ravel()
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(io.BytesIO(file_bytes))
                if df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    arr = df.iloc[:, 1].values
                else:
                    arr = df.values.ravel()
            else:
                return None, f"Unsupported file extension '{ext}'. Expected .csv, .xlsx, .xls, or .npy."

            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]  # Strip NaNs

            if len(arr) < 10:
                return None, f"Spectral file contains insufficient numeric data points ({len(arr)} found, minimum 10 required)."

            if len(arr) != expected_length:
                x_orig = np.linspace(0, 1, len(arr))
                x_target = np.linspace(0, 1, expected_length)
                arr = np.interp(x_target, x_orig, arr)

            return arr.reshape(1, -1), None
        except ValueError as val_err:
            return None, f"Invalid spectral content: File contains non-numeric or malformed data ({str(val_err)})."
        except Exception as exc:
            return None, f"Could not parse spectral file '{filename}': {str(exc)}"


# ── Module-level singleton ────────────────────────────────────────────────────
model_service = ModelService()

