"""
Script 09: FTIR Single-Spectrum Inference (Phase 4)

Loads the trained ATR-FTIR model artifact (`ml/models/ftir_classifier_best.joblib`)
and performs single-spectrum inference to detect Sudan IV adulteration and estimate concentration.

Usage:
    # Run on a sample test spectrum
    python ml/09_ftir_inference.py

    # Run on a specific spectrum CSV / Excel / NPY file
    python ml/09_ftir_inference.py --spectrum path/to/spectrum.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.config import cfg
from ml.utils.ftir_transforms import SNVTransformer, SavgolSNVTransformer


def resolve_ftir_checkpoint(custom_ckpt: Path | None = None) -> Path:
    if custom_ckpt and custom_ckpt.exists():
        return custom_ckpt
    default_ckpt = cfg.model_dir / "ftir_classifier_best.joblib"
    if not default_ckpt.exists():
        print(f"[ERROR] FTIR model artifact not found at {default_ckpt}. Run ml/08_train_ftir.py first.")
        sys.exit(1)
    return default_ckpt


def load_spectrum_file(file_path: Path, expected_length: int = 1610) -> np.ndarray:
    """Load a spectrum array from NPY, CSV, or Excel file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        arr = df.values.squeeze()
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        arr = df.values.squeeze()
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .npy, .csv, or .xlsx")

    # Flatten if 2D
    arr = np.array(arr, dtype=float).ravel()

    if len(arr) != expected_length:
        # Interpolate or slice to expected length if needed
        x_orig = np.linspace(0, 1, len(arr))
        x_target = np.linspace(0, 1, expected_length)
        arr = np.interp(x_target, x_orig, arr)

    return arr.reshape(1, -1)


def run_ftir_inference(spectrum_path: Path | None = None, ckpt_path: Path | None = None) -> dict:
    checkpoint = resolve_ftir_checkpoint(ckpt_path)
    payload = joblib.load(checkpoint)

    pipeline = payload["pipeline"]
    model_name = payload.get("model_name", "FTIR Classifier")
    wavenumbers = payload.get("wavenumbers", None)
    conc_pipeline = payload.get("conc_pipeline", None)

    print("\nChilliGuard -- Single-Spectrum ATR-FTIR Inference (Phase 4)")
    print("=" * 65)
    print(f"Checkpoint   : {checkpoint}")
    print(f"Model Name   : {model_name}")

    if spectrum_path is None:
        # Use a sample from processed dataset for demonstration
        proc_dir = cfg.data_root / "dataset_III_ftir" / "processed"
        X_raw = np.load(proc_dir / "X_raw.npy")
        df_splits = pd.read_csv(proc_dir / "dataset_III_splits.csv")
        test_mask = (df_splits['split'] == 'test')
        test_indices = df_splits[test_mask].index.tolist()
        sample_idx = test_indices[0]

        X_input = X_raw[sample_idx:sample_idx+1]
        sample_meta = df_splits.iloc[sample_idx]

        print(f"Input Spectrum: Sample #{sample_meta['sample_id']} (Ground Truth: {sample_meta['class_name']})")
    else:
        print(f"Input Spectrum: {spectrum_path}")
        X_input = load_spectrum_file(spectrum_path, expected_length=len(wavenumbers) if wavenumbers is not None else 1610)

    # Execute Prediction
    preds = pipeline.predict(X_input)
    pred_class_idx = int(preds[0])
    pred_label = "ADULTERATED" if pred_class_idx == 1 else "PURE"

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_input)[0]
        confidence = float(probs[pred_class_idx])
    else:
        probs = [0.0, 1.0] if pred_class_idx == 1 else [1.0, 0.0]
        confidence = 1.0

    # Estimate Sudan IV concentration (% w/w) if adulterated
    conc_est = None
    if pred_class_idx == 1 and conc_pipeline is not None:
        try:
            conc_pred = float(conc_pipeline.predict(X_input)[0])
            conc_est = max(0.0, conc_pred)
        except Exception:
            conc_est = None

    print("\nPrediction Results:")
    print("-" * 65)
    print(f"  Classification : {pred_label}")
    print(f"  Confidence     : {confidence * 100:.2f}%")
    print(f"  Probabilities  :")
    print(f"    - Pure Red Chilli Powder : {probs[0] * 100:.2f}%")
    print(f"    - Sudan IV Adulterated   : {probs[1] * 100:.2f}%")

    if conc_est is not None:
        print(f"  Estimated Sudan IV Conc  : {conc_est:.2f}% w/w")
    print("=" * 65 + "\n")

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "probabilities": {"Pure": float(probs[0]), "Adulterated": float(probs[1])},
        "estimated_concentration_percent": conc_est,
    }


def main():
    parser = argparse.ArgumentParser(description="ChilliGuard FTIR Single-Spectrum Inference")
    parser.add_argument("--spectrum", type=Path, default=None, help="Path to input spectrum file (.csv, .npy, .xlsx)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to custom model joblib checkpoint")
    args = parser.parse_args()

    run_ftir_inference(spectrum_path=args.spectrum, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
