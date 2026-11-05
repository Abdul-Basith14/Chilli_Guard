# ChilliGuard 🌶️
### Unified Food Adulteration Detection System

> Multi-Modal Food Authentication & Adulteration Detection System combining Image Classification (Binary & 19-Class) and ATR-FTIR Spectroscopic Analysis.

---

## 1. Overview

ChilliGuard is an integrated food authentication platform engineered to detect adulteration in red chilli powder (*Capsicum annuum L.*). The system unifies three distinct machine learning prediction pipelines into a cohesive FastAPI backend and React dashboard:

1. **Binary Image Classification** — Detects visual adulteration (Pure vs. Adulterated) using ResNet18.
2. **19-Class Image Classification** — Identifies fine-grained adulterant types and concentration classes using partial backbone fine-tuning.
3. **ATR-FTIR Spectral Classification** — Detects Sudan IV dye adulteration and estimates % w/w concentration from mid-infrared absorption spectra.

> [!NOTE]
> This application integrates three independent prediction pipelines for user selection and comparative evaluation. Predictions from different models are presented separately and do not constitute unvalidated multimodal data fusion.

---

## 2. System Architecture

```
                       ┌──────────────────────────────────────────┐
                       │          ChilliGuard React Frontend      │
                       │ (Mode Selector: Binary, Multiclass, FTIR)│
                       └────────────────────┬─────────────────────┘
                                            │ HTTP POST /api/predict
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │          FastAPI Backend Server          │
                       │     (app/services/model_service.py)      │
                       └─────┬──────────────────┬──────────────┬──┘
                             │                  │              │
              ┌──────────────┴───┐    ┌─────────┴────────┐    ┌┴─────────────────┐
              │ Binary ResNet18  │    │ 19-Class ResNet18│    │ ATR-FTIR Pipeline│
              │ (Visual Texture) │    │ (Partial Tuning) │    │ (SNV + SG + SVM) │
              └──────────────┬───┘    └─────────┬────────┘    └┬─────────────────┘
                             │                  │              │
                             ▼                  ▼              ▼
                     ml/models/binary_   ml/models/multi     ml/models/ftir_
                     exp_b_best.pth      class_exp_c_best.pth classifier_best.joblib
```

---

## 3. Supported Prediction Modes & Model Descriptions

### Mode A: Binary Image Classification
- **Task:** Classifies red chilli powder images into `Pure` (`C1_PKT`) or `Adulterated` (`C2_AKT`).
- **Architecture:** ResNet18 with custom FC head (Dropout = 0.2, CrossEntropyLoss).
- **Test Accuracy:** **82.01%** (Macro-F1: 0.8197)
- **Model Checkpoint:** `ml/models/binary_exp_b_best.pth`
- **Input Formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff` (Max 10 MB).

### Mode B: 19-Class Image Classification
- **Task:** Identifies specific adulterant mixtures and concentrations across 19 categories.
- **Architecture:** ResNet18 with partial fine-tuning (`layer4` unfrozen, differential learning rates: backbone LR = $10^{-4}$, head LR = $10^{-3}$).
- **Test Top-1 Accuracy:** **66.24%**
- **Test Top-5 Accuracy:** **96.34%**
- **Test Macro-F1:** **0.6070**
- **Model Checkpoint:** `ml/models/multiclass_exp_c_best.pth`
- **Input Formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff` (Max 10 MB).

### Mode C: ATR-FTIR Spectral Classification
- **Task:** Detects synthetic Sudan IV azo dye adulteration (0.1% to 1.0% w/w) and estimates adulterant concentration.
- **Preprocessing:** Standard Normal Variate (SNV) + Savitzky-Golay 1st derivative ($w=15, p=2, d=1$).
- **Classifier:** Linear Support Vector Machine ($\text{C}=1.0$).
- **Concentration Regressor:** Ridge Regression on spectral derivative features.
- **Test Accuracy:** **100.00%** (Macro-F1: 1.0000, 21/21 held-out test spectra correct).
- **Concentration MAE:** **0.0384% w/w** Sudan IV ($R^2 = 0.9998$).
- **Model Checkpoint:** `ml/models/ftir_classifier_best.joblib`
- **Input Formats:** `.csv`, `.xlsx`, `.xls`, `.npy` (Numeric 1610-wavenumber spectral matrices, Max 10 MB).

---

## 4. Dataset Accounting & Evaluation Methodology

| Dataset | Modality | Samples | Classes / Scope | Splitting Methodology |
| :--- | :--- | :---: | :--- | :--- |
| **Dataset I** | RGB Images | 1,339 | 2 Classes (`Pure`, `Adulterated`) | Stratified 70/15/15 train/val/test |
| **Dataset II** | RGB Images | 5,275 | 19 Classes (Adulterants & conc.) | Stratified 70/15/15 train/val/test |
| **Dataset III** | ATR-FTIR Spectra | 140 | Pure (56) vs. Sudan IV (84) | Group-Aware GroupShuffleSplit (Physical batches) |

> [!IMPORTANT]
> **Interpretation of FTIR 100% Test Accuracy:**  
> The 100% held-out test accuracy achieved by the ATR-FTIR classifier must be evaluated within the context of the dataset size ($N = 140$ total spectra across 21 physical sample groups) and scope (Sudan IV dye detection). Group-aware splitting was strictly enforced to ensure all 6 replicate measurements of a physical sample remained in the same fold, preventing replicate data leakage. The high accuracy reflects the strong, distinct mid-IR absorption bands of azo dyes ($-\text{N}=\text{N}-$ stretching near 1400-1500 cm⁻¹) that are absent in pure chilli powder.

---

## 5. API Reference

### Health & Status Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Server health check and version |
| `GET` | `/api/model/status` | Status and architecture details for loaded models |
| `GET` | `/api/model/metrics` | Model performance metrics summary |
| `GET` | `/api/dataset/status` | Dataset inspection status |

### Unified Prediction Endpoint

`POST /api/predict`

**Form Parameters:**
- `file` (*UploadFile*, required): Sample image file or FTIR spectral file.
- `mode` (*string*, default `"binary"`): Selection mode (`"binary"`, `"multiclass"`, or `"ftir"`).

**Example Response (`"binary"` mode):**
```json
{
  "status": "prediction_ready",
  "mode": "binary",
  "predicted_class": "Pure",
  "confidence": 0.9842,
  "class_probabilities": {
    "Pure": 0.9842,
    "Adulterated": 0.0158
  },
  "top5_predictions": null,
  "estimated_concentration_percent": null,
  "message": "Binary prediction complete.",
  "filename": "chilli_sample_01.jpg",
  "file_size_bytes": 142580
}
```

**Example Response (`"ftir"` mode with adulterated sample):**
```json
{
  "status": "prediction_ready",
  "mode": "ftir",
  "predicted_class": "Adulterated",
  "confidence": 1.0,
  "class_probabilities": {
    "Pure": 0.0,
    "Adulterated": 1.0
  },
  "top5_predictions": null,
  "estimated_concentration_percent": 0.42,
  "message": "FTIR spectral prediction complete.",
  "filename": "ftir_spectrum_05.csv",
  "file_size_bytes": 48210
}
```

---

## 6. Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend Setup & Server Start

```powershell
# Navigate to project root
cd c:\01.Myuse\chili_adult

# Activate Python environment & install dependencies
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt

# Start FastAPI server
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- API Docs: `http://127.0.0.1:8000/docs`

### 2. Frontend Setup & Build

```powershell
# Navigate to frontend directory
cd frontend
npm install

# Start Vite development server
npm run dev

# Or build for production
npm run build
```
- Web Application: `http://localhost:5173`

### 3. Running Test Suite

```powershell
# Run complete test suite (Unit, ML, API, and Integration tests)
pytest
```

---

## 7. Known Limitations & Scientific Disclaimers

1. **Binary Image Model Scope:** The binary image model evaluates visual surface texture and color features. It is trained on Dataset I (`C1_PKT` vs `C2_AKT`).
2. **19-Class Image Model Scope:** The 19-class model predicts among its trained fine-grained categories from Dataset II.
3. **FTIR Model Scope:** The FTIR classifier is specifically trained to detect Sudan IV dye adulteration in red chilli powder. Performance on un-modelled adulterants (e.g., Metanil Yellow, Rhodamine B) remains unverified.
4. **Independent Pipelines:** Predictions from the image and spectral models operate independently. No multimodal data fusion claim is made.
5. **Screening Tool Notice:** Model outputs are intended for preliminary screening, quality inspection, and research purposes. Predictions do not constitute definitive chemical verification.
