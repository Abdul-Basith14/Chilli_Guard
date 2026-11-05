# ChilliGuard — System Architecture Documentation

## System Overview

```
Browser / Client  ──→  React Frontend (Vite + Tailwind CSS, Port 5173)
                             │
                             │ HTTP POST /api/predict (FormData: file, mode)
                             ▼
                        FastAPI Backend Server (Port 8000)
                             │
                             ├── GET  /api/health
                             ├── GET  /api/model/status
                             ├── GET  /api/model/metrics
                             └── POST /api/predict
                                      │
                                      ▼
                               ModelService (Singleton)
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      Mode A: Binary           Mode B: Multiclass      Mode C: ATR-FTIR
  (Pure vs. Adulterated)     (19-Class Prediction)  (Sudan IV + % w/w Conc)
              │                       │                       │
      ResNet18 Checkpoint     ResNet18 Checkpoint     SNV + SG + SVM Joblib
(binary_exp_b_best.pth) (multiclass_exp_c_best.pth) (ftir_classifier_best.joblib)
```

## Component Responsibilities

| Component | Path | Responsibilities |
| :--- | :--- | :--- |
| **React Frontend** | `frontend/` | Mode selector (`binary`, `multiclass`, `ftir`), file dropzones, image & spectral previews, confidence progress bars, top-5 rankings, concentration estimation card, disclaimers. |
| **FastAPI Backend** | `backend/` | Input validation (MIME, size, numeric checks), routing, error handling, `ModelService` singleton state management, OpenAPI documentation schemas. |
| **ML Engine** | `ml/` | Inspection, preprocessing (SNV, Savitzky-Golay), model definitions (`BinaryCNN`, `MulticlassCNN`), training scripts (`03_train_binary.py`, `04b_finetune_multiclass.py`, `08_train_ftir.py`), single-sample inference utilities. |
| **Model Artifacts** | `ml/models/` | Trained PyTorch weights (`.pth`) and scikit-learn pipeline joblib artifacts (`.joblib`). |
| **Datasets** | `data/` | Local datasets (Dataset I, Dataset II, Dataset III ATR-FTIR). Excluded from Git tracking. |
| **Reports & Metrics**| `reports/`, `experiments/` | Audit markdown reports, confusion matrix figures, and JSON metric logs. |

## Execution Data Flow

1. User selects classification mode (`binary`, `multiclass`, or `ftir`) and uploads sample file.
2. Frontend sends `multipart/form-data` POST request to `/api/predict` containing `file` and `mode`.
3. Backend validates file extension, MIME type, file size (<10 MB), and numeric data integrity.
4. `ModelService` routes sample to corresponding model pipeline:
   - **Binary/Multiclass:** Evaluates RGB image tensor through ResNet18 and Softmax layers.
   - **FTIR:** Parses 1610-point wavenumber array, applies SNV + Savitzky-Golay 1st derivative, runs Linear SVM classifier and Ridge concentration regressor.
5. JSON response returned: `{ status, mode, predicted_class, confidence, class_probabilities, top5_predictions, estimated_concentration_percent, filename }`.
6. Frontend updates UI with classification verdict, confidence progress bars, top-5 breakdown, concentration card, and model limitation disclaimers.
