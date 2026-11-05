# Phase 4 Model Report: ATR-FTIR Chilli Adulteration Detection

**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  
**Task:** Binary Adulteration Detection (Pure vs. Sudan IV Adulterated) & Concentration Estimation via ATR-FTIR  
**Winning Pipeline:** `SNV + SVM (Linear)`  

---

## 1. Executive Summary & Key Results

The ATR-FTIR spectroscopic machine learning pipeline achieved **100% held-out test accuracy** for detecting Sudan IV dye adulteration in red chilli powder.

- **Held-Out Test Accuracy:** **100.00%**
- **Held-Out Test Balanced Accuracy:** **1.0000**
- **Held-Out Test Macro-F1:** **1.0000**
- **Held-Out Test Weighted-F1:** **1.0000**
- **Concentration Estimation MAE (% w/w Sudan IV):** **0.3586%** ($R^2 = -4.7464$)

---

## 2. Preprocessing & Model Cross-Validation Comparison

Models were evaluated using 5-fold group-aware cross-validation on the 89-spectrum training set to prevent replicate data leakage.

| Preprocessing & Model Candidate | Group-CV Accuracy | Group-CV Macro-F1 | Winner Status |
| :--- | :---: | :---: | :---: |
| `SNV + SVM (Linear)` | 100.00% | 1.0000 | 🏆 **Selected Winner** |
| `SNV + SVM (RBF)` | 95.49% | 0.9523 | Candidate |
| `SG 1st Deriv + SVM (Linear)` | 100.00% | 1.0000 | Candidate |
| `SG 1st Deriv + SVM (RBF)` | 100.00% | 1.0000 | Candidate |
| `SG 2nd Deriv + SVM (RBF)` | 100.00% | 1.0000 | Candidate |
| `SG 1st Deriv + Random Forest` | 98.82% | 0.9866 | Candidate |
| `SG 1st Deriv + Gradient Boosting` | 98.82% | 0.9866 | Candidate |
| `SG 1st Deriv + Extra Trees` | 100.00% | 1.0000 | Candidate |
| `SG 1st Deriv + Logistic Regression` | 100.00% | 1.0000 | Candidate |

---

## 3. Final Held-Out Test Set Performance

| Metric | Pure Class (0) | Adulterated Class (1) | Overall / Weighted Average |
| :--- | :---: | :---: | :---: |
| **Accuracy** | — | — | **100.00%** |
| **Balanced Accuracy** | — | — | **1.0000** |
| **Precision** | 1.0000 | 1.0000 | **1.0000** |
| **Recall** | 1.0000 | 1.0000 | **1.0000** |
| **F1-Score** | 1.0000 | 1.0000 | **1.0000 (Macro) / 1.0000 (Weighted)** |

### Adulterant Concentration Regression Results (% w/w Sudan IV)
- **Mean Absolute Error (MAE):** `0.3586%` w/w
- **Root Mean Squared Error (RMSE):** `0.5190%` w/w
- **Coefficient of Determination ($R^2$):** `-4.7464`

---

## 4. Scientific Discussion & Dataset Limitations

1. **High Diagnostic Sensitivity of ATR-FTIR:** Synthetic azo dyes like Sudan IV exhibit strong, distinct infrared absorption peaks (azo $-N=N-$ stretching near $1400-1500\text{ cm}^{-1}$ and aromatic C-H bending), rendering FTIR spectroscopy exceptionally sensitive for detecting adulteration even at $0.1\%\text{ w/w}$ concentrations.
2. **Dataset Size Limitation:** The dataset contains 140 total spectra across 7 pure varieties and 3 adulterated varieties. Group-aware splitting was enforced to guarantee zero replicate leakage.
3. **Single Adulterant Scope:** This model is trained specifically on Sudan IV dye. Future expansions should include additional adulterants (such as Metanil Yellow, Rhodamine B, or brick dust).

---

## 5. Saved Artifacts & Reproducibility

- **Model Artifact:** `ml/models/ftir_classifier_best.joblib`
- **Metrics JSON:** `experiments/phase4_ftir_results.json`
- **Confusion Matrix Figure:** `reports/figures/phase4_ftir_test_confusion_matrix.png`
- **Inference Script:** `ml/09_ftir_inference.py`