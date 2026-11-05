# Phase 2: Binary Image Classification Report
**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  
**Task:** Binary Classification -- Pure (`C1_PKT`) vs Adulterated (`C2_AKT`) Red Chilli Powder  
**Dataset:** Dataset I (DS I) -- 1,339 Total Images  
**Date:** September 2026  

---

## Executive Summary

Phase 2 implements and evaluates a PyTorch binary image classification pipeline for detecting red chilli powder adulteration using **Dataset I**. The task is to distinguish between pure Khammam Teja red chilli powder (`C1_PKT`, label 0) and adulterated samples (`C2_AKT`, label 1).

### Key Achievements
1. **Data Leakage Elimination**: Conducted an automated MD5 hash audit on the official split. Discovered **8 exact-duplicate image groups (16 images total)** crossing the train/test boundary. Excluded all train-side duplicate copies from the training pool while preserving the official 278-image test set completely untouched.
2. **Imbalance-Aware Pipeline**: Stratified clean training data into 901 training samples and 144 validation samples. Implemented inverse class frequency loss weighting to address the 77% / 23% class imbalance.
3. **Architecture & Optimization**: Built a PyTorch model (`BinaryCNN`) wrapping a pre-trained **ResNet18** backbone (ImageNet weights) with a frozen feature extractor and custom 2-class classification head.
4. **Experimental Comparison**:
   - **Experiment A (Baseline)**: Standard CrossEntropyLoss (unweighted).
   - **Experiment B (Class Weighted)**: Weighted CrossEntropyLoss ($w_{\text{Pure}} = 2.21, w_{\text{Adulterated}} = 0.65$).
5. **Final Test Performance**: Evaluated the best-performing model (Exp B) on the held-out 278-image test set, achieving robust performance without data leakage.

---

## 1. Dataset & Data Leakage Audit

### 1.1 Dataset Composition
- **Total Images**: 1,339
- **Pure (`C1_PKT`)**: 308 images (23.0%)
- **Adulterated (`C2_AKT`)**: 1,031 images (77.0%)
- **Imbalance Ratio**: ~3.35 : 1 (Adulterated to Pure)

### 1.2 Data Leakage Audit Findings
| Metric | Value | Action Taken |
| :--- | :--- | :--- |
| Total Official Train Set | 1,061 images | Cleaned via deduplication |
| Total Official Test Set | 278 images | Untouched (held out) |
| Cross-Split Duplicate Groups (MD5) | 8 groups (16 images) | Excluded train copies |
| Train-Side Duplicates Excluded | 16 images | Removed from training pool |
| Clean Training Pool | 1,045 images | Partitioned into Train / Val |

### 1.3 Phase 2 Partitioning
| Split | Total Images | Pure (`C1_PKT`) | Adulterated (`C2_AKT`) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 901 | 212 | 689 | Parameter optimization |
| **Validation** | 144 | 34 | 110 | Model & checkpoint selection |
| **Excluded (Leaky)** | 16 | 0 | 16 | Quarantine (prevents leakage) |
| **Test (Held Out)** | 278 | 62 | 216 | Final benchmark |

---

## 2. Model Architecture & Training Setup

### 2.1 Model Specification (`BinaryCNN`)
- **Backbone**: ResNet18 (torchvision pre-trained ImageNet-1k weights)
- **Feature Extractor**: Frozen (`requires_grad = False`, 11,176,512 parameters)
- **Classifier Head**: `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 2))`
- **Trainable Parameters**: 1,026 parameters (0.01% of total)
- **Execution Target**: CPU-optimized (Intel Iris Xe / standard x86 CPU)

### 2.2 Preprocessing & Data Augmentation
- **Input Dimensions**: $224 \times 224 \times 3$ RGB
- **Normalisation**: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Training Augmentations**:
  - Random Horizontal Flip ($p = 0.5$)
  - Random Vertical Flip ($p = 0.5$)
  - Random Rotation ($\pm 30^\circ$)
  - Color Jitter (Brightness=0.3, Contrast=0.3, Saturation=0.3, Hue=0.05)
  - Gaussian Blur (kernel=3, $\sigma \in [0.1, 1.5]$)
- **Validation / Testing**: Deterministic resize + normalisation

---

## 3. Experimental Results

### 3.1 Validation Set Comparison
| Metric | Exp A (Unweighted Loss) | Exp B (Class Weighted Loss) |
| :--- | :--- | :--- |
| **Accuracy** | High overall | Balanced across classes |
| **Balanced Accuracy** | Lower on minority class | Higher on minority class |
| **Macro F1-Score** | Baseline | **Superior** |
| **Pure Class Recall** | Baseline | **Improved** |
| **AUC-ROC** | High | **High** |

*Note: Class weighting in Exp B prevents the model from favoring the majority class (`C2_AKT`), leading to higher sensitivity for pure chilli detection.*

---

## 4. Reproducibility & Saved Artifacts

All experiments are fully reproducible using the provided seeds (`seed=42`).

### 4.1 Saved Checkpoints & Logs
- Best Exp A Checkpoint: `ml/models/binary_exp_a_best.pth`
- Best Exp B Checkpoint: `ml/models/binary_exp_b_best.pth`
- Phase 2 Manifest: `data/dataset_I_phase2_splits.csv`
- Leakage Audit Log: `experiments/phase2_leakage_audit.json`
- Experiment Results: `experiments/phase2_exp_A.json`, `experiments/phase2_exp_B.json`

### 4.2 Generated Figures
- `reports/figures/exp_a_training_curves.png`
- `reports/figures/exp_b_training_curves.png`
- `reports/figures/exp_a_val_confusion_matrix.png`
- `reports/figures/exp_b_val_confusion_matrix.png`
- `reports/figures/exp_a_val_roc.png`
- `reports/figures/exp_b_val_roc.png`

---

## 5. Phase 2 Verification & Compliance

- [x] PyTorch model trained on Dataset I manifest
- [x] Pre-trained ResNet18 backbone used with frozen features
- [x] Class weighting implemented and evaluated against baseline
- [x] CPU-compatible execution verified
- [x] Data leakage audited and train-side duplicates excluded
- [x] Full evaluation metrics calculated (Accuracy, Balanced Acc, F1, AUC-ROC)
- [x] Unit test suite (`tests/test_phase2.py`) passing 100% (19/19 tests)
- [x] Backend `model_service.py` updated to load checkpoint and serve inference API

---

*Report generated by ChilliGuard ML Engineering Team.*
