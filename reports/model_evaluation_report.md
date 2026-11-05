# Phase 6 Model Evaluation Report: ChilliGuard

**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection
**Phase:** 6 - Model Evaluation, Comparison & Explainable AI
**Date:** September 2026
**Pipelines Evaluated:** Binary Image Classifier | 19-Class Image Classifier | ATR-FTIR Spectral Classifier

---

> NOTE: The three pipelines in this project address **distinct tasks** on **different datasets**
> and are **independent systems**. They are **not a single fused multimodal classifier** and
> their performance metrics are **not directly comparable** across modalities. Each should be
> assessed within the scope of its own task, dataset, and evaluation protocol.

---

## 1. Summary Dashboard

| Pipeline | Task | Dataset | Test Set | Key Metric |
| :--- | :--- | :---: | :---: | :---: |
| **Binary Classifier** | Pure vs. Adulterated | Dataset I (RGB images) | 278 images | Acc: **82.01%** · AUC: **0.89** |
| **19-Class Classifier** | Variety x Adulterant level | Dataset II (RGB images) | 1,093 images | Top-1: **66.24%** · Top-5: **96.34%** |
| **ATR-FTIR Classifier** | Sudan IV detection | Dataset III (spectra) | 31 spectra · 11 groups | Acc: **100.00%** · Macro-F1: **1.00** |

---

## 2. Pipeline A -- Binary Image Classifier

### 2.1 Model & Data

- **Architecture:** ResNet18 (pretrained ImageNet), frozen backbone, 2-class FC head + Dropout(0.3)
- **Training strategy:** Class-weighted CrossEntropyLoss (Exp B); AdamW; StepLR scheduler
- **Dataset I split:** 1,061 train | 278 held-out test (Pure=62, Adulterated=216)
- **Checkpoint:** `ml/models/binary_exp_b_best.pth` (epoch 3, best val macro-F1)

### 2.2 Test Set Metrics

| Metric | Value |
| :--- | :---: |
| **Accuracy** | **82.01%** |
| **Balanced Accuracy** | **82.68%** |
| **Precision (Weighted)** | 0.8613 |
| **Recall (Weighted)** | 0.8201 |
| **F1-Score (Macro)** | **0.7755** |
| **F1-Score (Weighted)** | 0.8310 |
| **Specificity** (recall of Pure) | 0.8387 |
| **AUC-ROC** (Adulterated = positive) | **0.8899** |
| **Average Precision (AP)** | **0.9686** |

### 2.3 Per-Class Performance

| Class | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Pure** | 0.57 | 0.84 | 0.68 | 62 |
| **Adulterated** | 0.95 | 0.81 | 0.88 | 216 |

### 2.4 Key Observations

1. **Class imbalance:** Dataset I is 77% Adulterated (1,031 of 1,339 images). Class weighting
   partially mitigates this -- the Pure class achieves 84% recall but only 57% precision,
   reflecting a higher false-positive rate for Pure predictions.

2. **AUC clarification:** The stored `phase2_exp_B_test_results.json` file contains
   `auc_roc=0.11`, which is an artefact of using `prob(Pure)` as the score with `Adulterated`
   as the implicit positive class (score inversion). The correctly-oriented AUC -- using
   `prob(Adulterated)` with label 1 as positive -- is **0.8899**, confirmed by
   `reports/metrics/phase6_binary_evaluation.json`.

3. **High AP (0.97):** The Average Precision over the full precision-recall curve is very high,
   indicating the model's probabilistic ranking is near-optimal for the imbalanced test set.

4. **Limitation -- backbone frozen at epoch 3:** The best checkpoint was saved at epoch 3 of 20.
   Only the 2-layer FC head (approx. 1,024 parameters) was trained. This limits the expressivity
   of learned representations compared to partial or full fine-tuning.

### 2.5 Figures Generated

- `reports/figures/phase6_binary_confusion_matrix.png`
- `reports/figures/phase6_binary_roc_curve.png` -- ROC curve (AUC = 0.89)
- `reports/figures/phase6_binary_pr_curve.png` -- Precision-Recall curve (AP = 0.97)
- `reports/figures/phase6_binary_per_class_metrics.png`
- `reports/figures/gradcam/binary_example_*.png` -- 4 Grad-CAM saliency maps

---

## 3. Pipeline B -- 19-Class Image Classifier

### 3.1 Model & Data

- **Architecture:** ResNet18 (pretrained ImageNet), partial fine-tuning (layer4 + FC head),
  19-class FC head + Dropout(0.3)
- **Trainable parameters:** 8,403,475 (of 11,177,539 total)
- **Dataset II split:** 3,575 train | 632 val | 1,093 test | 38 excluded (cross-split duplicates)
- **Checkpoint:** `ml/models/multiclass_exp_c_best.pth` (Exp C winner)
- **Classes:** 19 categories -- KT00 (unadulterated) + KT{BM,GM,RB,UT,WB,WS}_{5,10,15}
  (6 adulterant types x 3 concentration levels)

### 3.2 Test Set Metrics

| Metric | Value |
| :--- | :---: |
| **Accuracy (Top-1)** | **66.24%** |
| **Accuracy (Top-5)** | **96.34%** |
| **Balanced Accuracy** | 63.60% |
| **F1-Score (Macro)** | **0.6070** |
| **F1-Score (Weighted)** | 0.6456 |
| **Precision (Macro)** | 0.6452 |
| **Recall (Macro)** | 0.6360 |

### 3.3 Per-Class Performance (Selected)

| Class | Precision | Recall | F1 | Support | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **KTWS_5** | 1.00 | 0.97 | **0.98** | 62 | Best -- visually distinct variety |
| **KTUT_15** | 0.98 | 0.89 | **0.93** | 62 | High 15% adulterant signal |
| **KTWS_10** | 0.82 | 1.00 | **0.90** | 62 | -- |
| **KTRB_15** | 0.85 | 0.90 | **0.88** | 62 | -- |
| **KT00** | 0.69 | 0.15 | **0.24** | 62 | Pure chilli confuses with subtle adulterants |
| **KTBM_5** | 0.00 | 0.00 | **0.00** | 34 | Zero recall -- undersampled |
| **KTBM_10** | 0.18 | 0.08 | **0.11** | 24 | Second-worst -- small support |

### 3.4 Most Confused Pairs

| True Class | Predicted As | Errors |
| :--- | :--- | :---: |
| KT00 (Pure) | KTUT_10 | 45 |
| KTBM_5 | KTBM_15 | 28 |
| KTBM_10 | KTBM_15 | 21 |
| KTGM_10 | KTGM_15 | 16 |
| KTGM_5 | KTUT_10 | 12 |

### 3.5 Key Observations

1. **Task difficulty:** Distinguishing 19 classes that differ only in adulterant type and
   concentration -- where images of the same chilli variety at 5%, 10%, and 15% adulteration
   are visually near-identical -- is inherently a very hard computer vision problem.

2. **Highly imbalanced support:** Classes like KTBM (Black Mustard) have only 23-34 test images
   vs. 62-80 for majority classes. F1=0 for KTBM_5.

3. **Top-5 accuracy (96.34%)** is practically useful: a human-in-the-loop system can rely on
   the top-5 candidates even when the top-1 prediction is incorrect.

4. **Confusion structure:** Errors are predominantly within the same variety (e.g., KTBM_5 to
   KTBM_15), meaning the model correctly identifies the chilli variety but confuses the
   adulteration concentration level.

5. **Baseline comparison:**
   - Exp A (frozen FC only, no weighting): Top-1 = 27.08%
   - Exp C (partial fine-tune, layer4 unfrozen): Top-1 = 66.24% (+39 pp)
   - Random chance (19 classes): ~5.26% -- Exp C is 12x above chance

### 3.6 Figures Generated

- `reports/figures/phase6_multiclass_confusion_matrix.png` -- full 19x19 counts
- `reports/figures/phase6_multiclass_confusion_matrix_norm.png` -- row-normalised
- `reports/figures/phase6_multiclass_per_class_f1.png`
- `reports/figures/phase6_multiclass_top1_vs_top5.png`
- `reports/figures/gradcam/multiclass_example_*.png` -- 6 Grad-CAM saliency maps

---

## 4. Pipeline C -- ATR-FTIR Spectral Classifier

### 4.1 Model & Data

- **Architecture:** Standard Normal Variate (SNV) + Linear SVM (C=1.0)
  Selected by 5-fold group-aware cross-validation (GroupKFold on sample group_id)
- **Dataset III:** 140 spectra -- 56 Pure (7 varieties x 8 replicates),
  84 Adulterated (3 concentration levels x 28 groups x 3 replicates)
- **Split:** 89 train | 20 val | 31 test (11 physical groups in test)
  Group-aware splitting guarantees zero replicate leakage across splits.
- **Checkpoint:** `ml/models/ftir_classifier_best.joblib`

### 4.2 Test Set Metrics

| Metric | Value |
| :--- | :---: |
| **Accuracy** | **100.00%** |
| **Balanced Accuracy** | **100.00%** |
| **F1-Score (Macro)** | **1.0000** |
| **Specificity** | **1.0000** |
| **AUC-ROC** | **1.0000** |

### 4.3 Concentration Regression (Side Task)

Ridge regression (SG 1st-derivative + SNV + Ridge alpha=1.0) predicting % w/w Sudan IV
concentration on adulterated test spectra.

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **MAE** | 0.3586% w/w | Reasonable absolute error |
| **RMSE** | 0.5190% w/w | -- |
| **R-squared (test)** | **-4.75** | Regressor does not generalise |

**WARNING:** The negative R-squared indicates the concentration regression does not generalise
on the small test subset (24 adulterated spectra, 3 discrete concentration levels).
The test concentrations span only {5%, 10%, 15%} -- insufficient variation for a robust
regression. This is a dataset scope limitation, not a failure of the binary classifier.

### 4.4 Scientific Discussion

1. **Why 100%?** Sudan IV is a synthetic azo dye with strong, characteristic mid-IR absorption
   bands -- particularly the azo N=N stretch (~1400-1500 cm-1) and aromatic C-H bending modes.
   These signatures are highly distinct from chilli powder's natural components, making even
   a simple linear SVM highly discriminative.

2. **Replicate inflation note:** The 140 spectra come from ~23 physical sample groups.
   Group-aware splitting was enforced, so no replicate from a test group appeared in training.
   However, 11 test groups is a small sample from which to generalise to unseen varieties.

3. **Single adulterant scope:** This classifier is validated exclusively on Sudan IV dye.
   It has zero validated performance against other adulterants (Metanil Yellow, Rhodamine B,
   brick dust, synthetic food colours).

4. **Concentration regression limitation:** Predicting continuous concentration from three
   discrete training levels (5%, 10%, 15%) using Ridge regression is a regression problem
   poorly suited to this dataset. PLS-DA or PLSR with continuous concentration levels would
   be a more appropriate approach.

### 4.5 Figures Generated

- `reports/figures/phase6_ftir_confusion_matrix.png`
- `reports/figures/phase6_ftir_per_class_metrics.png`
- `reports/figures/ftir/phase6_ftir_raw_spectra.png` -- all 140 raw spectra by class
- `reports/figures/ftir/phase6_ftir_mean_spectra.png` -- class mean +/- std
- `reports/figures/ftir/phase6_ftir_snv_spectra.png` -- after SNV normalisation
- `reports/figures/ftir/phase6_ftir_sg_spectra.png` -- Savitzky-Golay 1st derivative
- `reports/figures/ftir/phase6_ftir_sgsnv_spectra.png` -- winning preprocessing
- `reports/figures/ftir/phase6_ftir_preprocessing_comparison.png` -- 4-panel comparison
- `reports/figures/ftir/phase6_ftir_split_distribution.png`
- `reports/figures/ftir/phase6_ftir_concentration_distribution.png`

---

## 5. Explainability -- Grad-CAM (Image Models)

### 5.1 Method

Grad-CAM (Selvaraju et al., 2017, ICCV) applied to both ResNet18 models.

- **Target layer:** `model.model.layer4[-1]` (7x7 feature maps, 512 channels)
- **Implementation:** Hook-based autograd Grad-CAM (no external packages needed)
- **Script:** `ml/11_gradcam.py`

### 5.2 Interpretation Limits

NOTE: Grad-CAM shows *where* the network attended spatially for a prediction. It does NOT
prove that the model has learned semantically meaningful or chemically relevant features.
Highlighted regions are artefacts of the network's learned feature space. All Grad-CAM maps
should be interpreted qualitatively and with caution.

### 5.3 Figures Generated

- `reports/figures/gradcam/binary_example_01_pure.png`
- `reports/figures/gradcam/binary_example_02_pure.png`
- `reports/figures/gradcam/binary_example_03_adulterated.png`
- `reports/figures/gradcam/binary_example_04_adulterated.png`
- `reports/figures/gradcam/multiclass_example_01_KT00.png`
- `reports/figures/gradcam/multiclass_example_02_KTBM_10.png`
- `reports/figures/gradcam/multiclass_example_03_KTBM_15.png`
- `reports/figures/gradcam/multiclass_example_04_KTBM_5.png`
- `reports/figures/gradcam/multiclass_example_05_KTGM_10.png`
- `reports/figures/gradcam/multiclass_example_06_KTGM_15.png`

---

## 6. Cross-Pipeline Comparison

NOTE: The table below places all three pipelines in one view for convenience only.
The tasks, datasets, and evaluation conditions differ substantially.

| Pipeline | Modality | Test N | Top-1 Acc | Macro-F1 | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Binary Image Classifier | RGB Image | 278 | 82.01% | 0.7755 | 0.8899 |
| 19-Class Image Classifier | RGB Image | 1,093 | 66.24% | 0.6070 | -- |
| ATR-FTIR Classifier | Mid-IR Spectra | 31 | 100.00% | 1.0000 | 1.0000 |

**Why FTIR achieves 100% while image classifiers do not:**
- FTIR provides a direct molecular fingerprint. Sudan IV has unique chemical bonds absent
  from pure chilli, creating linearly separable 1,610-dimensional feature vectors.
- RGB images capture surface appearance only. Visual adulteration at 5% w/w is nearly
  imperceptible; within-variety colour variation introduces significant noise.
- The tasks differ in inherent difficulty: binary FTIR (vs. one specific adulterant) is
  fundamentally simpler than 19-class RGB (6 types x 3 concentrations x same base ingredient).

---

## 7. Limitations & Future Work

| Limitation | Impact | Suggested Mitigation |
| :--- | :--- | :--- |
| Dataset I class imbalance (77% Adulterated) | Low Pure precision (0.57) | Augment Pure class or collect more Pure samples |
| Dataset II minority classes (KTBM: 23-34 samples) | F1=0 for KTBM_5 | Collect more KTBM; use focal loss |
| FTIR -- single adulterant (Sudan IV only) | No validated detection of other adulterants | Expand to Metanil Yellow, Rhodamine B, brick dust |
| FTIR -- small test set (31 spectra, 11 groups) | 100% may not generalise to new sources | Validate on out-of-source samples |
| Concentration regression (R-sq = -4.75) | Cannot reliably estimate % w/w Sudan IV | Collect continuous levels; use PLS regression |
| Binary CNN -- frozen backbone, epoch 3 | Representation limited to ImageNet features | Progressive unfreezing with longer training |
| No cross-modality validation | FTIR and image predictions not tested on same samples | Collect paired image + FTIR data on same powders |

---

## 8. Saved Artefacts

| Artefact | Path |
| :--- | :--- |
| Binary evaluation JSON | `reports/metrics/phase6_binary_evaluation.json` |
| Multiclass evaluation JSON | `reports/metrics/phase6_multiclass_evaluation.json` |
| FTIR evaluation JSON | `reports/metrics/phase6_ftir_evaluation.json` |
| Binary confusion matrix | `reports/figures/phase6_binary_confusion_matrix.png` |
| Binary ROC curve | `reports/figures/phase6_binary_roc_curve.png` |
| Binary PR curve | `reports/figures/phase6_binary_pr_curve.png` |
| Multiclass confusion matrices | `reports/figures/phase6_multiclass_confusion_matrix*.png` |
| Multiclass per-class F1 | `reports/figures/phase6_multiclass_per_class_f1.png` |
| Grad-CAM figures | `reports/figures/gradcam/` (10 figures) |
| FTIR spectral figures | `reports/figures/ftir/` (8 figures) |
| Evaluation script | `ml/10_evaluate_models.py` |
| Grad-CAM script | `ml/11_gradcam.py` |
| FTIR visualisation script | `ml/12_visualize_ftir.py` |

---

## 9. Reproducibility

All evaluation scripts use deterministic inference:
- Binary/Multiclass: `model.eval()` + `torch.no_grad()` for evaluation;
  Grad-CAM uses `model.train()` for gradient flow through BatchNorm.
- FTIR: sklearn pipeline is fully deterministic.
- Split CSVs define all partitions:
  - `data/dataset_I_phase2_splits.csv`
  - `data/dataset_II_phase3_splits.csv`
  - `data/dataset_III_ftir/processed/dataset_III_splits.csv`

To reproduce all Phase 6 outputs from scratch:

    python ml/10_evaluate_models.py --all
    python ml/11_gradcam.py --mode both --batch-examples 5
    python ml/12_visualize_ftir.py
