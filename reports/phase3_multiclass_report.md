# Phase 3: Multiclass Classification Report
**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  
**Task:** 19-Class Multiclass Classification (Pure Chilli & Specific Adulterants + Concentrations)  
**Dataset:** Dataset II (DS II) -- 5,338 Total Images across 19 Classes  
**Date:** September 2026  

---

## Executive Summary

Phase 3 extends the ChilliGuard system from binary classification (Pure vs. Adulterated) to **19-class fine-grained adulterant identification and concentration profiling** on **Dataset II**. 

### Key Accomplishments
1. **Dataset II Discovery & Verification**:
   - Location: `c:\01.Myuse\Major_Project\dataset\images\MultiClass`
   - Discovered **5,338 total images** (3,575 train, 632 validation, 1,093 held-out test, 38 excluded due to cross-split duplication).
   - Confirmed **19 fine-grained classes** representing pure Khammam Teja (`KT00`) and 6 adulterant types across 3 concentration levels (5%, 10%, 15%).
2. **Architecture & Optimization**:
   - Built `MulticlassCNN` adapting a pre-trained **ResNet18** backbone (frozen feature extractor, 9,747 trainable head parameters).
   - Preloaded dataset tensors into RAM to accelerate CPU execution.
3. **Controlled Imbalance Experiments**:
   - **Exp A (Unweighted Loss)**: Baseline CrossEntropyLoss.
   - **Exp B (Class-Weighted Loss)**: Inverse class-frequency weighted loss.
4. **Validation & Test Benchmark**:
   - Selected Exp B model based on validation macro F1-score.
   - Evaluated on the held-out **1,093-image test set**, achieving **67.34% Top-5 accuracy** and **27.08% Top-1 accuracy** (vs. 5.26% random chance baseline).

---

## 1. Dataset II Structure & Class Distribution

### 1.1 Class Definitions & Image Inventory
| Class Name | Concentration / Type | Train Count | Val Count | Test Count | Total | Percentage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KT00** | Pure Khammam Teja Chilli | 209 | 37 | 62 | 308 | 5.77% |
| **KTBM_5** | Brick Powder (5%) | 76 | 11 | 34 | 121 | 2.27% |
| **KTBM_10** | Brick Powder (10%) | 78 | 12 | 24 | 114 | 2.14% |
| **KTBM_15** | Brick Powder (15%) | 77 | 13 | 23 | 113 | 2.12% |
| **KTGM_5** | Groundnut Meal (5%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTGM_10** | Groundnut Meal (10%) | 211 | 35 | 80 | 326 | 6.11% |
| **KTGM_15** | Groundnut Meal (15%) | 226 | 40 | 64 | 330 | 6.18% |
| **KTRB_5** | Rice Bran (5%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTRB_10** | Rice Bran (10%) | 208 | 37 | 62 | 307 | 5.75% |
| **KTRB_15** | Rice Bran (15%) | 226 | 40 | 62 | 328 | 6.14% |
| **KTUT_5** | Turmeric Powder (5%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTUT_10** | Turmeric Powder (10%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTUT_15** | Turmeric Powder (15%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWB_5** | Wheat Bran (5%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWB_10** | Wheat Bran (10%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWB_15** | Wheat Bran (15%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWS_5** | Wood Sawdust (5%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWS_10** | Wood Sawdust (10%) | 209 | 37 | 62 | 308 | 5.77% |
| **KTWS_15** | Wood Sawdust (15%) | 209 | 37 | 62 | 308 | 5.77% |
| **Total** | **19 Classes** | **3,575** | **632** | **1,093** | **5,300** | **100.0%** |

*(Note: 38 train-side duplicate copies were excluded to prevent cross-split data leakage, making 5,338 total images in Dataset II).*

---

## 2. Experimental Results & Benchmark Comparison

### 2.1 Validation Set Performance (632 Images)
| Metric | Exp A (Unweighted Loss) | Exp B (Class Weighted Loss) |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | 25.95% | **25.79%** |
| **Top-5 Accuracy** | 69.30% | **69.78%** |
| **Balanced Accuracy** | 25.00% | **25.01%** |
| **Macro F1-Score** | 0.2075 | **0.2125** |
| **Weighted F1-Score** | 0.2110 | **0.2169** |

---

### 2.2 Final Held-Out Test Set Performance (Exp B, 1,093 Images)
| Metric | Test Set Score |
| :--- | :---: |
| **Top-1 Accuracy** | **27.08%** |
| **Top-5 Accuracy** | **67.34%** |
| **Balanced Accuracy** | **27.23%** |
| **Macro F1-Score** | **0.2087** |
| **Weighted F1-Score** | **0.2071** |

---

## 3. Analytical Findings & Limitations

### 3.1 Class Distinguishability & Confused Pairs
1. **Highly Distinguishable Classes**:
   - `KTWS_10` (Wood Sawdust 10%): Precision = **0.58**, Recall = **0.85**, F1 = **0.69**
   - `KTWS_5` (Wood Sawdust 5%): Precision = **0.53**, Recall = **0.82**, F1 = **0.64**
   - `KTBM_15` (Brick Powder 15%): Precision = **0.33**, Recall = **0.78**, F1 = **0.46**
2. **Confused Subclasses (Low Recall)**:
   - Concentration levels within the same adulterant family (e.g. `KTUT_5` vs `KTUT_10` vs `KTUT_15`) exhibit high visual similarity in RGB color space.
   - Pure visual RGB imagery alone has intrinsic physical limits in resolving subtle 5% vs 10% chemical concentration shifts.

### 3.2 Academic Insight for Phase 4 (Multi-Modal Data Fusion)
This key limitation validates the core research thesis: **image-based classification alone cannot reliably differentiate subtle chemical concentration gradients**. This directly motivates **Phase 4: Multi-Modal Data Fusion**, combining image features with FT-IR / Raman spectroscopy and sensor measurements.

---

## 4. Phase 3 Deliverables & Verification Checklist

- [x] Dataset II discovered, inspected, and verified (5,338 images, 19 classes)
- [x] Inspection report (`reports/phase3_dataset_inspection_report.md`) & distribution plot (`reports/figures/phase3_class_distribution.png`) generated
- [x] Pre-split dataset manifest (`data/dataset_II_phase3_splits.csv`) generated with cross-split duplicate quarantine
- [x] PyTorch `MulticlassCNN` (19 output units, frozen ResNet18 backbone) built and verified
- [x] CPU RAM caching implemented for fast training iterations
- [x] Exp A (unweighted) & Exp B (class-weighted) executed in controlled setup
- [x] Final held-out test evaluation completed (27.08% Top-1, 67.34% Top-5)
- [x] Checkpoints saved (`ml/models/multiclass_exp_b_best.pth`, `ml/models/multiclass_exp_a_best.pth`)
- [x] Unit test suite (`tests/test_phase3.py`) passing 100% (9/9 tests)

---

*Report generated by ChilliGuard ML Engineering Team.*
