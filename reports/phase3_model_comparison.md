# Phase 3 Multiclass Model Comparison

**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  
**Task:** 19-Class Red Chilli Powder Classification (Dataset II)  
**Selection Criterion:** Validation Macro-F1 Score  

---

## Experiment Comparison Table (Validation Set)

| Experiment | Strategy | Backbone LR | Head LR | Trainable Params | Best Epoch | Val Top-1 Acc | Val Top-5 Acc | Val Balanced Acc | **Val Macro-F1** | Val Weighted-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp B** | Frozen backbone + FC head (class-weighted CELoss) | N/A | N/A | N/A | 2 | 25.79% | 69.78% | 25.01% | **0.2125** | 0.2169 |
| **Exp C 🏆** | Partial fine-tune: layer4 + FC head (class-weighted CELoss) | 0.0001 | 0.001 | 8,403,475 | 12 | 72.31% | 97.47% | 68.33% | **0.6748** | 0.7159 |
| **Exp D** | Full fine-tune: entire backbone + FC head (class-weighted CELoss) | 1e-05 | 0.001 | 11,186,259 | 15 | 63.92% | 96.68% | 60.37% | **0.5918** | 0.6305 |

> **Winner: Experiment C** (selected by highest validation macro-F1).

---

## Final Test Evaluation (Held-Out — 1,093 Images)

> The test set was untouched during all training and model selection steps.
> It was evaluated exactly **once** on the selected winning model.

## Final Test Evaluation (Held-Out — 1,093 Images)

> The test set was untouched during all training and model selection steps.
> It was evaluated exactly **once** on the selected winning model.

**Winning Model:** Experiment C  
**Checkpoint:** `multiclass_exp_c_best.pth`  

| Metric | Score |
| :--- | :---: |
| **Top-1 Accuracy** | 66.24% |
| **Top-5 Accuracy** | 96.34% |
| **Balanced Accuracy** | 63.60% |
| **Macro-F1** | 0.6070 |
| **Weighted-F1** | 0.6456 |
| **Macro-Precision** | 0.6452 |
| **Macro-Recall** | 0.6360 |

```
              precision    recall  f1-score   support

        KT00       0.69      0.15      0.24        62
     KTBM_10       0.18      0.08      0.11        24
     KTBM_15       0.29      0.87      0.43        23
      KTBM_5       0.00      0.00      0.00        34
     KTGM_10       0.62      0.56      0.59        80
     KTGM_15       0.49      0.86      0.62        64
      KTGM_5       0.93      0.40      0.56        62
     KTRB_10       0.61      0.44      0.51        62
     KTRB_15       0.85      0.90      0.88        62
      KTRB_5       0.69      0.84      0.76        62
     KTUT_10       0.37      0.74      0.50        62
     KTUT_15       0.98      0.89      0.93        62
      KTUT_5       0.61      0.60      0.60        62
     KTWB_10       0.74      0.55      0.63        62
     KTWB_15       0.65      0.85      0.74        62
      KTWB_5       0.73      0.58      0.65        62
     KTWS_10       0.82      1.00      0.90        62
     KTWS_15       1.00      0.81      0.89        62
      KTWS_5       1.00      0.97      0.98        62

    accuracy                           0.66      1093
   macro avg       0.65      0.64      0.61      1093
weighted avg       0.69      0.66      0.65      1093

```

---

## Methodology Notes

1. **Experiment B (Baseline):** ResNet18 backbone fully frozen. Only the 2-layer FC head (9,747 parameters) was trained.
   This represents the minimal fine-tuning setup where only task-specific linear weights are learned.
2. **Experiment C (Partial Fine-Tuning):** The last residual block (`layer4`, 8,393,728 parameters) was unfrozen,
   in addition to the FC head. A differential LR was applied: `backbone LR=1e-4, head LR=1e-3`.
3. **Experiment D (Full Fine-Tuning):** The entire ResNet18 backbone was unfrozen.
   Differential LR: `backbone LR=1e-5, head LR=1e-3` to prevent catastrophic forgetting.
4. **Common settings:** class-weighted CrossEntropyLoss, CosineAnnealingLR schedule, AdamW optimizer,
   early stopping (patience=5), fixed seed=42, batch size=32, max 15 epochs.
5. **Model selection:** Best model chosen exclusively by **validation macro-F1**, without any test set information.
