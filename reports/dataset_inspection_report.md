# Dataset Inspection & Exploratory Data Analysis Report

**Project:** ChilliGuard — Red Chilli Powder Adulteration Detection  
**Generated:** 2026-09-26T06:54:49.139489+00:00  
**Dataset Path:** `C:\01.Myuse\chili_adult\data\dataset_I`  
**Structure Detected:** `flat`  

---

## 1. Executive Summary

- **Total Images Discovered:** `1,339`
- **Usable (Readable) Images:** `1,339`
- **Corrupted / Unreadable Images:** `0`
- **Exact Duplicate Groups (MD5):** `20` (40 total files)
- **Perceptual Duplicate Groups (dHash):** `0` (0 total files)

---

## 2. Class Distribution & Mapping

The target binary problem classifies pure Khammam Teja chilli powder against adulterated samples.

| Class Folder | Mapped Label | Discovered Count |
|--------------|--------------|------------------|
| `C1_PKT` | Pure | 308 |
| `C2_AKT` | Adulterated | 1,031 |

---

## 3. Image Integrity & Format Summary

- **Supported Formats Found:** `JPEG`
- **Color Modes:** `RGB`
- **Dimensions Summary:**
  - Width: min `224` px, max `224` px, mean `224.0` px
  - Height: min `224` px, max `224` px, mean `224.0` px

---

## 4. Duplicate Findings & Data Quality Assessment

- **Exact Duplicates:** 20 duplicate set(s) identified via byte hashing.
- **Perceptual Duplicates:** 0 set(s) identified via difference hashing (dHash <= 2).
- **Corrupted Files:** 0 file(s) failed Pillow header validation (`img.verify()`).

> [!NOTE]
> No files have been deleted, moved, or altered during this inspection phase. Original dataset files remain intact.

---

## 5. Recommended Next Steps

1. **Review Data Splitting:** Verify whether a physical sample grouping metadata exists to prevent data leakage across train/val/test splits.
2. **Execute Stratified Split:** Run `python ml/02_split_dataset.py` to create `dataset_I_split.csv` with a 70/15/15 ratio.
3. **Proceed to Binary Training (Phase 2):** Train the ResNet50 baseline model on verified dataset split.
