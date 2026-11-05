# Phase 4 Dataset Audit: ATR-FTIR Spectral Dataset

**Project:** Hybrid Machine Learning Models for Multi-Modal Food Authentication and Adulteration Detection  
**Dataset:** Dataset III (ATR-FTIR Spectroscopy)  
**Workbook:** `FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx`  

---

## 1. Executive Summary & Workbook Accounting

- **Sheets Discovered:** `RcP_Adulterted_with_SudanIV, Pure_RcP`
- **Total FTIR Spectra Extracted:** **140**
  - **Pure Red Chilli Powder Spectra:** **56** (40.0%)
  - **Sudan IV Adulterated Spectra:** **84** (60.0%)
- **Wavenumber Range:** **700.7395 cm⁻¹** to **3699.3828 cm⁻¹**
- **Spectral Points per Spectrum:** **1610**
- **Spectral Resolution / Step:** **1.8637 cm⁻¹**
- **Corrupted / Invalid Spectra:** **0** (100% valid numerical spectra)

---

## 2. Spectral Grid & Axis Alignment Reconciliation

| Sheet | Raw Wavenumber Range | Raw Point Count | Aligned Range | Aligned Point Count | Alignment Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `RcP_Adulterted_with_SudanIV` | 650.42 to 3999.43 cm⁻¹ | 1,798 | 700.74 to 3699.38 cm⁻¹ | 1,610 | 100% Exact Sub-grid Match |
| `Pure_RcP` | 700.74 to 3699.38 cm⁻¹ | 1,610 | 700.74 to 3699.38 cm⁻¹ | 1,610 | Baseline Grid |

> [!NOTE]
> Both sheets share an identical wavenumber step size ($\Delta\nu \approx 1.8637\text{ cm}^{-1}$). Slicing the Adulterated sheet to $[700.74, 3699.38]\text{ cm}^{-1}$ retains 100% of the mid-IR fingerprint region while creating an exact $140 \times 1610$ feature matrix across both classes.

---

## 3. Sample Breakdown by Variety and Concentration

### Pure Varieties (7 Varieties x 8 Replicates = 56 Spectra)
| Variety Code | Variety Name / Description | Spectra Count |
| :--- | :--- | :---: |
| `KT` | Khammam Teja | 8 |
| `JP` | Jwala Powder | 8 |
| `WH` | Wonder Hot | 8 |
| `UT` | Uttara / Teja | 8 |
| `BM` | Byadgi Medium | 8 |
| `CH` | Chittoor / Cold House | 8 |
| `GM` | Guntur Medium | 8 |
| **Total Pure** | **7 Varieties** | **56** |

### Adulterated Samples (Sudan IV Adulterant)
| Variety Code | Concentration (% w/w) | Physical Batches | Replicates per Batch | Total Spectra |
| :--- | :---: | :---: | :---: | :---: |
| `GM` (Guntur Medium) | 0.1%, 0.2%, 0.4%, 0.6%, 1.0% | 5 | 6 | 30 |
| `UT` (Uttara / Teja) | 0.1%, 0.2%, 0.4%, 0.6%, 1.0% | 5 | 6 | 30 |
| `BM` (Byadgi Medium) | 0.1%, 0.2%, 0.4%, 0.6% | 4 | 6 | 24 |
| **Total Adulterated** | **0.1% to 1.0% w/w** | **14** | **6** | **84** |

---

## 4. Group-Aware Split Partitioning

To prevent data leakage caused by repeated spectral scans of the same physical sample batch, splitting was performed at the **`group_id` physical sample level**.

| Split Partition | Pure Spectra | Adulterated Spectra | Total Spectra | Physical Sample Groups |
| :--- | :---: | :---: | :---: | :---: |
| `train` | 41 | 48 | 89 | 49 |
| `val` | 8 | 12 | 20 | 10 |
| `test` | 7 | 24 | 31 | 11 |
| **Total** | **56** | **84** | **140** | **70** |

---

## 5. Summary & Next Steps

1. **Dataset Integrity:** 100% verified without missing values or corrupted spectra.
2. **Processed Artifacts Exported:**
   - `data/dataset_III_ftir/processed/X_raw.npy` (140 x 1610 array)
   - `data/dataset_III_ftir/processed/wavenumbers.npy` (1610 wavenumber grid)
   - `data/dataset_III_ftir/processed/dataset_III_splits.csv` (Sample manifest with group-aware splits)
3. **Ready for Training:** Proceed to `ml/08_train_ftir.py` for spectral preprocessing (SG smoothing, 1st/2nd derivative, SNV) and model comparison (SVM, Random Forest, Gradient Boosting).