"""
Scratch Script: build_sample_matrix.py
Builds and verifies the full (140, 1610) sample-by-feature matrix X and metadata DataFrame.
"""

from pathlib import Path
import pandas as pd
import numpy as np

excel_path = Path('data/dataset_III_ftir/raw/FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx')
df_adult_raw = pd.read_excel(excel_path, sheet_name='RcP_Adulterted_with_SudanIV')
df_pure_raw = pd.read_excel(excel_path, sheet_name='Pure_RcP')

# 1. Extract Wavenumbers
wn_pure = df_pure_raw.iloc[3:, 0].values.astype(float)
wn_adult_all = df_adult_raw.iloc[1:, 0].values.astype(float)

# Find start and end row indices in Adulterated sheet matching wn_pure
start_idx = np.where(np.isclose(wn_adult_all, wn_pure[0]))[0][0] + 1  # +1 because row 0 is header text
end_idx = start_idx + len(wn_pure)

print(f"Target Wavenumber Grid: {len(wn_pure)} points, [{wn_pure.min():.4f}, {wn_pure.max():.4f}] cm-1")

# 2. Extract Pure Matrix
# Row 3 onwards, columns 1 to 56
pure_intensities = df_pure_raw.iloc[3:, 1:].values.astype(float).T  # Shape: (56, 1610)
print(f"Pure matrix shape: {pure_intensities.shape}")

# 3. Extract Adulterated Matrix
# Rows start_idx to end_idx-1, columns 1 to 84
adult_intensities = df_adult_raw.iloc[start_idx:end_idx, 1:].values.astype(float).T  # Shape: (84, 1610)
print(f"Adulterated matrix shape: {adult_intensities.shape}")

# Combine into X (140, 1610)
X = np.vstack([pure_intensities, adult_intensities])
print(f"Combined X matrix shape: {X.shape}")
print(f"Any NaNs in X? {np.isnan(X).any()}")
print(f"X min intensity: {X.min():.4f}, max intensity: {X.max():.4f}")

# Build Metadata DataFrame
metadata = []

# Pure metadata
for idx, col in enumerate(df_pure_raw.columns[1:]):
    var = col[:2] if col[:2] in ['KT', 'JP', 'WH', 'UT', 'BM', 'CH', 'GM'] else col[:1]
    metadata.append({
        "sample_index": idx,
        "sample_id": col,
        "sheet": "Pure_RcP",
        "label": 0,
        "class_name": "Pure",
        "variety": var,
        "adulterant": "None",
        "concentration_percent": 0.0,
        "group_id": f"Pure_{var}_{col}", # Each pure column is an independent sample batch
    })

# Adulterated metadata
import re
for idx, col in enumerate(df_adult_raw.columns[1:], start=56):
    var_match = re.match(r"^([A-Z]{2})", col)
    var = var_match.group(1) if var_match else "UNKNOWN"
    conc_match = re.search(r"(\d+\.\d+)", col)
    conc = float(conc_match.group(1)) if conc_match else 0.0
    
    # Group ID: physical batch variety + concentration (e.g. GM_Sd_0.1)
    group = f"Adulterated_{var}_{conc}"
    metadata.append({
        "sample_index": idx,
        "sample_id": col,
        "sheet": "RcP_Adulterted_with_SudanIV",
        "label": 1,
        "class_name": "Adulterated",
        "variety": var,
        "adulterant": "Sudan IV",
        "concentration_percent": conc,
        "group_id": group,
    })

df_meta = pd.DataFrame(metadata)
print(f"Metadata rows: {len(df_meta)}")
print("Class breakdown:\n", df_meta['class_name'].value_counts())
print("Group breakdown:\n", df_meta['group_id'].value_counts())
