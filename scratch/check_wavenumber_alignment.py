"""
Scratch Script: check_wavenumber_alignment.py
Verifies exact wavenumber alignment between Pure and Adulterated sheets.
"""

from pathlib import Path
import pandas as pd
import numpy as np

excel_path = Path('data/dataset_III_ftir/raw/FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx')
df_adult = pd.read_excel(excel_path, sheet_name='RcP_Adulterted_with_SudanIV')
df_pure = pd.read_excel(excel_path, sheet_name='Pure_RcP')

# Adulterated wavenumbers
wn_adult = df_adult.iloc[1:, 0].values.astype(float)

# Pure wavenumbers
pure_raw = df_pure.iloc[3:, 0].values.astype(float)

print(f"Adulterated WNs count: {len(wn_adult)}, range: [{wn_adult.min():.4f}, {wn_adult.max():.4f}]")
print(f"Pure WNs count       : {len(pure_raw)}, range: [{pure_raw.min():.4f}, {pure_raw.max():.4f}]")

# Check if pure_raw is a contiguous slice of wn_adult
start_idx = np.where(np.isclose(wn_adult, pure_raw[0]))[0]
end_idx = np.where(np.isclose(wn_adult, pure_raw[-1]))[0]

print(f"Pure start matches Adulterated index: {start_idx}")
print(f"Pure end matches Adulterated index  : {end_idx}")

if len(start_idx) > 0 and len(end_idx) > 0:
    s_idx = start_idx[0]
    e_idx = end_idx[0] + 1
    slice_adult = wn_adult[s_idx:e_idx]
    diff = np.abs(slice_adult - pure_raw)
    max_diff = diff.max()
    print(f"Max difference between sliced Adulterated WNs and Pure WNs: {max_diff:.8f}")
    print(f"Are wavenumbers 100% perfectly aligned? {max_diff < 1e-4}")
