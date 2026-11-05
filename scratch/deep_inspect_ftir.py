"""
Scratch Script: deep_inspect_ftir.py
Thorough analysis of both sheets in the FTIR Excel dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np

excel_path = Path('data/dataset_III_ftir/raw/FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx')
xl = pd.ExcelFile(excel_path)

print("=================================================================")
print("  SHEET 1: RcP_Adulterted_with_SudanIV")
print("=================================================================")
df_adult = pd.read_excel(excel_path, sheet_name='RcP_Adulterted_with_SudanIV')
print(f"Raw shape: {df_adult.shape}")

# Row 0 has "Wavenumber" and "Intensity" strings
wavenumber_col_name = df_adult.columns[0]
sample_cols_adult = list(df_adult.columns[1:])
print(f"Wavenumber column name: '{wavenumber_col_name}'")
print(f"Number of sample columns: {len(sample_cols_adult)}")

# Row 0: "Wavenumber", "Intensity", ...
print("Row 0 values:", df_adult.iloc[0, :5].values)

# Wavenumbers start from row 1
wavenumbers_adult = df_adult.iloc[1:, 0].values.astype(float)
print(f"Wavenumber range (Adulterated): {wavenumbers_adult.min():.4f} cm-1 to {wavenumbers_adult.max():.4f} cm-1")
print(f"Number of wavenumber points (Adulterated): {len(wavenumbers_adult)}")
print(f"Wavenumber step: min={np.diff(wavenumbers_adult).min():.4f}, max={np.diff(wavenumbers_adult).max():.4f}, mean={np.diff(wavenumbers_adult).mean():.4f}")

# Sample column names breakdown
print("\nAdulterated sample column names:")
print(sample_cols_adult)

print("\n=================================================================")
print("  SHEET 2: Pure_RcP")
print("=================================================================")
df_pure = pd.read_excel(excel_path, sheet_name='Pure_RcP')
print(f"Raw shape: {df_pure.shape}")

# Inspect top 5 rows of df_pure
print("\nFirst 5 rows of Pure sheet:")
print(df_pure.iloc[:5, :10])

# Sample columns in pure sheet
sample_cols_pure = [c for c in df_pure.columns if c != 'Wavenumber']
print(f"\nNumber of sample columns (Pure): {len(sample_cols_pure)}")
print("Pure sample column names:", sample_cols_pure)

# Find where numeric wavenumbers start in Pure sheet
pure_wavenumbers_raw = df_pure.iloc[:, 0].values
print("\nFirst 10 values of column 0 in Pure sheet:", pure_wavenumbers_raw[:10])

# Filter numeric values
pure_wn_numeric = []
start_row_pure = None
for idx, val in enumerate(pure_wavenumbers_raw):
    try:
        fval = float(val)
        if not np.isnan(fval):
            pure_wn_numeric.append(fval)
            if start_row_pure is None:
                start_row_pure = idx
    except (ValueError, TypeError):
        pass

pure_wn_numeric = np.array(pure_wn_numeric)
print(f"Numeric wavenumbers in Pure sheet start at row: {start_row_pure}")
print(f"Wavenumber range (Pure): {pure_wn_numeric.min():.4f} cm-1 to {pure_wn_numeric.max():.4f} cm-1")
print(f"Number of wavenumber points (Pure): {len(pure_wn_numeric)}")
print(f"Wavenumber step: min={np.diff(pure_wn_numeric).min():.4f}, max={np.diff(pure_wn_numeric).max():.4f}, mean={np.diff(pure_wn_numeric).mean():.4f}")

# Wavenumber overlap analysis
overlap_min = max(wavenumbers_adult.min(), pure_wn_numeric.min())
overlap_max = min(wavenumbers_adult.max(), pure_wn_numeric.max())
print(f"\nCommon Wavenumber Overlap Range: {overlap_min:.4f} cm-1 to {overlap_max:.4f} cm-1")
