"""
Scratch Script: inspect_ftir_file.py
Detailed inspection of FTIR dataset Excel file.
"""

from pathlib import Path
import pandas as pd

raw_dir = Path("data/dataset_III_ftir/raw")
excel_files = list(raw_dir.glob("*.xlsx"))
print("Excel files in raw_dir:", [f.name for f in excel_files])

target_file = excel_files[0] if excel_files else None
if target_file:
    print(f"\nTarget File: {target_file}")
    xl = pd.ExcelFile(target_file)
    print("Sheet names:", xl.sheet_names)
    
    for sheet in xl.sheet_names:
        df = pd.read_excel(target_file, sheet_name=sheet)
        print(f"\n================ Sheet: '{sheet}' ================")
        print(f"Shape: {df.shape} (rows x columns)")
        print("Column names (first 15):", list(df.columns[:15]))
        print("Column names (last 5):", list(df.columns[-5:]))
        print("First 5 rows:")
        print(df.iloc[:5, :10])
        print("Summary of data types / NaNs:")
        print(f"Total NaNs in sheet: {df.isna().sum().sum()}")
