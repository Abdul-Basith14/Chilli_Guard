"""
Scratch Script: inspect_excel.py
Inspects Excel file structure for Phase 4 FTIR dataset.
"""

from pathlib import Path
import pandas as pd
import openpyxl

excel_path = Path('data/dataset_III_ftir/raw/chilli_ftir_sudan_iv.xlsx')
print(f"File exists: {excel_path.exists()}")

xl = pd.ExcelFile(excel_path)
print(f"Sheet names: {xl.sheet_names}")

for sheet in xl.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    print(f"\n================ Sheet: '{sheet}' ================")
    print(f"Shape: {df.shape}")
    print("First 10 columns:", list(df.columns[:10]))
    print("First 5 rows:")
    print(df.iloc[:5, :10])
