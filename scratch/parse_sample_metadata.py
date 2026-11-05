"""
Scratch Script: parse_sample_metadata.py
Parses sample metadata from column headers for both Pure and Adulterated sheets.
"""

from pathlib import Path
import pandas as pd
import re

excel_path = Path('data/dataset_III_ftir/raw/FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx')
df_adult = pd.read_excel(excel_path, sheet_name='RcP_Adulterted_with_SudanIV')
df_pure = pd.read_excel(excel_path, sheet_name='Pure_RcP')

sample_cols_adult = list(df_adult.columns[1:])
sample_cols_pure = list(df_pure.columns[1:])

print(f"Total Adulterated Spectra (columns): {len(sample_cols_adult)}")
print(f"Total Pure Spectra (columns)       : {len(sample_cols_pure)}")
print(f"Total Combined FTIR Spectra        : {len(sample_cols_adult) + len(sample_cols_pure)}")

print("\n--- Parsing Pure Samples ---")
pure_samples = []
for col in sample_cols_pure:
    # Variety code is letters prefix (KT, JP, WH, UT, BM, CH, GM)
    variety_code = re.match(r"([A-Za-z]+)", col).group(1)
    pure_samples.append({
        "column_header": col,
        "sheet": "Pure_RcP",
        "label": 0, # Pure
        "class_name": "Pure",
        "variety": variety_code,
        "adulterant": "None",
        "concentration_percent": 0.0,
        "physical_sample_id": col, # Each pure column is an independent pure sample/batch
    })

df_pure_meta = pd.DataFrame(pure_samples)
print("Pure Variety Breakdown:")
print(df_pure_meta['variety'].value_counts())

print("\n--- Parsing Adulterated Samples ---")
adult_samples = []
for col in sample_cols_adult:
    # Example format: GMSd_0.1(a), BMSd_0.6(b), UTSd_1.0(a).2, BMSd_0.6()
    # Extract variety prefix (GM, UT, BM)
    var_match = re.match(r"^([A-Z]{2})", col)
    variety = var_match.group(1) if var_match else "UNKNOWN"
    
    # Extract concentration (0.1, 0.2, 0.4, 0.6, 1.0)
    conc_match = re.search(r"(\d+\.\d+)", col)
    conc = float(conc_match.group(1)) if conc_match else None
    
    # Physical sample group: Variety + Concentration (e.g. GM_Sd_0.1)
    sample_group = f"{variety}_Sd_{conc}"
    
    adult_samples.append({
        "column_header": col,
        "sheet": "RcP_Adulterted_with_SudanIV",
        "label": 1, # Adulterated
        "class_name": "Adulterated",
        "variety": variety,
        "adulterant": "Sudan IV",
        "concentration_percent": conc,
        "physical_sample_id": sample_group,
    })

df_adult_meta = pd.DataFrame(adult_samples)
print("Adulterated Variety Breakdown:")
print(df_adult_meta['variety'].value_counts())
print("\nAdulterated Concentration Breakdown (% w/w):")
print(df_adult_meta['concentration_percent'].value_counts().sort_index())
print("\nPhysical Sample Groups (Variety x Concentration):")
print(df_adult_meta['physical_sample_id'].value_counts())

print("\nChecking for any nulls in metadata:")
print("Pure nulls:", df_pure_meta.isna().sum().to_dict())
print("Adulterated nulls:", df_adult_meta.isna().sum().to_dict())
