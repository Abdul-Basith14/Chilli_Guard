"""
Scratch Script: find_ftir.py
Searches for FTIR excel files in data/
"""

from pathlib import Path

data_path = Path('data')
all_files = list(data_path.rglob('*'))

print("All files in data/:")
for f in sorted(all_files):
    if f.is_file():
        print(f"  {f}")
