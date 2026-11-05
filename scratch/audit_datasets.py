"""
Scratch Script: audit_datasets.py
Performs precise data counting and discrepancy check for Dataset I and Dataset II.
"""

from pathlib import Path
import pandas as pd

data_dir = Path("data")
raw_ds1 = data_dir / "raw" / "dataset_I"
raw_ds2 = data_dir / "raw" / "dataset_II"

print("=========================================================")
print("  DATASET I AUDIT")
print("=========================================================")
ds1_split = data_dir / "dataset_I_split.csv"
if ds1_split.exists():
    df1 = pd.read_csv(ds1_split)
    print(f"Manifest path: {ds1_split}")
    print(f"Columns: {list(df1.columns)}")
    print(f"Total rows in manifest: {len(df1)}")
    print("Split breakdown:")
    print(df1['split'].value_counts())
    label_col = 'class_name' if 'class_name' in df1.columns else ('label' if 'label' in df1.columns else df1.columns[1])
    print(f"Label breakdown ({label_col}):")
    print(df1[label_col].value_counts())

# Disk audit DS I
if raw_ds1.exists():
    ds1_files = [p for p in raw_ds1.rglob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    print(f"Total image files found on disk in DS I: {len(ds1_files)}")

print("\n=========================================================")
print("  DATASET II AUDIT")
print("=========================================================")
ds2_inv = Path("reports/metrics/dataset_II_inventory.csv")
if ds2_inv.exists():
    df2 = pd.read_csv(ds2_inv)
    print(f"Inventory path: {ds2_inv}")
    print(f"Total rows in inventory: {len(df2)}")
    print("\nClass breakdown in inventory:")
    class_counts = df2['class_name'].value_counts()
    print(class_counts)
    print(f"\nSum of class counts in inventory: {class_counts.sum()}")
    
    if 'is_excluded_train_duplicate' in df2.columns:
        dups = df2[df2['is_excluded_train_duplicate'] == True]
        print(f"\nExcluded train-side duplicates count: {len(dups)}")
        print(f"Active valid images count (inventory minus train-side duplicates): {len(df2) - len(dups)}")

ds2_split = data_dir / "dataset_II_phase3_splits.csv"
if ds2_split.exists():
    df2_split = pd.read_csv(ds2_split)
    print(f"\nPhase 3 Split Manifest: {ds2_split}")
    print(f"Total rows in split manifest: {len(df2_split)}")
    print("Split breakdown in split manifest:")
    print(df2_split['split'].value_counts())
    print("Class breakdown in split manifest:")
    print(df2_split['class_name'].value_counts())
    print(f"Sum of class counts in split manifest: {df2_split['class_name'].value_counts().sum()}")

# Disk audit DS II
if raw_ds2.exists():
    ds2_files = [p for p in raw_ds2.rglob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    print(f"\nTotal image files found on disk in DS II raw folder: {len(ds2_files)}")
    class_dirs = [d for d in raw_ds2.iterdir() if d.is_dir()]
    print(f"Total class subdirectories in DS II: {len(class_dirs)}")
    disk_counts = {}
    for d in sorted(class_dirs):
        imgs = [p for p in d.rglob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
        disk_counts[d.name] = len(imgs)
    
    print("\nPer-class disk counts:")
    total_disk_imgs = 0
    for cls, count in disk_counts.items():
        print(f"  {cls}: {count}")
        total_disk_imgs += count
    print(f"Total disk images sum across 19 subdirectories: {total_disk_imgs}")

print("\n---------------------------------------------------------")
print("  INVESTIGATING 5,338 vs 5,335 DISCREPANCY REASON")
print("---------------------------------------------------------")
# Let's search previous reports to see where 5338 and 5335 came from
reports_dir = Path("reports")
for rfile in reports_dir.glob("*.md"):
    content = rfile.read_text(encoding="utf-8", errors="ignore")
    if "5338" in content or "5335" in content or "5,338" in content or "5,335" in content:
        print(f"Found mentions in {rfile.name}:")
        for line in content.splitlines():
            if "5338" in line or "5335" in line or "5,338" in line or "5,335" in line:
                print(f"  {line}")
