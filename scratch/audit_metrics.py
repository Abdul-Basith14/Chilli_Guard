"""
Scratch Script: audit_metrics.py
Verifies metric calculations and reproducibility across all Phase 3 experiment JSONs.
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

exp_dir = Path("experiments")

def check_json(json_path):
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
    with open(json_path) as f:
        data = json.load(f)
    print(f"\n--- Checking {json_path.name} ---")
    print(f"Exp ID: {data.get('exp_id', 'N/A')}")
    acc = data.get('val_accuracy') or data.get('accuracy')
    macro_f1 = data.get('val_macro_f1') or data.get('f1_macro')
    top5 = data.get('val_top5_accuracy') or data.get('top5_accuracy')
    
    print(f"Reported Acc: {acc}")
    print(f"Reported Macro-F1: {macro_f1}")
    print(f"Reported Top5 Acc: {top5}")
    
    # Check if predictions exist in payload
    if "y_true" in data and "y_pred" in data:
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        calc_acc = accuracy_score(y_true, y_pred)
        calc_bal_acc = balanced_accuracy_score(y_true, y_pred)
        calc_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        calc_weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        print("Recalculated from y_true & y_pred:")
        print(f"  Accuracy:         {calc_acc:.6f} (matches: {abs(calc_acc - acc) < 1e-5})")
        print(f"  Balanced Acc:     {calc_bal_acc:.6f}")
        print(f"  Macro-F1:         {calc_macro_f1:.6f} (matches: {abs(calc_macro_f1 - macro_f1) < 1e-5})")
        print(f"  Weighted-F1:      {calc_weighted_f1:.6f}")

for exp in ["A", "B", "C", "D"]:
    check_json(exp_dir / f"phase3_exp_{exp}.json")

check_json(exp_dir / "phase3_winner_exp_C_test_results.json")
