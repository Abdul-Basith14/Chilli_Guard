"""
Scratch Script: test_inference_cli.py
Tests single image inference CLI for both binary and multiclass modes.
"""

import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location("inference_module", Path("ml/06_inference.py"))
inf_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf_mod)

def find_first_img(p):
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        matches = list(p.rglob(ext))
        if matches:
            return matches[0]
    return None

sample_ds1 = find_first_img(Path('data'))
print(f"Sample image found: {sample_ds1}")

if sample_ds1:
    print("\n=== TESTING BINARY INFERENCE ===")
    inf_mod.run_binary_inference(sample_ds1)

    print("\n=== TESTING MULTICLASS INFERENCE ===")
    inf_mod.run_multiclass_inference(sample_ds1)
