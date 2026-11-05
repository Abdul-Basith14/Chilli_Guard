"""
Scratch Script: audit_checkpoint.py
Verifies checkpoint loading for Exp C.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ml.models.multiclass_cnn import load_multiclass_checkpoint, MulticlassCNN

ckpt_path = Path('ml/models/multiclass_exp_c_best.pth')
print(f"Checkpoint exists: {ckpt_path.exists()}")

device = torch.device('cpu')
model, payload = load_multiclass_checkpoint(ckpt_path, device=device)

print(f"Payload keys: {list(payload.keys())}")
print(f"Payload fine_tune_mode: {payload.get('fine_tune_mode')}")
print(f"Payload val_macro_f1: {payload.get('val_macro_f1')}")
print(f"Payload epoch: {payload.get('epoch')}")
print(f"Payload class_to_idx count: {len(payload.get('class_to_idx', {}))}")
print(f"Model fine_tune_mode: {model.fine_tune_mode}")
print(f"Model num_classes: {model.num_classes}")
print(f"Model type: {type(model).__name__}")

dummy_input = torch.randn(1, 3, 224, 224)
model.eval()
with torch.no_grad():
    out = model(dummy_input)

print(f"Output logits shape: {out.shape}")
assert out.shape == (1, 19), f"Expected (1, 19), got {out.shape}"
print("Checkpoint loading & forward pass verification PASSED!")
