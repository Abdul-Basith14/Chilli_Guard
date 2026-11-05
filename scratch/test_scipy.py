"""
Scratch Script: test_scipy.py
"""
import scipy
from scipy.signal import savgol_filter
import numpy as np

x = np.random.randn(10, 1610)
sg1 = savgol_filter(x, window_length=15, polyorder=2, deriv=1, axis=1)
print(f"Scipy version: {scipy.__version__}")
print(f"SG1 shape: {sg1.shape}")
