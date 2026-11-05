"""
FTIR Spectral Transformers for Chemometrics (SNV & Savitzky-Golay)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate (SNV) normalization across wavenumber axis per spectrum."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SNVTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std


class SavgolSNVTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay derivative filtering followed by SNV normalization."""

    def __init__(self, window_length: int = 15, polyorder: int = 2, deriv: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SavgolSNVTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        sg = savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, axis=1)
        mean = np.mean(sg, axis=1, keepdims=True)
        std = np.std(sg, axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (sg - mean) / std
