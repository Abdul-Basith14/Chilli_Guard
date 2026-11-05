"""
Script 12: FTIR Spectral Visualisation - Phase 6

Generates publication-quality plots of ATR-FTIR spectral data across
all preprocessing stages for the chilli powder dataset.

Figures generated:
  1. Raw spectra — all 140 spectra coloured by class (Pure vs Adulterated)
  2. Mean ± std spectra — class-level mean spectra with standard-deviation ribbons
  3. SNV-preprocessed spectra — after Standard Normal Variate normalisation
  4. SG 1st-derivative spectra — after Savitzky-Golay (window=15, poly=2, deriv=1)
  5. SG + SNV spectra (winning preprocessing) — Savitzky-Golay then SNV normalisation
  6. Preprocessing comparison — 4-panel side-by-side for one example spectrum
  7. Train/val/test split distribution — stacked bar of split composition

Outputs:
  reports/figures/ftir/phase6_ftir_raw_spectra.png
  reports/figures/ftir/phase6_ftir_mean_spectra.png
  reports/figures/ftir/phase6_ftir_snv_spectra.png
  reports/figures/ftir/phase6_ftir_sg_spectra.png
  reports/figures/ftir/phase6_ftir_sgsnv_spectra.png
  reports/figures/ftir/phase6_ftir_preprocessing_comparison.png
  reports/figures/ftir/phase6_ftir_split_distribution.png

Usage:
    python ml/12_visualize_ftir.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors with Unicode chars
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.config import cfg
from ml.utils.ftir_transforms import SNVTransformer, SavgolSNVTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

PALETTE = {
    "pure":  "#2563EB",   # blue
    "adult": "#DC2626",   # red
    "train": "#16A34A",   # green
    "val":   "#EA580C",   # orange
    "test":  "#7C3AED",   # purple
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load processed FTIR arrays and split manifest."""
    proc_dir = cfg.data_root / "dataset_III_ftir" / "processed"
    X_raw = np.load(proc_dir / "X_raw.npy")
    wavenumbers = np.load(proc_dir / "wavenumbers.npy")
    df = pd.read_csv(proc_dir / "dataset_III_splits.csv")
    return X_raw, wavenumbers, df


# ── 1. Raw spectra ────────────────────────────────────────────────────────────

def plot_raw_spectra(X_raw, wn, df, out_dir: Path) -> None:
    """Plot all raw spectra coloured by class."""
    fig, ax = plt.subplots(figsize=(11, 5))

    labels = df["label"].values
    pure_plotted = adult_plotted = False

    for i in range(len(X_raw)):
        label = labels[i]
        color = PALETTE["pure"] if label == 0 else PALETTE["adult"]
        alpha = 0.35
        lw = 0.7
        if label == 0 and not pure_plotted:
            ax.plot(wn, X_raw[i], color=color, alpha=alpha, lw=lw, label="Pure")
            pure_plotted = True
        elif label == 1 and not adult_plotted:
            ax.plot(wn, X_raw[i], color=color, alpha=alpha, lw=lw, label="Sudan IV Adulterated")
            adult_plotted = True
        else:
            ax.plot(wn, X_raw[i], color=color, alpha=alpha, lw=lw)

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance (A.U.)")
    ax.set_title(
        "ATR-FTIR Raw Spectra — All 140 Spectra\n"
        "Red chilli powder: Pure vs. Sudan IV Adulterated",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.invert_xaxis()  # Wavenumber convention: high to low
    ax.grid(True, alpha=0.2)

    # Annotate key IR regions
    _annotate_ir_bands(ax)

    fig.tight_layout()
    path = out_dir / "phase6_ftir_raw_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


def _annotate_ir_bands(ax):
    """Annotate key IR absorption bands as vertical spans."""
    bands = [
        (1400, 1500, "#FEF9C3", "Azo N=N\nstretch"),  # Sudan IV marker
        (2800, 3050, "#DBEAFE", "C-H\nstretch"),
        (1600, 1680, "#DCFCE7", "C=C\naromatic"),
    ]
    for x_lo, x_hi, color, label in bands:
        # Only annotate if in wavenumber range
        ax.axvspan(x_lo, x_hi, alpha=0.18, color=color, label=f"_{label}")
        mid = (x_lo + x_hi) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 0.95,
                label, fontsize=6.5, ha="center", va="top", alpha=0.7,
                rotation=90)


# ── 2. Mean ± std spectra ─────────────────────────────────────────────────────

def plot_mean_spectra(X_raw, wn, df, out_dir: Path) -> None:
    """Plot class-level mean spectra with ±1 std-dev ribbons."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for label_idx, label_name, color in [(0, "Pure", PALETTE["pure"]),
                                          (1, "Sudan IV Adulterated", PALETTE["adult"])]:
        mask = df["label"].values == label_idx
        X_class = X_raw[mask]
        mean = X_class.mean(axis=0)
        std = X_class.std(axis=0)

        ax.plot(wn, mean, color=color, lw=2.0,
                label=f"{label_name} (n={mask.sum()}, mean)")
        ax.fill_between(wn, mean - std, mean + std, alpha=0.18, color=color,
                        label=f"{label_name} ±1 std")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance (A.U.)")
    ax.set_title(
        "ATR-FTIR Mean Spectra ± 1 Standard Deviation\n"
        "Pure vs. Sudan IV Adulterated Chilli Powder",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = out_dir / "phase6_ftir_mean_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 3. SNV-preprocessed spectra ───────────────────────────────────────────────

def plot_snv_spectra(X_raw, wn, df, out_dir: Path) -> None:
    """Plot SNV-normalised spectra coloured by class."""
    snv = SNVTransformer()
    X_snv = snv.transform(X_raw)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = df["label"].values
    pure_p = adult_p = False

    for i in range(len(X_snv)):
        color = PALETTE["pure"] if labels[i] == 0 else PALETTE["adult"]
        if labels[i] == 0 and not pure_p:
            ax.plot(wn, X_snv[i], color=color, alpha=0.35, lw=0.7, label="Pure")
            pure_p = True
        elif labels[i] == 1 and not adult_p:
            ax.plot(wn, X_snv[i], color=color, alpha=0.35, lw=0.7, label="Sudan IV Adulterated")
            adult_p = True
        else:
            ax.plot(wn, X_snv[i], color=color, alpha=0.35, lw=0.7)

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("SNV-Normalised Absorbance")
    ax.set_title(
        "ATR-FTIR Spectra after Standard Normal Variate (SNV) Normalisation\n"
        "SNV removes multiplicative scatter — mean-centres and scales each spectrum",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = out_dir / "phase6_ftir_snv_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 4. Savitzky-Golay 1st Derivative ─────────────────────────────────────────

def plot_sg_spectra(X_raw, wn, df, out_dir: Path) -> None:
    """Plot Savitzky-Golay 1st derivative spectra (before SNV)."""
    X_sg = savgol_filter(X_raw, window_length=15, polyorder=2, deriv=1, axis=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = df["label"].values
    pure_p = adult_p = False

    for i in range(len(X_sg)):
        color = PALETTE["pure"] if labels[i] == 0 else PALETTE["adult"]
        if labels[i] == 0 and not pure_p:
            ax.plot(wn, X_sg[i], color=color, alpha=0.35, lw=0.7, label="Pure")
            pure_p = True
        elif labels[i] == 1 and not adult_p:
            ax.plot(wn, X_sg[i], color=color, alpha=0.35, lw=0.7, label="Sudan IV Adulterated")
            adult_p = True
        else:
            ax.plot(wn, X_sg[i], color=color, alpha=0.35, lw=0.7)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("1st Derivative of Absorbance")
    ax.set_title(
        "ATR-FTIR Spectra — Savitzky-Golay 1st Derivative\n"
        "Window=15, Poly=2 — Enhances spectral resolution and resolves overlapping bands",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = out_dir / "phase6_ftir_sg_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 5. SG + SNV (winning preprocessing) ──────────────────────────────────────

def plot_sgsnv_spectra(X_raw, wn, df, out_dir: Path) -> None:
    """Plot spectra after winning preprocessing: SG 1st-deriv then SNV."""
    sg_snv = SavgolSNVTransformer(window_length=15, polyorder=2, deriv=1)
    X_proc = sg_snv.transform(X_raw)

    # Class-level means for the processed data
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = df["label"].values

    # Left: all spectra
    ax = axes[0]
    pure_p = adult_p = False
    for i in range(len(X_proc)):
        color = PALETTE["pure"] if labels[i] == 0 else PALETTE["adult"]
        if labels[i] == 0 and not pure_p:
            ax.plot(wn, X_proc[i], color=color, alpha=0.3, lw=0.7, label="Pure")
            pure_p = True
        elif labels[i] == 1 and not adult_p:
            ax.plot(wn, X_proc[i], color=color, alpha=0.3, lw=0.7, label="Sudan IV Adulterated")
            adult_p = True
        else:
            ax.plot(wn, X_proc[i], color=color, alpha=0.3, lw=0.7)
    ax.set_title("All Spectra — SG-Deriv + SNV", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Preprocessed Signal")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Right: class means
    ax = axes[1]
    for label_idx, label_name, color in [(0, "Pure", PALETTE["pure"]),
                                          (1, "Sudan IV Adulterated", PALETTE["adult"])]:
        mask = labels == label_idx
        mean = X_proc[mask].mean(axis=0)
        std = X_proc[mask].std(axis=0)
        ax.plot(wn, mean, color=color, lw=2.0, label=f"{label_name} mean")
        ax.fill_between(wn, mean - std, mean + std, alpha=0.15, color=color)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Class Means ± 1 Std — SG-Deriv + SNV", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Preprocessed Signal")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Winning Preprocessing: Savitzky-Golay 1st Derivative + SNV Normalisation\n"
        "Used by the winning ATR-FTIR SVM pipeline",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "phase6_ftir_sgsnv_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 6. Preprocessing comparison (single spectrum) ────────────────────────────

def plot_preprocessing_comparison(X_raw, wn, df, out_dir: Path) -> None:
    """4-panel figure showing one spectrum through all preprocessing stages."""
    # Pick one pure and one adulterated spectrum
    pure_idx = int(np.where(df["label"].values == 0)[0][0])
    adult_idx = int(np.where(df["label"].values == 1)[0][0])

    snv = SNVTransformer()
    X_snv = snv.transform(X_raw)
    X_sg = savgol_filter(X_raw, window_length=15, polyorder=2, deriv=1, axis=1)
    sg_snv = SavgolSNVTransformer(window_length=15, polyorder=2, deriv=1)
    X_sgsnv = sg_snv.transform(X_raw)

    stages = [
        ("Raw Absorbance", X_raw,   "Absorbance (A.U.)"),
        ("After SNV",      X_snv,   "SNV-Normalised"),
        ("After SG 1st Deriv", X_sg, "1st Derivative"),
        ("After SG + SNV\n(Winning Prep)", X_sgsnv, "Preprocessed Signal"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()

    for ax, (title, X_stage, ylabel) in zip(axes_flat, stages):
        ax.plot(wn, X_stage[pure_idx], color=PALETTE["pure"], lw=1.5, label="Pure example")
        ax.plot(wn, X_stage[adult_idx], color=PALETTE["adult"], lw=1.5, label="Adulterated example")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "ATR-FTIR Preprocessing Stages — Single Spectrum Example\n"
        "Pure vs. Sudan IV Adulterated Chilli Powder",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "phase6_ftir_preprocessing_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 7. Split distribution ─────────────────────────────────────────────────────

def plot_split_distribution(df, out_dir: Path) -> None:
    """Stacked bar chart of pure/adulterated spectra across train/val/test splits."""
    splits = ["train", "val", "test"]
    pure_counts  = [sum((df["split"] == s) & (df["label"] == 0)) for s in splits]
    adult_counts = [sum((df["split"] == s) & (df["label"] == 1)) for s in splits]
    totals = [p + a for p, a in zip(pure_counts, adult_counts)]

    x = np.arange(len(splits))
    width = 0.45

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_pure  = ax.bar(x, pure_counts,  width, label="Pure",    color=PALETTE["pure"],  alpha=0.87)
    bars_adult = ax.bar(x, adult_counts, width, bottom=pure_counts,
                        label="Sudan IV Adulterated", color=PALETTE["adult"], alpha=0.87)

    for bar, v in zip(bars_pure, pure_counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"Pure\n{v}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for bar, p_c, v in zip(bars_adult, pure_counts, adult_counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, p_c + v / 2,
                    f"Adult\n{v}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for xi, total in zip(x, totals):
        ax.text(xi, total + 1, f"n={total}", ha="center", va="bottom", fontsize=9)

    # Annotate physical group counts if available
    if "group_id" in df.columns:
        for xi, split in enumerate(splits):
            mask = df["split"] == split
            n_groups = df.loc[mask, "group_id"].nunique()
            ax.text(xi, -8, f"({n_groups} groups)", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of Spectra")
    ax.set_title("ATR-FTIR Dataset Split Distribution\nDataset III — 140 Spectra Total", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "phase6_ftir_split_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── 8. Concentration boxplot ──────────────────────────────────────────────────

def plot_concentration_distribution(df, out_dir: Path) -> None:
    """Boxplot of Sudan IV concentration across splits (adulterated spectra only)."""
    if "concentration_percent" not in df.columns:
        print("  [SKIP] concentration_percent column not found.")
        return

    adult_df = df[df["label"] == 1].copy()
    if adult_df.empty:
        return

    splits = ["train", "val", "test"]
    data_by_split = [adult_df.loc[adult_df["split"] == s, "concentration_percent"].values
                     for s in splits]
    data_by_split = [d for d in data_by_split if len(d) > 0]
    split_labels = [s.capitalize() for s, d in zip(splits, [adult_df.loc[adult_df["split"] == s] for s in splits]) if len(d) > 0]

    concentrations = adult_df["concentration_percent"].unique()
    conc_sorted = sorted(concentrations)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: boxplot by split — use tick_labels (new API ≥3.9) or labels (old API)
    import matplotlib
    _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    _bp_kwargs = {"tick_labels": split_labels} if _mpl_ver >= (3, 9) else {"labels": split_labels}
    axes[0].boxplot(data_by_split, **_bp_kwargs, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE["adult"], alpha=0.5),
                    medianprops=dict(color="black", lw=2))
    axes[0].set_ylabel("Sudan IV Concentration (% w/w)")
    axes[0].set_title("Concentration Distribution by Split\n(Adulterated Spectra Only)", fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: count per concentration level
    conc_counts = adult_df.groupby("concentration_percent").size()
    axes[1].bar(conc_counts.index.astype(str), conc_counts.values,
                color=PALETTE["adult"], alpha=0.8, edgecolor="white")
    for xi, (conc, cnt) in enumerate(conc_counts.items()):
        axes[1].text(xi, cnt + 0.3, str(cnt), ha="center", fontsize=9)
    axes[1].set_xlabel("Concentration Level (% w/w)")
    axes[1].set_ylabel("Number of Spectra")
    axes[1].set_title("Spectra Count per Concentration Level", fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("ATR-FTIR Sudan IV Concentration Analysis — Dataset III",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "phase6_ftir_concentration_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  ChilliGuard Phase 6 — ATR-FTIR Spectral Visualisation")
    print("=" * 65)

    out_dir = cfg.reports_dir / "figures" / "ftir"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {out_dir}")

    X_raw, wn, df = load_data()
    print(f"  Loaded: {X_raw.shape[0]} spectra, {X_raw.shape[1]} wavenumber points")
    print(f"  Wavenumber range: {wn.min():.0f} — {wn.max():.0f} cm⁻¹")
    print(f"  Classes: Pure={sum(df['label']==0)}, Adulterated={sum(df['label']==1)}")

    plot_raw_spectra(X_raw, wn, df, out_dir)
    plot_mean_spectra(X_raw, wn, df, out_dir)
    plot_snv_spectra(X_raw, wn, df, out_dir)
    plot_sg_spectra(X_raw, wn, df, out_dir)
    plot_sgsnv_spectra(X_raw, wn, df, out_dir)
    plot_preprocessing_comparison(X_raw, wn, df, out_dir)
    plot_split_distribution(df, out_dir)
    plot_concentration_distribution(df, out_dir)

    print(f"\n  [Done] All FTIR visualisation figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
