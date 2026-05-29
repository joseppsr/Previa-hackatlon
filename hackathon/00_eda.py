"""
EDA rápido: visualización de los 6 índices, correlaciones, y detección de Index_D.
Ejecutar: python 00_eda.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from utils import load_data, INDEX_COLS, find_ghost_source, train_val_split

data = load_data()
train = data["train_indices"][INDEX_COLS]
print(f"Train shape: {train.shape}")
print(train.describe().round(2))

# ---- Plot all indices ----
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
for ax, col in zip(axes.flatten(), INDEX_COLS):
    ax.plot(train.index, train[col], lw=0.7)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("eda_indices.png", dpi=100)
print("Saved: eda_indices.png")

# ---- Correlation matrix ----
corr = train.corr()
print("\nCorrelation matrix:")
print(corr.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(INDEX_COLS, rotation=45, ha="right")
ax.set_yticklabels(INDEX_COLS)
plt.colorbar(im, ax=ax)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=100)
print("Saved: eda_correlation.png")

# ---- Ghost detection ----
print("\nSearching for Index_D source ...")
find_ghost_source(train, target_col="Index_D", max_lag=30)

# ---- Stationarity (ADF) ----
try:
    from statsmodels.tsa.stattools import adfuller
    print("\nADF test (p-value):")
    for col in INDEX_COLS:
        result = adfuller(train[col].dropna())
        print(f"  {col}: p={result[1]:.4f} {'STATIONARY' if result[1] < 0.05 else 'non-stationary'}")
except ImportError:
    print("statsmodels not installed, skipping ADF test.")

# ---- Macro factors ----
if "train_macro_factors" in data:
    macro = data["train_macro_factors"]
    print(f"\nMacro factors shape: {macro.shape}")
    print(macro.head())

# ---- Network metrics ----
if "train_network_metrics" in data:
    net = data["train_network_metrics"]
    print(f"\nNetwork metrics shape: {net.shape}")
    print(net.head())

plt.show()
