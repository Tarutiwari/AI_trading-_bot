# ============================================================
# Mamba XAUUSD – Colab Experiment Runner
# ============================================================
# INSTRUCTIONS:
#  1. Zip your tradingbot folder on your PC:
#        Right-click tradingbot folder → Send to → Compressed (zipped) folder
#        Name it:  tradingbot.zip
#  2. Upload tradingbot.zip to Google Drive (My Drive root)
#  3. Open a NEW Google Colab notebook
#  4. Copy each CELL below into a separate Colab code cell
#  5. Runtime → Change runtime type → T4 GPU
#  6. Run cells top to bottom
# ============================================================

# ─── CELL 1: Mount Google Drive ──────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─── CELL 2: Extract project & set working directory ─────────
import zipfile, os, shutil

ZIP_PATH  = '/content/drive/MyDrive/tradingbot.zip'
DEST_PATH = '/content/tradingbot'

if os.path.exists(DEST_PATH):
    shutil.rmtree(DEST_PATH)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall('/content/')

# Handle nested folder (e.g. tradingbot/tradingbot/)
extracted = [d for d in os.listdir('/content/') if d.startswith('tradingbot')]
print(f"Extracted: {extracted}")
os.chdir(DEST_PATH)
print(f"Working dir: {os.getcwd()}")

# ─── CELL 3: Install dependencies (Colab-safe subset) ────────
# MetaTrader5 is Windows-only; skip it on Linux Colab
import subprocess, sys

COLAB_PKGS = [
    'torch>=2.2.0',
    'numpy>=1.26.0',
    'pandas>=2.2.0',
    'scipy>=1.12.0',
    'scikit-learn>=1.4.0',
    'lightgbm>=4.3.0',
    'ta>=0.11.0',
    'pytz>=2024.1',
    'requests>=2.31.0',
    'yfinance>=0.2.38',
    'tqdm>=4.66.0',
    'matplotlib>=3.8.0',
    'joblib>=1.3.0',
]

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + COLAB_PKGS)
print("Dependencies installed.")

# ─── CELL 4: Verify GPU & project structure ──────────────────
import torch

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import sys
sys.path.insert(0, '/content/tradingbot')

# Quick import sanity-check
from config import CONFIG
from training.train import ModelTrainer
from training.loss import MultiTaskTradingLoss
from models.mamba_model import MambaTradingModel
print("All project imports OK.")
print(f"Checkpoints dir : {CONFIG.paths.CHECKPOINTS_DIR}")
print(f"Processed data  : {CONFIG.paths.PROCESSED_DATA_DIR}")

# ─── CELL 5: Upload pre-processed data from Drive ────────────
# On your PC, after running features/pipeline.py, copy:
#   tradingbot/data/processed/X_features.parquet
#   tradingbot/data/processed/y_labels.parquet
# to Google Drive root BEFORE running this cell.

import shutil
from pathlib import Path

DRIVE_DATA = Path('/content/drive/MyDrive/tradingbot_data')
LOCAL_DATA = CONFIG.paths.PROCESSED_DATA_DIR
LOCAL_DATA.mkdir(parents=True, exist_ok=True)

for fname in ['X_features.parquet', 'y_labels.parquet']:
    src = DRIVE_DATA / fname
    dst = LOCAL_DATA / fname
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {fname} -> {dst}")
    else:
        print(f"[WARN] {src} not found on Drive. Upload it first.")

# ─── CELL 6 (OPTION A - RECOMMENDED): Purged Walk-Forward CV ───────────
# Trains the Mamba model on 5 chronological folds with purge + embargo.
# This is the HONEST out-of-sample estimate (Lopez de Prado methodology).
# Estimated time on T4 GPU: ~2-4 hours (5 folds x 8 epochs, early stopping cuts most folds)

import os
os.chdir('/content/tradingbot')
from training.walkforward import run as run_walkforward

df_results = run_walkforward(
    backend="mamba",   # full Mamba multi-task model per fold
    n_folds=5,
    epochs=8,          # early stopping (patience=3) will usually cut earlier
    batch_size=256,
    patience=3,
)

# ─── CELL 6 (OPTION B): Quick A/B/C experiments ───────────────────────
# UNCOMMENT if you want the old 3-variant single-split comparison instead.
# import os
# os.chdir('/content/tradingbot')
# from training.run_experiments import run_all_experiments
# df_results = run_all_experiments()

# ─── CELL 7: Save checkpoints & results back to Drive ────────
import shutil
from pathlib import Path

CKPT_SRC  = CONFIG.paths.CHECKPOINTS_DIR
LOGS_SRC  = CONFIG.paths.LOGS_DIR
DRIVE_OUT = Path('/content/drive/MyDrive/tradingbot_results')
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

# Copy all experiment checkpoints
for f in CKPT_SRC.glob('*.pt'):
    shutil.copy(f, DRIVE_OUT / f.name)
    print(f"Saved checkpoint: {f.name}")

# Copy scalers
for f in CKPT_SRC.glob('*.joblib'):
    shutil.copy(f, DRIVE_OUT / f.name)
    print(f"Saved scaler: {f.name}")

# Copy results CSVs (walk-forward + old experiments)
for csv_name in ['experiment_results.csv', 'walkforward_mamba.csv', 'walkforward_hist.csv']:
    results_csv = LOGS_SRC / csv_name
    if results_csv.exists():
        shutil.copy(results_csv, DRIVE_OUT / csv_name)
        print(f"Saved {csv_name}")

print(f"\nAll outputs saved to: {DRIVE_OUT}")

# ─── CELL 8 (Optional): Plot comparison chart ────────────────
import matplotlib.pyplot as plt
import pandas as pd
import os

_wfcsv = '/content/drive/MyDrive/tradingbot_results/walkforward_mamba.csv'
if os.path.exists(_wfcsv):
    df = pd.read_csv(_wfcsv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Mamba XAUUSD – Purged Walk-Forward (OOS, honest)', fontsize=14, fontweight='bold')

    x = df['fold'].astype(str)
    axes[0].bar(x - 0.2, df['acc'] * 100, width=0.4, label='Model', color='#3b82f6')
    axes[0].bar(x + 0.2, df['baseline_acc'] * 100, width=0.4, label='Baseline', color='#94a3b8')
    axes[0].set_title('Direction Accuracy (%)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()

    colors = ['#10b981' if e > 0 else '#ef4444' for e in df['edge_vs_baseline']]
    axes[1].bar(x, df['edge_vs_baseline'] * 100, color=colors)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Edge vs Majority Baseline (%)')
    axes[1].set_ylabel('Edge (%)')

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/tradingbot_results/walkforward_chart.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Walk-forward chart saved to Drive.")
else:
    print("[WARN] walkforward_mamba.csv not found — run CELL 6 Option A first.")
