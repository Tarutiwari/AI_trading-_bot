"""
Phase 1 Diagnostic Script — Investigate BEFORE changing the model.

Inspects:
1. Target direction class distribution (train / val / test)
2. Target return distribution per class
3. Target volatility distribution
4. Regime distribution
5. Feature statistics & NaN audit
6. Multi-task loss decomposition (direction vs volatility vs regime)
7. Confusion matrix & per-class precision/recall on validation
8. What the model actually predicts (prediction distribution)
9. Baseline comparisons (majority class, random, logistic regression, LightGBM)
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows (avoid cp1252 encoding errors)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import Counter

import torch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression

from config import CONFIG
from training.dataset import TimeSeriesSequenceDataset
from training.loss import MultiTaskTradingLoss


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def load_data():
    proc_dir = CONFIG.paths.PROCESSED_DATA_DIR
    X = pd.read_parquet(proc_dir / "X_features.parquet")
    y = pd.read_parquet(proc_dir / "y_labels.parquet")
    return X, y


def split_data(X, y):
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    splits = {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "val":   (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        "test":  (X.iloc[val_end:], y.iloc[val_end:]),
    }
    return splits


def diagnose_target_distribution(splits):
    section("1. TARGET DIRECTION CLASS DISTRIBUTION")
    for name, (X_s, y_s) in splits.items():
        counts = y_s["target_direction"].value_counts().sort_index()
        pcts = y_s["target_direction"].value_counts(normalize=True).sort_index() * 100
        print(f"  [{name.upper()}] Total: {len(y_s):,}")
        for cls_id in sorted(counts.index):
            label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}.get(cls_id, f"Class {cls_id}")
            print(f"    Class {cls_id} ({label}): {counts[cls_id]:>8,} ({pcts[cls_id]:5.1f}%)")
        print()


def diagnose_return_distribution(splits):
    section("2. TARGET RETURN DISTRIBUTION PER CLASS")
    for name, (X_s, y_s) in splits.items():
        print(f"  [{name.upper()}]")
        for cls_id in sorted(y_s["target_direction"].unique()):
            label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}.get(cls_id, f"Class {cls_id}")
            returns = y_s.loc[y_s["target_direction"] == cls_id, "target_return"]
            print(f"    Class {cls_id} ({label}): mean={returns.mean():.6f}, "
                  f"std={returns.std():.6f}, min={returns.min():.6f}, "
                  f"max={returns.max():.6f}, median={returns.median():.6f}")
        print()


def diagnose_volatility_distribution(splits):
    section("3. TARGET VOLATILITY DISTRIBUTION")
    for name, (X_s, y_s) in splits.items():
        vol = y_s["target_volatility"]
        print(f"  [{name.upper()}]: mean={vol.mean():.6f}, std={vol.std():.6f}, "
              f"min={vol.min():.6f}, max={vol.max():.6f}, "
              f"p25={vol.quantile(0.25):.6f}, p50={vol.median():.6f}, "
              f"p75={vol.quantile(0.75):.6f}, p95={vol.quantile(0.95):.6f}")
    print()


def diagnose_regime_distribution(splits):
    section("4. REGIME DISTRIBUTION")
    regime_map = {0: "Bull Trend", 1: "Bear Trend", 2: "Chop/Range", 3: "Vol Spike"}
    for name, (X_s, y_s) in splits.items():
        counts = y_s["target_regime"].value_counts().sort_index()
        pcts = y_s["target_regime"].value_counts(normalize=True).sort_index() * 100
        print(f"  [{name.upper()}]")
        for cls_id in sorted(counts.index):
            label = regime_map.get(cls_id, f"Regime {cls_id}")
            print(f"    Regime {cls_id} ({label}): {counts[cls_id]:>8,} ({pcts[cls_id]:5.1f}%)")
        print()


def diagnose_feature_quality(splits):
    section("5. FEATURE QUALITY AUDIT")
    X_train, _ = splits["train"]
    
    # NaN check
    nan_counts = X_train.isnull().sum()
    if nan_counts.sum() > 0:
        print("  ⚠️  NaN features found in training data:")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {cnt} NaNs ({cnt/len(X_train)*100:.1f}%)")
    else:
        print("  ✅ No NaN values in training features.")
    
    # Constant/near-constant features
    print(f"\n  Feature count: {X_train.shape[1]}")
    stds = X_train.std()
    near_const = stds[stds < 1e-8]
    if len(near_const) > 0:
        print(f"  ⚠️  Near-constant features (std < 1e-8):")
        for col in near_const.index:
            print(f"    {col}: std={stds[col]:.2e}")
    else:
        print("  ✅ No near-constant features.")
    
    # Top correlated feature pairs
    print(f"\n  Top 10 most correlated feature pairs (absolute):")
    corr = X_train.corr().abs()
    # Zero out diagonal
    np.fill_diagonal(corr.values, 0)
    # Get upper triangle pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_corr = upper.stack().nlargest(10)
    for (f1, f2), val in top_corr.items():
        print(f"    {f1} ↔ {f2}: {val:.4f}")
    print()


def diagnose_loss_decomposition(splits):
    section("6. MULTI-TASK LOSS DECOMPOSITION (on validation set)")
    
    X_train_raw, y_train = splits["train"]
    X_val_raw, y_val = splits["val"]
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw.values)
    X_val = scaler.transform(X_val_raw.values)
    
    seq_len = CONFIG.features.LOOKBACK_SEQUENCE_LENGTH
    
    val_ds = TimeSeriesSequenceDataset(
        X_val,
        y_val["target_direction"].values,
        y_val["target_volatility"].values,
        y_val["target_regime"].values,
        seq_len=seq_len,
    )
    
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
    
    # Load best model if exists
    best_path = CONFIG.paths.CHECKPOINTS_DIR / "mamba_xauusd_best.pt"
    if not best_path.exists():
        print("  ⚠️  No trained model checkpoint found. Skipping loss decomposition.")
        return None, None, None
    
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    from models.mamba_model import MambaTradingModel
    
    m_cfg = CONFIG.mamba
    model = MambaTradingModel(
        in_features=X_train_raw.shape[1],
        d_model=m_cfg.D_MODEL,
        n_layers=m_cfg.N_LAYERS,
        d_state=m_cfg.D_STATE,
        d_conv=m_cfg.D_CONV,
        expand=m_cfg.EXPAND_FACTOR,
        dropout=m_cfg.DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    criterion = MultiTaskTradingLoss(
        alpha_direction=m_cfg.ALPHA_DIRECTION,
        beta_volatility=m_cfg.BETA_VOLATILITY,
        gamma_regime=m_cfg.GAMMA_REGIME,
    )
    
    # Collect individual losses and predictions
    dir_losses, vol_losses, reg_losses = [], [], []
    all_preds_dir, all_trues_dir = [], []
    all_preds_regime, all_trues_regime = [], []
    all_probs = []
    
    with torch.no_grad():
        for x_b, y_dir, y_vol, y_reg in val_loader:
            preds = model(x_b)
            
            # Individual losses (unweighted)
            from torch.nn import functional as F
            dl = F.cross_entropy(preds["direction_logits"], y_dir, reduction="mean").item()
            vl = F.smooth_l1_loss(preds["volatility"], y_vol, reduction="mean").item()
            rl = F.cross_entropy(preds["regime_logits"], y_reg, reduction="mean").item()
            
            dir_losses.append(dl)
            vol_losses.append(vl)
            reg_losses.append(rl)
            
            pred_dirs = preds["direction_logits"].argmax(dim=-1).numpy()
            all_preds_dir.extend(pred_dirs)
            all_trues_dir.extend(y_dir.numpy())
            
            pred_regimes = preds["regime_logits"].argmax(dim=-1).numpy()
            all_preds_regime.extend(pred_regimes)
            all_trues_regime.extend(y_reg.numpy())
            
            all_probs.extend(preds["direction_probs"].numpy())
    
    avg_dir = np.mean(dir_losses)
    avg_vol = np.mean(vol_losses)
    avg_reg = np.mean(reg_losses)
    
    print(f"  UNWEIGHTED individual losses (raw):")
    print(f"    Direction CE:    {avg_dir:.4f}")
    print(f"    Volatility L1:   {avg_vol:.4f}")
    print(f"    Regime CE:       {avg_reg:.4f}")
    print()
    print(f"  WEIGHTED contributions to total loss:")
    print(f"    α × Direction:   {m_cfg.ALPHA_DIRECTION} × {avg_dir:.4f} = {m_cfg.ALPHA_DIRECTION * avg_dir:.4f}")
    print(f"    β × Volatility:  {m_cfg.BETA_VOLATILITY} × {avg_vol:.4f} = {m_cfg.BETA_VOLATILITY * avg_vol:.4f}")
    print(f"    γ × Regime:      {m_cfg.GAMMA_REGIME} × {avg_reg:.4f} = {m_cfg.GAMMA_REGIME * avg_reg:.4f}")
    total = (m_cfg.ALPHA_DIRECTION * avg_dir + m_cfg.BETA_VOLATILITY * avg_vol + m_cfg.GAMMA_REGIME * avg_reg)
    print(f"    TOTAL:           {total:.4f}")
    
    dir_pct = (m_cfg.ALPHA_DIRECTION * avg_dir / total) * 100
    vol_pct = (m_cfg.BETA_VOLATILITY * avg_vol / total) * 100
    reg_pct = (m_cfg.GAMMA_REGIME * avg_reg / total) * 100
    print(f"\n  Loss share: Direction {dir_pct:.1f}% | Volatility {vol_pct:.1f}% | Regime {reg_pct:.1f}%")
    
    return all_preds_dir, all_trues_dir, all_probs


def diagnose_predictions(all_preds_dir, all_trues_dir, all_probs):
    if all_preds_dir is None:
        return
    
    section("7. CONFUSION MATRIX & CLASSIFICATION REPORT (Validation)")
    
    label_names = ["Bearish (0)", "Neutral (1)", "Bullish (2)"]
    print(classification_report(
        all_trues_dir, all_preds_dir,
        target_names=label_names, digits=4, zero_division=0
    ))
    
    cm = confusion_matrix(all_trues_dir, all_preds_dir, labels=[0, 1, 2])
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>12s} Pred_Bear  Pred_Neut  Pred_Bull")
    for i, name in enumerate(["Actual_Bear", "Actual_Neut", "Actual_Bull"]):
        print(f"  {name:>12s}   {cm[i, 0]:>6d}     {cm[i, 1]:>6d}     {cm[i, 2]:>6d}")
    
    section("8. PREDICTION DISTRIBUTION — What the model actually outputs")
    
    pred_counts = Counter(all_preds_dir)
    true_counts = Counter(all_trues_dir)
    total_p = len(all_preds_dir)
    
    print(f"  ACTUAL distribution:")
    for cls in [0, 1, 2]:
        label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}[cls]
        cnt = true_counts.get(cls, 0)
        print(f"    Class {cls} ({label}): {cnt:>8,} ({cnt/total_p*100:5.1f}%)")
    
    print(f"\n  PREDICTED distribution:")
    for cls in [0, 1, 2]:
        label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}[cls]
        cnt = pred_counts.get(cls, 0)
        print(f"    Class {cls} ({label}): {cnt:>8,} ({cnt/total_p*100:5.1f}%)")
    
    # Probability confidence analysis
    probs = np.array(all_probs)
    max_probs = probs.max(axis=1)
    print(f"\n  Prediction confidence (max softmax probability):")
    print(f"    Mean: {max_probs.mean():.4f}")
    print(f"    Std:  {max_probs.std():.4f}")
    print(f"    Min:  {max_probs.min():.4f}")
    print(f"    Max:  {max_probs.max():.4f}")
    print(f"    >0.65 (trade threshold): {(max_probs > 0.65).sum()} / {len(max_probs)} ({(max_probs > 0.65).mean()*100:.1f}%)")
    print(f"    >0.50:                   {(max_probs > 0.50).sum()} / {len(max_probs)} ({(max_probs > 0.50).mean()*100:.1f}%)")


def diagnose_baselines(splits):
    section("9. BASELINE COMPARISONS (Direction only)")
    
    X_train_raw, y_train = splits["train"]
    X_val_raw, y_val = splits["val"]
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw.values)
    X_val = scaler.transform(X_val_raw.values)
    
    y_train_dir = y_train["target_direction"].values
    y_val_dir = y_val["target_direction"].values
    
    results = []
    
    # Baseline 1: Majority class
    majority_class = Counter(y_train_dir).most_common(1)[0][0]
    maj_preds = np.full(len(y_val_dir), majority_class)
    maj_acc = accuracy_score(y_val_dir, maj_preds)
    maj_f1 = f1_score(y_val_dir, maj_preds, average="macro", zero_division=0)
    results.append(("Majority Class", maj_acc, maj_f1))
    print(f"  Majority Class ({majority_class}):  Acc={maj_acc*100:.2f}%  Macro-F1={maj_f1:.4f}")
    
    # Baseline 2: Random (class-weighted)
    np.random.seed(42)
    class_probs = np.bincount(y_train_dir) / len(y_train_dir)
    rand_preds = np.random.choice(len(class_probs), size=len(y_val_dir), p=class_probs)
    rand_acc = accuracy_score(y_val_dir, rand_preds)
    rand_f1 = f1_score(y_val_dir, rand_preds, average="macro", zero_division=0)
    results.append(("Random (weighted)", rand_acc, rand_f1))
    print(f"  Random (weighted):    Acc={rand_acc*100:.2f}%  Macro-F1={rand_f1:.4f}")
    
    # Baseline 3: Logistic Regression (last timestep features, no sequence)
    print(f"  Training Logistic Regression (no sequence, raw features)...")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", n_jobs=-1)
    lr.fit(X_train, y_train_dir)
    lr_preds = lr.predict(X_val)
    lr_acc = accuracy_score(y_val_dir, lr_preds)
    lr_f1 = f1_score(y_val_dir, lr_preds, average="macro", zero_division=0)
    results.append(("Logistic Regression", lr_acc, lr_f1))
    print(f"  Logistic Regression:  Acc={lr_acc*100:.2f}%  Macro-F1={lr_f1:.4f}")
    
    # Baseline 4: LightGBM
    try:
        import lightgbm as lgb
        print(f"  Training LightGBM...")
        gbm = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        )
        gbm.fit(
            X_train, y_train_dir,
            eval_set=[(X_val, y_val_dir)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        gbm_preds = gbm.predict(X_val)
        gbm_acc = accuracy_score(y_val_dir, gbm_preds)
        gbm_f1 = f1_score(y_val_dir, gbm_preds, average="macro", zero_division=0)
        results.append(("LightGBM", gbm_acc, gbm_f1))
        print(f"  LightGBM:             Acc={gbm_acc*100:.2f}%  Macro-F1={gbm_f1:.4f}")
        
        # Feature importance from LightGBM
        print(f"\n  Top 15 LightGBM feature importances:")
        feat_imp = pd.Series(gbm.feature_importances_, index=X_train_raw.columns)
        feat_imp_sorted = feat_imp.sort_values(ascending=False).head(15)
        for feat_name, imp in feat_imp_sorted.items():
            print(f"    {feat_name}: {imp}")
    except ImportError:
        print("  ⚠️  LightGBM not installed. Run: pip install lightgbm")
    
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY TABLE:")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print(f"  {'─' * 50}")
    for name, acc, f1 in results:
        print(f"  {name:<25s} {acc*100:>9.2f}% {f1:>10.4f}")
    print(f"  {'─' * 50}")


def diagnose_target_leakage(splits):
    """Check if regime labels leak into direction prediction."""
    section("10. TARGET LEAKAGE CHECK (Regime ↔ Direction)")
    _, y_train = splits["train"]
    
    # Check cross-tabulation
    ct = pd.crosstab(
        y_train["target_direction"],
        y_train["target_regime"],
        margins=True
    )
    ct.index = [
        {0: "Bearish", 1: "Neutral", 2: "Bullish"}.get(i, str(i))
        for i in ct.index
    ]
    ct.columns = [
        {0: "Bull Trend", 1: "Bear Trend", 2: "Chop", 3: "Vol Spike"}.get(c, str(c))
        for c in ct.columns
    ]
    print("  Direction × Regime Cross-Tab (Train):")
    print(ct.to_string())
    print()
    
    # Check: is regime nearly deterministic given direction?
    print("  ⚠️  Note: Regime is DERIVED from direction labels + volatility threshold.")
    print("  This means regime labels are NOT independent of direction labels.")
    print("  The model gets a 'free hint' from the regime task about direction.")
    print("  Consider: is the regime head actually helping or just memorizing direction?")


if __name__ == "__main__":
    X, y = load_data()
    print(f"Loaded: {len(X):,} samples × {X.shape[1]} features")
    print(f"Label columns: {list(y.columns)}")
    print(f"Date range: {X.index.min()} -> {X.index.max()}")
    
    splits = split_data(X, y)
    
    # Phase 1 diagnostics — no model changes
    diagnose_target_distribution(splits)
    diagnose_return_distribution(splits)
    diagnose_volatility_distribution(splits)
    diagnose_regime_distribution(splits)
    diagnose_feature_quality(splits)
    diagnose_target_leakage(splits)
    
    # Phase 1.5 — requires trained model checkpoint
    all_preds, all_trues, all_probs = diagnose_loss_decomposition(splits)
    diagnose_predictions(all_preds, all_trues, all_probs)
    
    # Phase 1.6 — Regime accuracy info
    section("BONUS: REGIME ACCURACY NOTE")
    print("  Regime accuracy is now tracked in the training loop and printed per epoch.")
    print("  Check the training output for 'Reg Acc: XX.XX%' in validation metrics.")
    best_path = CONFIG.paths.CHECKPOINTS_DIR / "mamba_xauusd_best.pt"
    if best_path.exists():
        import torch
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        print(f"  Model best epoch: {ckpt.get('epoch', '?')}, "
              f"val_reg_acc: {ckpt.get('val_reg_acc', 'N/A')}")
    
    # Phase 2 — baselines
    diagnose_baselines(splits)
    
    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSIS COMPLETE")
    print(f"{'=' * 70}")
