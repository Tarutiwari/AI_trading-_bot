

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from training.dataset import PurgedWalkForwardCV

BACKENDS = ("hist", "mamba")


def _classification_report_fold(name, y_true, y_pred, y_ret, train_majority):
    """Compute honest per-fold metrics + money-relevant signal edge."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    base_acc = float(np.mean(y_true == train_majority))
    # Baseline macro-F1 of always predicting train-majority
    base_f1 = f1_score(y_true, np.full_like(y_true, train_majority),
                       average="macro", zero_division=0)

    # Signal edge: mean realized barrier return when model says BUY vs SELL
    buy_mask, sell_mask = (y_pred == 2), (y_pred == 0)
    buy_ret = float(np.mean(y_ret[buy_mask])) if buy_mask.any() else np.nan
    sell_ret = float(np.mean(y_ret[sell_mask])) if sell_mask.any() else np.nan
    n_buy, n_sell = int(buy_mask.sum()), int(sell_mask.sum())

    pred_dist = np.bincount(y_pred, minlength=3) / max(1, len(y_pred))

    return {
        "fold": name,
        "acc": acc,
        "macro_f1": f1,
        "baseline_acc": base_acc,
        "baseline_f1": base_f1,
        "edge_vs_baseline": acc - base_acc,
        "buy_trades": n_buy,
        "buy_mean_ret": buy_ret,
        "sell_trades": n_sell,
        "sell_mean_ret": sell_ret,
        "pred_bear_pct": pred_dist[0],
        "pred_neut_pct": pred_dist[1],
        "pred_bull_pct": pred_dist[2],
    }


def run(backend: str = "hist", n_folds: int = 5, epochs: int = 8,
        batch_size: int = 256, patience: int = 3, X=None, y=None) -> pd.DataFrame:
    assert backend in BACKENDS, f"backend must be one of {BACKENDS}"

    if X is None or y is None:
        proc = CONFIG.paths.PROCESSED_DATA_DIR
        X = pd.read_parquet(proc / "X_features.parquet")
        y = pd.read_parquet(proc / "y_labels.parquet")
    n = len(X)
    print(f"\n[WALK-FORWARD] backend={backend} | samples={n:,} | features={X.shape[1]}")
    print(f"[WALK-FORWARD] index range: {X.index.min()} -> {X.index.max()}")

    seq_len = CONFIG.features.LOOKBACK_SEQUENCE_LENGTH
    hold = CONFIG.labeling.MAX_HOLDING_BARS
    purge = seq_len + hold                      # sequence + triple-barrier overlap
    embargo = max(int(n * 0.01), hold)          # 1% embargo after purge

    cv = PurgedWalkForwardCV(n_splits=n_folds, holding_period_bars=purge,
                             embargo_pct=embargo / n)
    results = []
    device_note = ""

    if backend == "mamba":
        import torch
        if not torch.cuda.is_available():
            print("[WARN] No CUDA device. Mamba walk-forward on CPU is extremely slow "
                  "(~16 s/batch). Use Colab T4 (see colab_experiments.py) or --backend hist.")

    for k, (train_idx, val_idx) in enumerate(cv.split(n), 1):
        train_idx = np.asarray(train_idx)
        # Carve inner validation tail out of the training window for early stopping.
        # Must exceed seq_len + purge so the sequence dataset yields >0 val samples.
        iv_n = max(seq_len + purge, int(len(train_idx) * 0.15))
        inner_cut = max(1, len(train_idx) - iv_n)
        fit_idx, inner_val_idx = train_idx[:inner_cut], train_idx[inner_cut:]

        test_start = int(val_idx[0]) if len(val_idx) else n
        test_end = min(test_start + len(val_idx), n)
        if test_start >= n:
            break
        test_idx = np.arange(test_start, test_end)

        t_range = f"{X.index[test_start]} -> {X.index[test_end - 1]}"
        print(f"\n----- FOLD {k}/{n_folds} -----")
        print(f"  train: {len(fit_idx):,} bars | inner-val: {len(inner_val_idx):,} | "
              f"test: {len(test_idx):,} bars ({t_range})")
        print(f"  purge={purge} bars, embargo={embargo} bars")

        y_train_dir = y["target_direction"].iloc[fit_idx].values
        train_majority = pd.Series(y_train_dir).mode()[0]
        y_test_dir = y["target_direction"].iloc[test_idx].values
        y_test_ret = y["target_return"].iloc[test_idx].values

        t0 = time.time()
        if backend == "hist":
            preds, eval_offset = _fit_predict_hist(X, y, fit_idx, inner_val_idx, test_idx), 0
        else:
            preds, eval_offset, device_note = _fit_predict_mamba(
                X, y, fit_idx, inner_val_idx, test_idx,
                epochs=epochs, batch_size=batch_size, patience=patience, fold=k,
            )
        elapsed = time.time() - t0

        # Mamba can only predict bars with a full lookback window behind them,
        # so its predictions start eval_offset bars into the test slice.
        y_eval_dir = y_test_dir[eval_offset:]
        y_eval_ret = y_test_ret[eval_offset:]

        row = _classification_report_fold(k, y_eval_dir, preds, y_eval_ret, train_majority)
        row["seconds"] = round(elapsed, 1)
        results.append(row)

        print(f"  -> acc={row['acc']*100:.2f}% (baseline {row['baseline_acc']*100:.2f}%, "
              f"edge {row['edge_vs_baseline']*100:+.2f}%) | macroF1={row['macro_f1']:.4f}")
        print(f"     BUY: {row['buy_trades']} trades, mean ret {row['buy_mean_ret']:+.6f} | "
              f"SELL: {row['sell_trades']} trades, mean ret {row['sell_mean_ret']:+.6f}")
        print(f"     pred dist bear/neut/bull = "
              f"{row['pred_bear_pct']*100:.0f}/{row['pred_neut_pct']*100:.0f}/{row['pred_bull_pct']*100:.0f}")
        print(f"     fold time: {elapsed:.0f}s {device_note}")

    df = pd.DataFrame(results)
    if len(df):
        print("\n" + "=" * 78)
        print("  WALK-FORWARD SUMMARY (honest out-of-sample)")
        print("=" * 78)
        print(f"  Mean acc        : {df['acc'].mean()*100:.2f}%  "
              f"(baseline {df['baseline_acc'].mean()*100:.2f}%)")
        print(f"  Mean macro F1   : {df['macro_f1'].mean():.4f}  "
              f"(baseline {df['baseline_f1'].mean():.4f})")
        print(f"  Mean edge       : {df['edge_vs_baseline'].mean()*100:+.2f}%")
        pos_buy = df["buy_mean_ret"].mean()
        pos_sell = df["sell_mean_ret"].mean()
        print(f"  Avg BUY ret     : {pos_buy:+.6f} | Avg SELL ret: {pos_sell:+.6f}")

        # Economic gate: convert mean winning-side return to GOLD POINTS and
        # compare against spread + commission, not just sign.
        approx_price = 2400.0  # rough XAUUSD level for unit conversion
        best_side_ret = max(pos_buy, -pos_sell)
        edge_points = best_side_ret * approx_price * 100.0  # 1 point = $0.01/oz
        cost_points = (CONFIG.risk.MAX_ALLOWED_SPREAD_POINTS + 7.0)  # spread + $7/lot commission
        tradable = edge_points >= 2.0 * cost_points
        if tradable:
            verdict = ("TRADABLE EDGE — proceed to Mamba walk-forward on Colab, "
                       "then cost-aware backtest")
        elif df["edge_vs_baseline"].mean() > 0.005:
            verdict = ("STATISTICAL BUT NOT TRADABLE — edge exists but is smaller "
                       "than 2x transaction costs. Widen barriers, raise timeframe, "
                       "or add stronger features (DXY/10Y) before retraining")
        else:
            verdict = "NO RELIABLE EDGE — features/labels need work before retraining"
        print(f"  Gross edge      : ~{edge_points:.1f} points/trade vs cost ~{cost_points:.0f} pts "
              f"(spread+commission) -> {'TRADABLE' if tradable else 'NOT TRADABLE on M5 costs'}")
        print(f"  VERDICT         : {verdict}")
        print("=" * 78)

        out = CONFIG.paths.LOGS_DIR / f"walkforward_{backend}.csv"
        CONFIG.paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")
    return df


def _fit_predict_hist(X, y, fit_idx, inner_val_idx, test_idx) -> np.ndarray:
    """LightGBM on last-bar features. Fast honest signal check on CPU."""
    import lightgbm as lgb

    X_fit, y_fit = X.iloc[fit_idx].values, y["target_direction"].iloc[fit_idx].values
    X_iv, y_iv = X.iloc[inner_val_idx].values, y["target_direction"].iloc[inner_val_idx].values
    X_te = X.iloc[test_idx].values

    gbm = lgb.LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
        verbose=-1, n_jobs=-1, random_state=42,
    )
    gbm.fit(X_fit, y_fit, eval_set=[(X_iv, y_iv)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return gbm.predict(X_te)


def _fit_predict_mamba(X, y, fit_idx, inner_val_idx, test_idx,
                       epochs, batch_size, patience, fold):
    """Full Mamba multi-task model per fold using ModelTrainer with custom split.

    Returns (preds, eval_offset, device_note) where preds are aligned to
    test positions [eval_offset:] (eval_offset = seq_len - 1 because a full
    lookback window is required before the first predictable bar).
    """
    import torch
    from training.train import ModelTrainer
    from training.dataset import TimeSeriesSequenceDataset

    trainer = ModelTrainer(CONFIG)
    # Scaler fit ONLY on fold-train inside prepare_dataloaders; val = inner tail for early stopping
    train_l, val_l, _test_l, in_feats = trainer.prepare_dataloaders(
        X, y, train_idx=fit_idx, val_idx=inner_val_idx, test_idx=test_idx,
        batch_size=batch_size,
    )

    trainer.train_model(
        train_l, val_l, in_feats,
        max_epochs=epochs, patience=patience,
        experiment_name=f"wf_fold{fold}", save_best_by="val_f1",
    )
    model = trainer._last_model
    seq_len = CONFIG.features.LOOKBACK_SEQUENCE_LENGTH

    # Stride-1 dataset over the test slice so every bar with a full window gets a prediction
    X_te_scaled = trainer.scaler.transform(X.iloc[test_idx].values)
    te_offset = min(len(test_idx), len(test_idx))  # guard tiny folds
    y_te = y.iloc[test_idx]
    te_ds = TimeSeriesSequenceDataset(
        X_te_scaled,
        y_te["target_direction"].values,
        trainer.vol_scaler.transform(y_te[["target_volatility"]].values).flatten(),
        y_te["target_regime"].values,
        seq_len=seq_len, stride=1,
    )
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb, *_ in te_loader:
            out = model(xb.to(trainer.device))
            preds.extend(out["direction_logits"].argmax(dim=-1).cpu().numpy())
    preds = np.asarray(preds)
    assert len(preds) == len(test_idx) - seq_len + 1, \
        f"pred count {len(preds)} != expected {len(test_idx) - seq_len + 1}"
    device_note = f"on {trainer.device}"
    return preds, seq_len - 1, device_note


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Purged walk-forward CV runner")
    ap.add_argument("--backend", choices=BACKENDS, default="hist")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8, help="mamba backend only")
    ap.add_argument("--batch-size", type=int, default=256, help="mamba backend only")
    ap.add_argument("--patience", type=int, default=3, help="mamba backend only")
    args = ap.parse_args()
    run(backend=args.backend, n_folds=args.folds, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience)
"""
Purged & Embargoed Walk-Forward Cross-Validation Runner.

Gives an HONEST out-of-sample performance estimate of the current feature set
before any live deployment. Two backends:

  hist  -> LightGBM on last-bar features (CPU, minutes). Answers:
           "Do these 86 features contain ANY directional signal?"
  mamba -> Full Mamba multi-task sequence model per fold (GPU strongly advised;
           ~16 s/batch on CPU). Intended for Colab T4 via colab_experiments.py.

Leakage protection per fold:
  - purge  = LOOKBACK_SEQUENCE_LENGTH + MAX_HOLDING_BARS  (sequence + label overlap)
  - embargo= 1% of dataset (Lopez de Prado) AFTER purge

Usage:
  python training/walkforward.py --backend hist  --folds 5
  python training/walkforward.py --backend mamba --folds 5 --epochs 8
"""