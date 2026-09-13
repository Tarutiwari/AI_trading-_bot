"""
Systematic Multi-Task Experiment Runner.
Trains and evaluates three Mamba configurations:
  A. Direction only
  B. Direction + Volatility (standardized)
  C. Direction + Volatility + Regime (current baseline)

Run with:
  python training/run_experiments.py
"""

from pathlib import Path
import sys
import time

import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import CONFIG
from training.train import ModelTrainer
from training.loss import MultiTaskTradingLoss


EXPERIMENTS = [
    {"name": "exp_A", "label": "A: Direction only",           "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    {"name": "exp_B", "label": "B: Direction + Volatility",   "alpha": 1.0, "beta": 0.5, "gamma": 0.0},
    {"name": "exp_C", "label": "C: Dir + Vol + Regime",       "alpha": 1.0, "beta": 0.5, "gamma": 0.3},
]


def run_all_experiments():
    print("\n" + "=" * 65)
    print("  SYSTEMATIC MULTI-TASK EXPERIMENT RUNNER")
    print("  Mamba XAUUSD -- Loss Ablation Study")
    print("=" * 65)

    # Load data ONCE, share splits across all experiments
    master_trainer = ModelTrainer(CONFIG)
    X, y = master_trainer.load_processed_data()
    train_l, val_l, test_l, in_feats = master_trainer.prepare_dataloaders(X, y)

    if master_trainer.vol_scaler is not None:
        m = master_trainer.vol_scaler.mean_[0]
        s = master_trainer.vol_scaler.scale_[0]
        print(f"[INFO] Vol scaler fitted on train: mean={m:.6f}  scale={s:.6f}")

    results = []

    for exp in EXPERIMENTS:
        sep = "-" * 65
        print(f"\n{sep}")
        print(f"  Running Experiment {exp['label']}")
        print(f"  alpha={exp['alpha']}  beta={exp['beta']}  gamma={exp['gamma']}")
        print(sep)
        t_start = time.time()

        # Fresh trainer per experiment; reuse same vol_scaler
        exp_trainer = ModelTrainer(CONFIG)
        exp_trainer.vol_scaler = master_trainer.vol_scaler

        model = exp_trainer.train_model(
            train_loader=train_l,
            val_loader=val_l,
            in_features=in_feats,
            alpha_direction=exp["alpha"],
            beta_volatility=exp["beta"],
            gamma_regime=exp["gamma"],
            experiment_name=exp["name"],
        )
        elapsed = time.time() - t_start

        # Rebuild criterion matching training weights for evaluation
        train_targets = train_l.dataset.target_directions
        class_counts = torch.bincount(train_targets, minlength=3).float()
        class_counts = torch.clamp(class_counts, min=1.0)
        dir_weights = (len(train_targets) / (3.0 * class_counts)).to(exp_trainer.device)
        test_criterion = MultiTaskTradingLoss(
            alpha_direction=exp["alpha"],
            beta_volatility=exp["beta"],
            gamma_regime=exp["gamma"],
            direction_weights=dir_weights,
        )

        test_loss, test_acc, test_f1, test_reg_acc, test_vol_mae, test_ld = exp_trainer.evaluate(
            model, test_l, test_criterion
        )

        print(f"\n  OOS Test Results [{exp['label']}]")
        print(f"     Direction Accuracy : {test_acc*100:.2f}%")
        print(f"     Macro F1           : {test_f1:.4f}")
        print(f"     Regime Accuracy    : {test_reg_acc*100:.2f}%")
        print(f"     Vol MAE (raw)      : {test_vol_mae:.6f}")
        print(f"     Total Test Loss    : {test_loss:.4f}")
        print(f"     Dir Loss           : {test_ld['direction_loss']:.4f}")
        print(f"     Vol Loss           : {test_ld['volatility_loss']:.6f}")
        print(f"     Reg Loss           : {test_ld['regime_loss']:.4f}")
        print(f"     Training Time      : {elapsed/60:.1f} min")

        results.append({
            "Experiment":       exp["label"],
            "alpha":            exp["alpha"],
            "beta":             exp["beta"],
            "gamma":            exp["gamma"],
            "OOS Dir Acc":      f"{test_acc*100:.2f}%",
            "OOS Macro F1":     f"{test_f1:.4f}",
            "OOS Regime Acc":   f"{test_reg_acc*100:.2f}%",
            "OOS Vol MAE":      f"{test_vol_mae:.6f}",
            "OOS Total Loss":   f"{test_loss:.4f}",
            "Train Time (min)": f"{elapsed/60:.1f}",
        })

    # Final summary table
    df = pd.DataFrame(results)
    sep = "=" * 65
    print(f"\n{sep}")
    print("  EXPERIMENT COMPARISON SUMMARY (Out-of-Sample Test Set)")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    out_path = CONFIG.paths.LOGS_DIR / "experiment_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] Results table -> {out_path}")

    # Auto-verdict
    accs = [float(r["OOS Dir Acc"].strip("%")) for r in results]
    labels = [r["Experiment"] for r in results]
    best_idx = accs.index(max(accs))
    print(f"\n[RESULT] Best Dir Accuracy: {labels[best_idx]} ({accs[best_idx]:.2f}%)")
    if accs[0] >= accs[1] >= accs[2]:
        print("[VERDICT] A >= B >= C  -> Auxiliary tasks are HURTING. Use Direction-only.")
    elif accs[0] <= accs[1] <= accs[2]:
        print("[VERDICT] A <= B <= C  -> Auxiliary tasks are HELPING. Keep full 3-task model.")
    else:
        print("[VERDICT] Mixed result -> Cross-check F1 + Vol MAE for optimal config.")

    return df


if __name__ == "__main__":
    run_all_experiments()
