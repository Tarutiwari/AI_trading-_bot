"""
Optuna Bayesian Hyperparameter Search for Mamba Trading Model.
Searches over architecture, training, and loss hyperparameters.

Usage:
    python training/hyperparam_search.py                   # Default 30 trials
    python training/hyperparam_search.py --trials 50       # Custom trial count
    python training/hyperparam_search.py --timeout 7200    # 2-hour time limit
    python training/hyperparam_search.py --resume best_trial.json  # Resume from file
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CONFIG, BotConfig, MambaModelConfig
from training.train import ModelTrainer
from training.loss import MultiTaskTradingLoss
from models.mamba_model import MambaTradingModel


def create_config_from_params(params: dict, base_config: BotConfig) -> BotConfig:
    """Create a BotConfig with overridden hyperparameters from search params."""
    from dataclasses import replace

    mamba_cfg = replace(base_config.mamba,
        D_MODEL=params["d_model"],
        N_LAYERS=params["n_layers"],
        D_STATE=params["d_state"],
        D_CONV=params["d_conv"],
        EXPAND_FACTOR=params["expand"],
        DROPOUT=params["dropout"],
        LEARNING_RATE=params["lr"],
        WEIGHT_DECAY=params["weight_decay"],
        BATCH_SIZE=params["batch_size"],
        MAX_EPOCHS=params["max_epochs"],
        EARLY_STOPPING_PATIENCE=params["patience"],
        ALPHA_DIRECTION=params["alpha_direction"],
        BETA_VOLATILITY=params["beta_volatility"],
        GAMMA_REGIME=params["gamma_regime"],
    )

    return replace(base_config, mamba=mamba_cfg)


def objective(trial, X, y, base_config: BotConfig, device: str = None):
    """Optuna objective function: trains model and returns negative val F1."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    # --- Architecture search space ---
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 48, 64, 96, 128]),
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
        "d_conv": trial.suggest_int("d_conv", 3, 6),
        "expand": trial.suggest_categorical("expand", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),

        # --- Training search space ---
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "max_epochs": 30,  # Fixed for search efficiency
        "patience": trial.suggest_int("patience", 5, 15),

        # --- Loss weight search space ---
        "alpha_direction": trial.suggest_float("alpha_direction", 0.5, 2.0),
        "beta_volatility": trial.suggest_float("beta_volatility", 0.0, 1.0),
        "gamma_regime": trial.suggest_float("gamma_regime", 0.0, 0.5),
    }

    # Create custom config for this trial
    trial_config = create_config_from_params(params, base_config)

    # Initialize trainer with trial config
    trainer = ModelTrainer(config=trial_config, device=device)

    # Use prepared dataloaders (shared across trials)
    train_loader, val_loader, test_loader, in_features = trainer.prepare_dataloaders(X, y)

    try:
        model = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            in_features=in_features,
            alpha_direction=params["alpha_direction"],
            beta_volatility=params["beta_volatility"],
            gamma_regime=params["gamma_regime"],
            experiment_name=f"optuna_t{trial.number}",
            patience=params["patience"],
            max_epochs=params["max_epochs"],
            save_best_by="val_f1",
        )

        # Final evaluation on validation set
        criterion = MultiTaskTradingLoss(
            alpha_direction=params["alpha_direction"],
            beta_volatility=params["beta_volatility"],
            gamma_regime=params["gamma_regime"],
        )
        val_loss, val_acc, val_f1, val_reg_acc, val_vol_mae, _ = trainer.evaluate(
            model, val_loader, criterion
        )

        # Report intermediate values for pruning
        trial.report(val_f1, step=0)

        # Return negative F1 (Optuna minimizes by default)
        return -val_f1

    except Exception as e:
        print(f"  [ERROR] Trial {trial.number} failed: {e}")
        return 0.0  # Worst possible score


def run_search(n_trials: int = 30, timeout: int = None, resume_from: str = None, device: str = None):
    """Run the full Optuna hyperparameter search."""
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("[ERROR] Optuna not installed. Run: pip install optuna")
        print("        For visualization: pip install optuna-dashboard")
        sys.exit(1)

    # Load data once (shared across all trials)
    print("\n" + "=" * 65)
    print("  OPTUNA BAYESIAN HYPERPARAMETER SEARCH")
    print("  Mamba XAUUSD Trading Model")
    print("=" * 65)

    trainer = ModelTrainer(config=CONFIG, device=device)
    X, y = trainer.load_processed_data()

    # Store data references for objective function
    _search_data = {"X": X, "y": y, "base_config": CONFIG}

    # Create or load study
    storage = "sqlite:///optuna_mamba_search.db"
    study_name = "mamba_xauusd_hpo"

    load_if_exists = resume_from is None
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    if resume_from:
        # Load previous trials from JSON
        with open(resume_from, "r") as f:
            history = json.load(f)
        for trial_data in history.get("trials", []):
            if trial_data.get("state") == "COMPLETE":
                study.add_trial(
                    optuna.trial.create_trial(
                        params=trial_data["params"],
                        values=trial_data["values"],
                    )
                )
        print(f"[INFO] Resumed {len(study.trials)} previous trials from {resume_from}")

    print(f"\n[CONFIG] Trials: {n_trials} | Timeout: {timeout or 'None'}s | Device: {device or 'auto'}")
    print(f"[CONFIG] Storage: {storage} | Study: {study_name}")
    print(f"[CONFIG] Starting from trial {len(study.trials)}\n")

    # Objective wrapper to pass shared data
    def obj(trial):
        return objective(trial, _search_data["X"], _search_data["y"],
                        _search_data["base_config"], device)

    start_time = time.time()
    study.optimize(obj, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - start_time

    # --- Results Summary ---
    print("\n" + "=" * 65)
    print("  SEARCH COMPLETE")
    print("=" * 65)

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

    print(f"\n  Total trials:    {len(study.trials)}")
    print(f"  Completed:       {len(complete_trials)}")
    print(f"  Pruned:          {len(pruned_trials)}")
    print(f"  Failed:          {len.failed_trials}")
    print(f"  Total time:      {elapsed/60:.1f} min")

    if complete_trials:
        best = study.best_trial
        print(f"\n  {'─' * 55}")
        print(f"  BEST TRIAL: #{best.number}")
        print(f"  {'─' * 55}")
        print(f"  Objective (neg F1): {best.value:.4f}  (Val F1 = {-best.value:.4f})")
        print(f"\n  Best hyperparameters:")
        for key, val in best.params.items():
            if isinstance(val, float):
                print(f"    {key:25s}: {val:.6f}")
            else:
                print(f"    {key:25s}: {val}")

        # Save best params
        best_path = CONFIG.paths.LOGS_DIR / "best_hyperparams.json"
        with open(best_path, "w") as f:
            json.dump({
                "best_trial": best.number,
                "best_val_f1": -best.value,
                "params": best.params,
            }, f, indent=2, default=str)
        print(f"\n  [SAVED] Best params -> {best_path}")

        # Save full study results
        results_path = CONFIG.paths.LOGS_DIR / "optuna_study.json"
        study_results = {
            "best_trial": best.number,
            "best_val_f1": -best.value,
            "n_trials": len(study.trials),
            "trials": [
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "params": t.params,
                }
                for t in study.trials
            ],
        }
        with open(results_path, "w") as f:
            json.dump(study_results, f, indent=2, default=str)
        print(f"  [SAVED] Full study -> {results_path}")

        # Print top 5 trials
        sorted_trials = sorted(complete_trials, key=lambda t: t.value)[:5]
        print(f"\n  TOP 5 TRIALS:")
        print(f"  {'─' * 55}")
        for t in sorted_trials:
            f1 = -t.value
            print(f"  Trial #{t.number:3d} | Val F1: {f1:.4f} | "
                  f"LR: {t.params.get('lr', 0):.2e} | "
                  f"D: {t.params.get('d_model', 0)} | "
                  f"L: {t.params.get('n_layers', 0)} | "
                  f"Drop: {t.params.get('dropout', 0):.2f}")

    # Return best params for easy access
    return study.best_trial.params if complete_trials else {}


def print_recommended_config(params: dict):
    """Print a ready-to-use config snippet from best params."""
    print("\n  # --- Recommended config.py update ---")
    print("  class MambaModelConfig:")
    print(f"      D_MODEL: int = {params.get('d_model', 64)}")
    print(f"      N_LAYERS: int = {params.get('n_layers', 3)}")
    print(f"      D_STATE: int = {params.get('d_state', 16)}")
    print(f"      D_CONV: int = {params.get('d_conv', 4)}")
    print(f"      EXPAND_FACTOR: int = {params.get('expand', 2)}")
    print(f"      DROPOUT: float = {params.get('dropout', 0.15):.2f}")
    print(f"      LEARNING_RATE: float = {params.get('lr', 5e-4):.2e}")
    print(f"      WEIGHT_DECAY: float = {params.get('weight_decay', 1e-4):.2e}")
    print(f"      BATCH_SIZE: int = {params.get('batch_size', 256)}")
    print(f"      ALPHA_DIRECTION: float = {params.get('alpha_direction', 1.0):.2f}")
    print(f"      BETA_VOLATILITY: float = {params.get('beta_volatility', 0.5):.2f}")
    print(f"      GAMMA_REGIME: float = {params.get('gamma_regime', 0.0):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search for Mamba XAUUSD")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Time limit in seconds")
    parser.add_argument("--resume", type=str, default=None, help="Resume from JSON file")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    args = parser.parse_args()

    best_params = run_search(
        n_trials=args.trials,
        timeout=args.timeout,
        resume_from=args.resume,
        device=args.device,
    )

    if best_params:
        print_recommended_config(best_params)
