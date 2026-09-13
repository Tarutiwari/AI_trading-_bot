"""
End-to-End Feature Pipeline Orchestrator.
Coordinates multi-timeframe alignment, microstructure extraction,
session encodings, Triple Barrier labeling, and correlation pruning.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig
from features.multi_timeframe import MultiTimeframeAligner
from features.sessions import SessionFeatureExtractor
from features.labeler import TripleBarrierLabeler
from features.correlation_pruner import CorrelationPruner


class FeaturePipeline:
    """
    Master feature engineering and labeling pipeline for Gold (XAUUSD).
    Guarantees strict zero-lookahead bias and broker scale invariance.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.mtf_aligner = MultiTimeframeAligner()
        self.session_extractor = SessionFeatureExtractor()
        self.labeler = TripleBarrierLabeler(
            tp_multiplier=config.labeling.TP_ATR_MULTIPLIER,
            sl_multiplier=config.labeling.SL_ATR_MULTIPLIER,
            max_holding_bars=config.labeling.MAX_HOLDING_BARS,
        )
        self.pruner = CorrelationPruner(threshold=config.features.CORRELATION_THRESHOLD)
        self.scaler = RobustScaler()
        self.feature_names_: list = []

    def build_dataset(
        self,
        raw_data: Dict[str, pd.DataFrame],
        is_training: bool = True,
        save_to_disk: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes multi-timeframe raw data into model-ready features X and labels Y.
        """
        primary_tf = self.config.instrument.PRIMARY_TIMEFRAME
        if primary_tf not in raw_data or raw_data[primary_tf].empty:
            raise ValueError(f"Primary timeframe [{primary_tf}] data missing from raw inputs.")

        primary_df = raw_data[primary_tf]
        higher_tfs = {k: v for k, v in raw_data.items() if k != primary_tf}

        print(f"\n=======================================================")
        print(f"  Running Master Feature Pipeline for {primary_tf}")
        print(f"=======================================================")

        # 1. Multi-Timeframe Alignment & Lagging
        aligned_features = self.mtf_aligner.extract_and_align(
            primary_df=primary_df,
            higher_tf_dfs=higher_tfs,
            primary_tf_name=primary_tf,
        )

        # 2. Session Encodings, Cyclical Time & News Distance
        session_features = self.session_extractor.compute_session_features(primary_df.index)
        
        # Merge all feature blocks
        full_features = pd.concat([aligned_features, session_features], axis=1)

        # Drop constant (zero-variance) features dynamically
        constant_cols = [col for col in full_features.columns if full_features[col].nunique() <= 1]
        if constant_cols:
            print(f"[INFO] Dropping {len(constant_cols)} constant (zero-variance) features: {constant_cols}")
            full_features.drop(columns=constant_cols, inplace=True)

        # 3. Triple Barrier Labeling
        atr_series = aligned_features["atr"]
        target_df = self.labeler.generate_labels(df=primary_df, atr_series=atr_series)

        # 4. Clean NaNs created by rolling indicators & higher-TF lagging
        # Drop the warmup period
        valid_mask = ~full_features.isnull().any(axis=1) & ~target_df.isnull().any(axis=1)
        # Also drop the last max_bars where future labels are undefined
        valid_mask.iloc[-self.config.labeling.MAX_HOLDING_BARS:] = False

        X_clean = full_features[valid_mask].copy()
        y_clean = target_df[valid_mask].copy()

        # 5. Correlation Pruning (Fit only if training; Load from disk if inference)
        pruner_path = self.config.paths.PROCESSED_DATA_DIR / "correlation_pruner.joblib"
        if is_training:
            X_clean = self.pruner.fit_transform(X_clean)
            self.feature_names_ = list(X_clean.columns)
        else:
            if pruner_path.exists():
                import joblib as _joblib
                self.pruner = _joblib.load(pruner_path)
            X_clean = self.pruner.transform(X_clean)

        print(f"[SUCCESS] Dataset built successfully: {len(X_clean):,} samples x {X_clean.shape[1]} features.")

        # 6. Save Processed Artifacts
        if save_to_disk:
            self._save_processed(X_clean, y_clean)

        return X_clean, y_clean

    def _save_processed(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Save feature matrix X and target labels Y to processed directory."""
        proc_dir = self.config.paths.PROCESSED_DATA_DIR
        x_path = proc_dir / "X_features.parquet"
        y_path = proc_dir / "y_labels.parquet"
        pruner_path = proc_dir / "correlation_pruner.joblib"

        X.to_parquet(x_path)
        y.to_parquet(y_path)
        joblib.dump(self.pruner, pruner_path)
        print(f"[SAVED] Saved processed features -> {x_path.name} & labels -> {y_path.name}")


if __name__ == "__main__":
    import glob
    pipeline = FeaturePipeline(CONFIG)
    raw_dir = CONFIG.paths.RAW_DATA_DIR
    
    print(f"[INFO] Scanning raw data directory: {raw_dir}")
    # Discover available raw files
    raw_files = list(raw_dir.glob("*_raw.parquet"))
    if not raw_files:
        raw_files = list(raw_dir.glob("*_raw.csv"))

    if not raw_files:
        print("[ERROR] No raw files found in data/raw/. Please run collectors/mt5_collector.py first.")
    else:
        raw_data = {}
        for f in raw_files:
            # Extract timeframe from filename (e.g. XAUUSD_t_M5_raw.parquet -> M5)
            parts = f.stem.split("_")
            tf = [p for p in parts if p in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]]
            if tf:
                tf_name = tf[0]
                if f.suffix == ".parquet":
                    raw_data[tf_name] = pd.read_parquet(f)
                else:
                    raw_data[tf_name] = pd.read_csv(f, index_col=0, parse_dates=True)
                print(f"[LOADED] {tf_name}: {len(raw_data[tf_name]):,} bars from {f.name}")

        if CONFIG.instrument.PRIMARY_TIMEFRAME in raw_data:
            X, y = pipeline.build_dataset(raw_data, is_training=True, save_to_disk=True)
            print(f"\n[DONE] Successfully processed & saved {len(X):,} samples with {X.shape[1]} features.")
        else:
            print(f"[ERROR] Primary timeframe [{CONFIG.instrument.PRIMARY_TIMEFRAME}] not found in raw files.")
