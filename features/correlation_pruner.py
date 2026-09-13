"""
Multicollinearity Reduction & Correlation Pruning Module.
Detects and eliminates redundant, highly correlated features (|r| > threshold)
to prevent neural network overfitting and parameter bloat.
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class CorrelationPruner:
    """
    Identifies and removes collinear features using pairwise correlation analysis.
    Fit strictly on historical training data to prevent lookahead data leakage.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.dropped_features_: List[str] = []
        self.selected_features_: List[str] = []

    def fit(self, df_features: pd.DataFrame) -> "CorrelationPruner":
        """
        Computes pairwise correlation matrix and identifies redundant columns to drop.
        """
        # Ensure only numeric columns are analyzed
        numeric_df = df_features.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty:
            self.selected_features_ = list(df_features.columns)
            return self

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop: Set[str] = set()
        for col in upper_tri.columns:
            # If any earlier column is highly correlated with 'col', drop 'col' (the later one)
            high_corr_with_col = upper_tri.index[upper_tri[col] > self.threshold].tolist()
            if high_corr_with_col:
                # 'col' is correlated with a previously kept feature → drop 'col'
                to_drop.add(col)

        self.dropped_features_ = sorted(list(to_drop))
        self.selected_features_ = [c for c in df_features.columns if c not in to_drop]

        print(f"[INFO] Correlation Pruning (threshold={self.threshold}):")
        print(f"       Initial Features : {df_features.shape[1]}")
        print(f"       Dropped Redundant: {len(self.dropped_features_)}")
        print(f"       Selected Features: {len(self.selected_features_)}")
        return self

    def transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Filters dataframe to keep only the selected non-redundant features.
        """
        if not self.selected_features_:
            return df_features
        available_cols = [c for c in self.selected_features_ if c in df_features.columns]
        return df_features[available_cols]

    def fit_transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in a single step."""
        return self.fit(df_features).transform(df_features)
