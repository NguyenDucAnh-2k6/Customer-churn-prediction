"""
Round 3 Point-in-Time Dataset implementation (data/processed/round3/).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.base import BaseDataset, SplitResult
from src.data.weights import apply_feature_filters


class Round3Dataset(BaseDataset):
    """
    Round 3 Point-in-Time 34-feature Customer Dataset (data/processed/round3/).
    Features: 1 row / customer, rolling 60d + all-time statistics across Silver Lakehouse.
    Zero customer leakage and zero target leakage.
    """

    def __init__(
        self,
        train_path: str = "data/processed/round3/churn_train.csv",
        test_path: str = "data/processed/round3/churn_test.csv",
        target_col: str = "churn",
        val_size: float = 0.2,
        seed: int = 42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.val_size = val_size
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        train_path = kwargs.get("train_path", self.train_path)
        test_path = kwargs.get("test_path", self.test_path)
        target_col = kwargs.get("target_col", self.target_col)
        val_size = kwargs.get("val_size", self.val_size)
        seed = kwargs.get("seed", self.seed)
        use_cv = kwargs.get("use_cv", False)
        drop_low_mi = kwargs.get("drop_low_mi", False)
        drop_collinear = kwargs.get("drop_collinear", False)

        print(f"[DATASET] Loading Round 3 dataset from '{os.path.dirname(train_path)}'...")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing Round 3 dataset files: '{train_path}' or '{test_path}'")

        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
        print(f"[DATASET] Loaded Train: {len(df_train_raw):,} rows, Test: {len(df_test_raw):,} rows")

        # Exclude non-feature columns
        non_feature_cols = ["customer_id", target_col, "cv_fold", "Unnamed: 0"]
        feature_cols = [c for c in df_train_raw.columns if c not in non_feature_cols]

        # Categorical column encoding (gender, plan_tier)
        cat_cols = ["gender", "plan_tier"]
        for c in cat_cols:
            if c in feature_cols:
                cats = {val: idx for idx, val in enumerate(df_train_raw[c].dropna().unique())}
                df_train_raw[c] = df_train_raw[c].map(cats).fillna(-1).astype(int)
                df_test_raw[c] = df_test_raw[c].map(cats).fillna(-1).astype(int)

        # Validation split from training data
        if use_cv or val_size <= 0:
            df_train = df_train_raw
            df_val = df_test_raw
            split_type_name = "round3_stratified_cv"
        else:
            if "cv_fold" in df_train_raw.columns and df_train_raw["cv_fold"].nunique() > 1:
                df_train = df_train_raw[df_train_raw["cv_fold"] != 0].reset_index(drop=True)
                df_val = df_train_raw[df_train_raw["cv_fold"] == 0].reset_index(drop=True)
                split_type_name = "round3_precomputed_fold_val"
            else:
                from sklearn.model_selection import train_test_split
                df_train, df_val = train_test_split(
                    df_train_raw,
                    test_size=val_size,
                    random_state=seed,
                    stratify=df_train_raw[target_col].astype(int),
                )
                split_type_name = "round3_stratified_holdout"

        df_test = df_test_raw

        feature_cols = apply_feature_filters(
            feature_cols=feature_cols,
            drop_low_mi=drop_low_mi,
            drop_collinear=drop_collinear,
            df_train=df_train,
            target_col=target_col,
        )

        X_train = df_train[feature_cols].reset_index(drop=True)
        y_train = df_train[target_col].astype(int).reset_index(drop=True)

        X_val = df_val[feature_cols].reset_index(drop=True)
        y_val = df_val[target_col].astype(int).reset_index(drop=True)

        X_test = df_test[feature_cols].reset_index(drop=True)
        y_test = df_test[target_col].astype(int).reset_index(drop=True)

        train_groups = df_train["customer_id"].values if "customer_id" in df_train.columns else None

        result = SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            train_weights=None,
            metadata={
                "dataset_name": "round3",
                "train_path": train_path,
                "test_path": test_path,
                "target_col": target_col,
                "split_type": split_type_name,
                "strategy": "Round 3: Point-in-Time 34-Feature Customer Stats",
                "train_groups": train_groups,
            },
        )
        result.print_summary()
        return result
