"""
Round 3 Point-in-Time Dataset implementation (data/processed/round3/).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.base import BaseDataset, SplitResult
from src.data.weights import apply_feature_filters
from src.features.preprocessor import ChurnFeaturePreprocessor


class Round3Dataset(BaseDataset):
    """
    Round 3 Point-in-Time Customer Dataset (data/processed/round3/).
    Features: 1 row / customer, rolling 30d/60d, contract dynamics, recency & stock indicators.
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
        stock_features = kwargs.get("stock_features", True)
        dynamic_contract = kwargs.get("dynamic_contract_features", True)

        print(f"[DATASET] Loading Round 3 dataset from '{os.path.dirname(train_path)}'...")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing Round 3 dataset files: '{train_path}' or '{test_path}'")

        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)
        print(f"[DATASET] Loaded Train: {len(df_train_raw):,} rows, Test: {len(df_test_raw):,} rows")

        # Feature Preprocessing & Enrichment (Stock Features + Contract Dynamics)
        preprocessor = ChurnFeaturePreprocessor(
            merge_static_master=False,
            include_stock_features=stock_features,
            include_contract_dynamics=dynamic_contract,
        )
        df_train_raw = preprocessor.fit_transform(df_train_raw)
        df_test_raw = preprocessor.transform(df_test_raw)

        # Exclude non-feature columns and target aliases
        non_feature_cols = [
            "customer_id", "churn", "label_churn", "label_churn_30d", "churn_30d",
            target_col, "cv_fold", "Unnamed: 0", "snapshot_dt", "snapshot_date",
            "snapshot_month", "churn_reason", "account_status"
        ]
        feature_cols = [c for c in df_train_raw.columns if c not in non_feature_cols]

        from src.features.financial_indicators import STOCK_FEATURE_COLS
        if not stock_features:
            dropped_stock = [c for c in feature_cols if c in STOCK_FEATURE_COLS]
            feature_cols = [c for c in feature_cols if c not in STOCK_FEATURE_COLS]
            print(f"[ABLATION] Excluded {len(dropped_stock)} Quantitative Stock/Market features in Point-in-Time Round 3.")

        # Categorical column encoding (gender, plan_tier, region, city, subscription_tier)
        cat_cols = ["gender", "plan_tier", "region", "city", "subscription_tier"]
        for c in cat_cols:
            if c in feature_cols:
                cats = {val: idx for idx, val in enumerate(df_train_raw[c].dropna().unique())}
                df_train_raw[c] = df_train_raw[c].map(cats).fillna(-1).astype(int)
                df_test_raw[c] = df_test_raw[c].map(cats).fillna(-1).astype(int)

        # Ensure all feature_cols are strictly numeric
        for c in feature_cols:
            if df_train_raw[c].dtype == object:
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
            filter_kde=kwargs.get("filter_kde", False),
            filter_categorical=kwargs.get("filter_categorical", False),
            filter_boxplot=kwargs.get("filter_boxplot", False),
            ks_threshold=kwargs.get("ks_threshold", 0.05),
            cramers_v_threshold=kwargs.get("cramers_v_threshold", 0.03),
            iv_threshold=kwargs.get("iv_threshold", 0.02),
            cohens_d_threshold=kwargs.get("cohens_d_threshold", 0.08),
            iqr_overlap_threshold=kwargs.get("iqr_overlap_threshold", 0.90),
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
                "strategy": "Round 3: Point-in-Time Enriched Customer Stats",
                "train_groups": train_groups,
                "stock_features": stock_features,
                "dynamic_contract_features": dynamic_contract,
            },
        )
        result.print_summary()
        return result
