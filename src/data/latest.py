"""
Pre-split Latest Dataset implementations (data/processed/latest/).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.base import BaseDataset, SplitResult
from src.data.weights import apply_feature_filters, compute_dynamic_sample_weights
from src.features.preprocessor import ChurnFeaturePreprocessor


class PreSplitLatestDataset(BaseDataset):
    """Pre-split dataset provided by team in data/processed/latest/ (churn_train.csv, churn_val.csv, churn_test.csv).
    Supports both default pre-split loading and Customer-Level Stratified Group Splitting across all snapshots.
    """

    def __init__(
        self,
        train_path: str = "data/processed/latest/churn_train.csv",
        val_path: str = "data/processed/latest/churn_val.csv",
        test_path: str = "data/processed/latest/churn_test.csv",
        target_col: str = "churn_30d",
        split_strategy: str = "time_based",
        test_size: float = 0.2,
        val_size: float = 0.15,
        seed: int = 42,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.target_col = target_col
        self.split_strategy = split_strategy
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        train_path = kwargs.get("train_path", self.train_path)
        val_path = kwargs.get("val_path", self.val_path)
        test_path = kwargs.get("test_path", self.test_path)
        target_col = kwargs.get("target_col", self.target_col)
        split_strategy = kwargs.get("split_strategy", self.split_strategy)
        test_size = kwargs.get("test_size", self.test_size)
        val_size = kwargs.get("val_size", self.val_size)
        seed = kwargs.get("seed", self.seed)
        use_cv = kwargs.get("use_cv", False)
        decay_half_life = kwargs.get("decay_half_life", None)
        customer_weight_power = kwargs.get("customer_weight_power", 0.0)
        use_usage_weight = kwargs.get("use_usage_weight", False)
        drop_low_mi = kwargs.get("drop_low_mi", False)
        drop_collinear = kwargs.get("drop_collinear", False)
        behavioral_only = kwargs.get("behavioral_only", False) or kwargs.get("drop_static_tiers", False)
        stock_features = kwargs.get("stock_features", True)

        print(f"[DATASET] Loading pre-split dataset from '{os.path.dirname(train_path)}'...")
        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"One or more pre-split files missing in '{os.path.dirname(train_path)}'")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        print(f"[DATASET] Loaded Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")

        all_churn_targets = [
            "churn_30d", "churn_60d", "churn_case1_30d",
            "churn_case2_30d", "churn_case2a_30d", "churn_case2b_30d"
        ]
        non_feature_cols = ["customer_id", "snapshot_month", "snapshot_date", "account_status"] + all_churn_targets
        cat_cols = ["gender", "region", "city"]

        train_groups = None

        if split_strategy in ["group_stratified", "stratified_group", "customer_stratified"]:
            from sklearn.model_selection import StratifiedGroupKFold

            df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
            print(f"[DATASET] Combined Pool: {len(df_all):,} snapshots across {df_all['customer_id'].nunique():,} unique customers")

            preprocessor = ChurnFeaturePreprocessor(merge_static_master=True, include_stock_features=stock_features)
            df_all = preprocessor.transform(df_all)

            feature_cols = [c for c in df_all.columns if c not in non_feature_cols]

            cust_churn = df_all.groupby("customer_id")[target_col].max()
            strat_label = df_all["customer_id"].map(cust_churn)

            n_splits_test = int(round(1.0 / test_size)) if test_size > 0 else 5
            sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)

            for tr_idx, te_idx in sgkf_test.split(df_all, strat_label, df_all["customer_id"]):
                train_full_mask = np.zeros(len(df_all), dtype=bool)
                train_full_mask[tr_idx] = True
                test_mask = np.zeros(len(df_all), dtype=bool)
                test_mask[te_idx] = True
                break

            if use_cv:
                train_mask = train_full_mask
                val_mask = test_mask
                split_type_name = "group_stratified_cv"
            elif val_size > 0:
                df_tr_full = df_all[train_full_mask].reset_index()
                strat_tr = df_tr_full["customer_id"].map(cust_churn)
                n_splits_val = int(round(1.0 / val_size))
                sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)

                for sub_tr_idx, sub_va_idx in sgkf_val.split(df_tr_full, strat_tr, df_tr_full["customer_id"]):
                    actual_tr_idx = df_tr_full.loc[sub_tr_idx, "index"].values
                    actual_va_idx = df_tr_full.loc[sub_va_idx, "index"].values
                    train_mask = np.zeros(len(df_all), dtype=bool)
                    train_mask[actual_tr_idx] = True
                    val_mask = np.zeros(len(df_all), dtype=bool)
                    val_mask[actual_va_idx] = True
                    break
                split_type_name = "group_stratified_holdout"
            else:
                train_mask = train_full_mask
                val_mask = test_mask
                split_type_name = "group_stratified_holdout"

            train_groups = df_all.loc[train_mask, "customer_id"].values

            feature_cols = apply_feature_filters(
                feature_cols=feature_cols,
                drop_low_mi=drop_low_mi,
                drop_collinear=drop_collinear,
                behavioral_only=behavioral_only,
                df_train=df_all.loc[train_mask],
                target_col=target_col,
            )

            cat_cols_present = [c for c in cat_cols if c in feature_cols]
            X_all = df_all[feature_cols].copy()
            for col in cat_cols_present:
                cats = {val: idx for idx, val in enumerate(df_all.loc[train_mask, col].unique())}
                X_all[col] = X_all[col].map(cats).fillna(-1).astype(int)

            X_train = X_all.loc[train_mask].reset_index(drop=True)
            y_train = df_all.loc[train_mask, target_col].astype(int).reset_index(drop=True)
            train_times = df_all.loc[train_mask, "snapshot_month"].values

            X_val = X_all.loc[val_mask].reset_index(drop=True)
            y_val = df_all.loc[val_mask, target_col].astype(int).reset_index(drop=True)

            X_test = X_all.loc[test_mask].reset_index(drop=True)
            y_test = df_all.loc[test_mask, target_col].astype(int).reset_index(drop=True)

            target_df_for_weights = df_all.loc[train_mask]
        else:
            preprocessor = ChurnFeaturePreprocessor(merge_static_master=True, include_stock_features=stock_features)
            df_train = preprocessor.transform(df_train)
            df_val = preprocessor.transform(df_val)
            df_test = preprocessor.transform(df_test)

            feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

            df_train_eval = pd.concat([df_train, df_val], ignore_index=True) if use_cv else df_train

            feature_cols = apply_feature_filters(
                feature_cols=feature_cols,
                drop_low_mi=drop_low_mi,
                drop_collinear=drop_collinear,
                behavioral_only=behavioral_only,
                df_train=df_train_eval,
                target_col=target_col,
            )

            cat_cols_present = [c for c in cat_cols if c in feature_cols]

            if use_cv:
                df_train_full = pd.concat([df_train, df_val], ignore_index=True)
                X_train = df_train_full[feature_cols].copy()
                X_val = df_val[feature_cols].copy()
                X_test = df_test[feature_cols].copy()

                for col in cat_cols_present:
                    cats = {val: idx for idx, val in enumerate(df_train_full[col].unique())}
                    X_train[col] = X_train[col].map(cats).fillna(-1).astype(int)
                    X_val[col] = X_val[col].map(cats).fillna(-1).astype(int)
                    X_test[col] = X_test[col].map(cats).fillna(-1).astype(int)

                y_train = df_train_full[target_col].astype(int)
                y_val = df_val[target_col].astype(int)
                y_test = df_test[target_col].astype(int)
                train_times = df_train_full["snapshot_month"].values
                split_type_name = "walk_forward_cv_team"
                target_df_for_weights = df_train_full
            else:
                X_train = df_train[feature_cols].copy()
                X_val = df_val[feature_cols].copy()
                X_test = df_test[feature_cols].copy()

                for col in cat_cols_present:
                    cats = {val: idx for idx, val in enumerate(df_train[col].unique())}
                    X_train[col] = X_train[col].map(cats).fillna(-1).astype(int)
                    X_val[col] = X_val[col].map(cats).fillna(-1).astype(int)
                    X_test[col] = X_test[col].map(cats).fillna(-1).astype(int)

                y_train = df_train[target_col].astype(int)
                y_val = df_val[target_col].astype(int)
                y_test = df_test[target_col].astype(int)
                train_times = df_train["snapshot_month"].values
                split_type_name = "pre_split_team"
                target_df_for_weights = df_train

        has_weights = (
            (decay_half_life and float(decay_half_life) > 0)
            or (customer_weight_power and float(customer_weight_power) > 0)
            or use_usage_weight
        )
        train_weights = None
        if has_weights:
            train_weights = compute_dynamic_sample_weights(
                df=target_df_for_weights,
                snapshot_month_col="snapshot_month",
                customer_id_col="customer_id",
                decay_half_life=decay_half_life,
                customer_weight_power=customer_weight_power,
                use_usage_weight=use_usage_weight,
            )

        result = SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            train_weights=train_weights,
            metadata={
                "dataset_name": "latest",
                "train_path": train_path,
                "val_path": val_path,
                "test_path": test_path,
                "target_col": target_col,
                "split_type": split_type_name,
                "split_strategy": split_strategy,
                "train_groups": train_groups,
                "train_time_series": train_times,
                "decay_half_life": decay_half_life,
                "customer_weight_power": customer_weight_power,
                "use_usage_weight": use_usage_weight,
                "time_col": "snapshot_month",
            },
        )
        result.print_summary()
        return result


class GroupStratifiedLatestDataset(PreSplitLatestDataset):
    """Latest dataset (Static Master + Dynamic TimeSeries) using Customer-Level Stratified Group Splitting (Zero Customer Leakage)."""
    def __init__(self, **kwargs: Any):
        kwargs["split_strategy"] = "group_stratified"
        super().__init__(**kwargs)
