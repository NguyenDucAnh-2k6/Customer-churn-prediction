"""
TimeSeries Dataset implementations for monthly snapshot panel data.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.base import BaseDataset, SplitResult
from src.data.weights import apply_feature_filters, compute_dynamic_sample_weights
from src.features.preprocessor import ChurnFeaturePreprocessor


class TimeSeriesDataset(BaseDataset):
    """Dataset with monthly snapshot time-series structure (churn_feature_dataset_processed.csv)."""

    def __init__(
        self,
        data_path: str = "data/processed/churn_feature_dataset_processed.csv",
        train_end_month: str = "2025-09",
        val_end_month: str = "2025-12",
        test_start_month: str = "2026-01",
        split_strategy: str = "time_based",
        test_size: float = 0.2,
        val_size: float = 0.15,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.train_end_month = train_end_month
        self.val_end_month = val_end_month
        self.test_start_month = test_start_month
        self.split_strategy = split_strategy
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        data_path = kwargs.get("data_path", self.data_path)
        train_end_month = kwargs.get("train_end_month", self.train_end_month)
        val_end_month = kwargs.get("val_end_month", self.val_end_month)
        test_start_month = kwargs.get("test_start_month", self.test_start_month)
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
        stock_features = kwargs.get("stock_features", True)

        print(f"[DATASET] Loading time-series dataset from '{data_path}'...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        print(f"[DATASET] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns, {df['customer_id'].nunique():,} unique customers")

        preprocessor = ChurnFeaturePreprocessor(merge_static_master=False, include_stock_features=stock_features)
        df = preprocessor.transform(df)

        exclude_cols = ["customer_id", "snapshot_month", "snapshot_date", "account_status", "label_churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        train_groups = None

        if split_strategy in ["group_stratified", "stratified_group", "customer_stratified"]:
            from sklearn.model_selection import StratifiedGroupKFold

            cust_churn = df.groupby("customer_id")["label_churn"].max()
            strat_label = df["customer_id"].map(cust_churn)

            n_splits_test = int(round(1.0 / test_size)) if test_size > 0 else 5
            sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)

            for tr_idx, te_idx in sgkf_test.split(df, strat_label, df["customer_id"]):
                train_full_mask = np.zeros(len(df), dtype=bool)
                train_full_mask[tr_idx] = True
                test_mask = np.zeros(len(df), dtype=bool)
                test_mask[te_idx] = True
                break

            if use_cv:
                train_mask = train_full_mask
                val_mask = test_mask
                split_type_name = "group_stratified_cv"
            elif val_size > 0:
                df_tr_full = df[train_full_mask].reset_index()
                strat_tr = df_tr_full["customer_id"].map(cust_churn)
                n_splits_val = int(round(1.0 / val_size))
                sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)

                for sub_tr_idx, sub_va_idx in sgkf_val.split(df_tr_full, strat_tr, df_tr_full["customer_id"]):
                    actual_tr_idx = df_tr_full.loc[sub_tr_idx, "index"].values
                    actual_va_idx = df_tr_full.loc[sub_va_idx, "index"].values
                    train_mask = np.zeros(len(df), dtype=bool)
                    train_mask[actual_tr_idx] = True
                    val_mask = np.zeros(len(df), dtype=bool)
                    val_mask[actual_va_idx] = True
                    break
                split_type_name = "group_stratified_holdout"
            else:
                train_mask = train_full_mask
                val_mask = test_mask
                split_type_name = "group_stratified_holdout"

            train_groups = df.loc[train_mask, "customer_id"].values
        else:
            test_mask = df["snapshot_month"] >= test_start_month
            if use_cv:
                train_mask = df["snapshot_month"] < test_start_month
                val_mask = (df["snapshot_month"] > train_end_month) & (df["snapshot_month"] <= val_end_month)
                split_type_name = "walk_forward_cv"
            else:
                train_mask = df["snapshot_month"] <= train_end_month
                val_mask = (df["snapshot_month"] > train_end_month) & (df["snapshot_month"] <= val_end_month)
                split_type_name = "time_based_holdout"

        behavioral_only = kwargs.get("behavioral_only", False) or kwargs.get("drop_static_tiers", False)

        feature_cols = apply_feature_filters(
            feature_cols=feature_cols,
            drop_low_mi=drop_low_mi,
            drop_collinear=drop_collinear,
            behavioral_only=behavioral_only,
            df_train=df.loc[train_mask],
            target_col="label_churn",
        )

        X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train = df.loc[train_mask, "label_churn"].astype(int).reset_index(drop=True)
        train_times = df.loc[train_mask, "snapshot_month"].values

        X_val = df.loc[val_mask, feature_cols].reset_index(drop=True)
        y_val = df.loc[val_mask, "label_churn"].astype(int).reset_index(drop=True)

        X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)
        y_test = df.loc[test_mask, "label_churn"].astype(int).reset_index(drop=True)

        has_weights = (
            (decay_half_life and float(decay_half_life) > 0)
            or (customer_weight_power and float(customer_weight_power) > 0)
            or use_usage_weight
        )
        train_weights = None
        if has_weights:
            train_weights = compute_dynamic_sample_weights(
                df=df.loc[train_mask],
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
                "dataset_name": "timeseries",
                "data_path": data_path,
                "split_type": split_type_name,
                "split_strategy": split_strategy,
                "train_end_month": train_end_month,
                "val_end_month": val_end_month,
                "test_start_month": test_start_month,
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


class GroupStratifiedTimeSeriesDataset(TimeSeriesDataset):
    """TimeSeries dataset using Customer-Level Stratified Group Splitting (Zero Customer Leakage)."""
    def __init__(self, **kwargs: Any):
        kwargs["split_strategy"] = "group_stratified"
        super().__init__(**kwargs)
