"""
Round 3 (lqminh) Lakehouse Time-Series Panel Dataset.
File: data/processed/round3_timeseries/churn_timeseries_master.csv
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.base import BaseDataset, SplitResult
from src.data.weights import compute_dynamic_sample_weights, apply_feature_filters


class Round3TimeSeriesDataset(BaseDataset):
    """
    Round 3 Lakehouse (lqminh) Time-Series Panel Dataset.
    Generates monthly panel observations with multi-component churn target:
    1. Account deletion / closed account.
    2. Free tier inactive horizon.
    3. Downgrade inactive horizon.
    """

    def __init__(
        self,
        data_path: str = "data/processed/round3_timeseries/churn_timeseries_master.csv",
        target_col: str = "label_churn",
        val_size: float = 0.2,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.val_size = val_size
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        data_path = kwargs.get("data_path", self.data_path)
        target_col = kwargs.get("target_col", self.target_col)
        val_size = kwargs.get("val_size", self.val_size)
        seed = kwargs.get("seed", self.seed)
        use_cv = kwargs.get("use_cv", False)
        drop_low_mi = kwargs.get("drop_low_mi", False)
        drop_collinear = kwargs.get("drop_collinear", False)
        decay_half_life = kwargs.get("decay_half_life", None)
        customer_weight_power = kwargs.get("customer_weight_power", 0.0)
        stock_features = kwargs.get("stock_features", True)
        if isinstance(stock_features, str):
            stock_features = stock_features.lower() in ["true", "1", "yes"]

        print(f"[DATASET] Loading Round 3 TimeSeries dataset from '{data_path}' (stock_features={stock_features})...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"TimeSeries dataset not found at '{data_path}'. Please run build_round3_timeseries.py first.")

        df = pd.read_csv(data_path)
        print(f"[DATASET] Loaded {len(df):,} rows x {df.shape[1]} columns across {df['customer_id'].nunique():,} unique customers.")

        exclude_cols = [
            "customer_id",
            "snapshot_month",
            "snapshot_date",
            "churn",
            "label_churn",
            "label_churn_30d",
            "churn_30d",
            target_col,
            "gender",
            "plan_tier",
            "region",
            "city",
            "churn_reason",
            "account_status",
            "Unnamed: 0",
        ]
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        from src.features.financial_indicators import STOCK_FEATURE_COLS
        if not stock_features:
            dropped_stock = [c for c in feature_cols if c in STOCK_FEATURE_COLS]
            feature_cols = [c for c in feature_cols if c not in STOCK_FEATURE_COLS]
            print(f"[ABLATION] Excluded {len(dropped_stock)} Quantitative Stock/Market features: {dropped_stock}")

        # Customer-Stratified Group Split (Zero Customer Leakage)
        cust_churn = df.groupby("customer_id")[target_col].max()
        strat_label = df["customer_id"].map(cust_churn)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(df, strat_label, df["customer_id"]))

        df_train_full = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        if use_cv or val_size <= 0:
            df_train = df_train_full
            df_val = df_test
            split_type_name = "round3_timeseries_group_stratified_cv"
        else:
            cust_tr = df_train_full.groupby("customer_id")[target_col].max()
            sgkf_val = StratifiedGroupKFold(n_splits=int(1 / val_size) if val_size > 0 else 5, shuffle=True, random_state=seed)
            sub_tr_idx, val_idx = next(sgkf_val.split(df_train_full, df_train_full["customer_id"].map(cust_tr), df_train_full["customer_id"]))
            df_train = df_train_full.iloc[sub_tr_idx].reset_index(drop=True)
            df_val = df_train_full.iloc[val_idx].reset_index(drop=True)
            split_type_name = "round3_timeseries_group_stratified_holdout"

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

        train_weights = compute_dynamic_sample_weights(
            df=df_train,
            decay_half_life=decay_half_life,
            customer_weight_power=customer_weight_power,
            use_usage_weight=False,
            snapshot_month_col="snapshot_month",
        )

        X_train = df_train[feature_cols].reset_index(drop=True)
        y_train = df_train[target_col].astype(int).reset_index(drop=True)

        X_val = df_val[feature_cols].reset_index(drop=True)
        y_val = df_val[target_col].astype(int).reset_index(drop=True)

        X_test = df_test[feature_cols].reset_index(drop=True)
        y_test = df_test[target_col].astype(int).reset_index(drop=True)

        train_groups = df_train["customer_id"].values

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
                "dataset_name": "round3_timeseries",
                "data_path": data_path,
                "target_col": target_col,
                "split_type": split_type_name,
                "strategy": "Round 3 Lakehouse (lqminh) Time-Series Panel",
                "train_groups": train_groups,
            },
        )
        result.print_summary()
        return result
