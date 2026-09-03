"""
Point-in-Time Dataset implementation (1 snapshot row per customer with dynamic time-series features).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.base import BaseDataset, SplitResult
from src.data.weights import apply_feature_filters, compute_dynamic_sample_weights
from src.features.preprocessor import ChurnFeaturePreprocessor


class PointInTimeTimeSeriesDataset(BaseDataset):
    """Chiến Lược 4: Point-in-Time Customer-Level Dataset with Dynamic Time-Series Rolling Features.

    Each unique customer is represented by exactly 1 row (their most recent historical snapshot point),
    retaining all 42 high-dimensional rolling, trend, lag, and slope features.
    """

    def __init__(
        self,
        data_path: str = "data/processed/churn_feature_dataset_processed.csv",
        test_size: float = 0.2,
        val_size: float = 0.15,
        test_start_month: Optional[str] = "2026-01",
        stratify: bool = True,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.test_start_month = test_start_month
        self.stratify = stratify
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        data_path = kwargs.get("data_path", self.data_path)
        test_size = kwargs.get("test_size", self.test_size)
        val_size = kwargs.get("val_size", self.val_size)
        test_start_month = kwargs.get("test_start_month", self.test_start_month)
        stratify_flag = kwargs.get("stratify", self.stratify)
        seed = kwargs.get("seed", self.seed)
        use_cv = kwargs.get("use_cv", False)
        decay_half_life = kwargs.get("decay_half_life", None)
        customer_weight_power = kwargs.get("customer_weight_power", 0.0)
        use_usage_weight = kwargs.get("use_usage_weight", False)
        drop_low_mi = kwargs.get("drop_low_mi", False)
        drop_collinear = kwargs.get("drop_collinear", False)

        print(f"[DATASET] Loading Point-in-Time time-series dataset from '{data_path}'...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        print(f"[DATASET] Loaded {df.shape[0]:,} raw snapshots, {df['customer_id'].nunique():,} unique customers")

        preprocessor = ChurnFeaturePreprocessor(merge_static_master=False)
        df = preprocessor.transform(df)

        exclude_cols = ["customer_id", "snapshot_month", "label_churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        feature_cols = apply_feature_filters(
            feature_cols=feature_cols,
            drop_low_mi=drop_low_mi,
            drop_collinear=drop_collinear,
        )

        if test_start_month:
            df_hist = df[df["snapshot_month"] < test_start_month].sort_values("snapshot_month")
            df_test_raw = df[df["snapshot_month"] >= test_start_month].sort_values("snapshot_month")

            df_train_pit = df_hist.groupby("customer_id").last().reset_index()
            df_test_pit = df_test_raw.groupby("customer_id").last().reset_index()

            if use_cv or val_size <= 0:
                df_train = df_train_pit
                df_val = df_test_pit
            else:
                strat = df_train_pit["label_churn"].astype(int) if stratify_flag else None
                df_train, df_val = train_test_split(
                    df_train_pit, test_size=val_size, random_state=seed, stratify=strat
                )
            df_test = df_test_pit
            split_type_name = "temporal_point_in_time"
        else:
            df_pit = df.sort_values("snapshot_month").groupby("customer_id").last().reset_index()
            strat_y = df_pit["label_churn"].astype(int) if stratify_flag else None
            df_train_full, df_test = train_test_split(
                df_pit, test_size=test_size, random_state=seed, stratify=strat_y
            )
            if use_cv or val_size <= 0:
                df_train = df_train_full
                df_val = df_test
            else:
                strat_val = df_train_full["label_churn"].astype(int) if stratify_flag else None
                df_train, df_val = train_test_split(
                    df_train_full, test_size=val_size, random_state=seed, stratify=strat_val
                )
            split_type_name = "stratified_point_in_time"

        X_train = df_train[feature_cols].reset_index(drop=True)
        y_train = df_train["label_churn"].astype(int).reset_index(drop=True)
        train_times = df_train["snapshot_month"].values

        X_val = df_val[feature_cols].reset_index(drop=True)
        y_val = df_val["label_churn"].astype(int).reset_index(drop=True)

        X_test = df_test[feature_cols].reset_index(drop=True)
        y_test = df_test["label_churn"].astype(int).reset_index(drop=True)

        has_weights = (
            (decay_half_life and float(decay_half_life) > 0)
            or (customer_weight_power and float(customer_weight_power) > 0)
            or use_usage_weight
        )
        train_weights = None
        if has_weights:
            train_weights = compute_dynamic_sample_weights(
                df=df_train,
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
                "dataset_name": "point_in_time",
                "data_path": data_path,
                "split_type": split_type_name,
                "strategy": "Chiến Lược 4: Point-in-Time Dynamic Features",
                "train_time_series": train_times,
                "decay_half_life": decay_half_life,
                "customer_weight_power": customer_weight_power,
                "use_usage_weight": use_usage_weight,
            },
        )
        result.print_summary()
        return result
