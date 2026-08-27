"""
Concrete dataset implementations and Dataset Registry.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.base import BaseDataset, SplitResult
from src.features.preprocessor import ChurnFeaturePreprocessor
from src.features.selection import (
    make_mi_scores,
    filter_features_by_mi,
    filter_multicollinear_features,
)


def compute_dynamic_sample_weights(
    df: pd.DataFrame,
    snapshot_month_col: str = "snapshot_month",
    customer_id_col: str = "customer_id",
    decay_half_life: Optional[float] = None,
    customer_weight_power: float = 0.0,
    use_usage_weight: bool = False,
    active_days_col: str = "total_active_days_30d",
) -> np.ndarray:
    """Compute Multi-Factor Dynamic Sample Weights for time-series observations.

    Formula:
        W = w_time * w_cust * w_usage

    Where:
        - w_time = 2 ^ (- (T_max - t) / decay_half_life)  [Recency Decay Weighting]
        - w_cust = 1 / (N_customer ^ customer_weight_power)  [Customer Frequency Balancing]
        - w_usage = 1 + log(1 + active_days)  [Engagement Activity Scaling]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of training observations containing snapshot dates and customer IDs.
    snapshot_month_col : str, default='snapshot_month'
        Column name for snapshot month/date.
    customer_id_col : str, default='customer_id'
        Column name for customer ID.
    decay_half_life : float, optional
        Half-life in months for exponential recency decay.
    customer_weight_power : float, default=0.0
        Power alpha for inverse customer frequency balancing (e.g. 0.5 or 1.0).
    use_usage_weight : bool, default=False
        Whether to scale weights by engagement activity (active days).
    active_days_col : str, default='total_active_days_30d'
        Column name for 30-day active days.

    Returns
    -------
    np.ndarray
        Array of normalized positive sample weights with mean = 1.0.
    """
    n_samples = len(df)
    weights = np.ones(n_samples, dtype=float)
    applied_components = []

    # 1. Time-Series Recency Decay Weight
    if decay_half_life and float(decay_half_life) > 0 and snapshot_month_col in df.columns:
        dates = pd.to_datetime(df[snapshot_month_col])
        max_date = dates.max()
        if hasattr(dates, "dt"):
            month_diffs = (max_date.year - dates.dt.year) * 12 + (max_date.month - dates.dt.month)
        else:
            month_diffs = (max_date.year - dates.year) * 12 + (max_date.month - dates.month)
        w_time = np.power(2.0, -month_diffs.values / float(decay_half_life))
        weights *= w_time
        applied_components.append(f"RecencyDecay(half_life={decay_half_life}m)")

    # 2. Customer Frequency Balancing Weight
    if customer_weight_power and float(customer_weight_power) > 0 and customer_id_col in df.columns:
        cust_counts = df[customer_id_col].map(df[customer_id_col].value_counts()).values
        w_cust = np.power(cust_counts.astype(float), -float(customer_weight_power))
        weights *= w_cust
        applied_components.append(f"CustBalancing(power={customer_weight_power})")

    # 3. Usage Engagement Weighting
    if use_usage_weight and active_days_col in df.columns:
        active_days = np.maximum(0.0, df[active_days_col].fillna(0).values.astype(float))
        w_usage = 1.0 + np.log1p(active_days)
        weights *= w_usage
        applied_components.append("UsageEngagement")

    # Normalize weights so mean(weights) = 1.0
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w

    if applied_components:
        comp_str = " + ".join(applied_components)
        print(f"[WEIGHTS] Applied Dynamic Sample Weights [{comp_str}] | Min: {weights.min():.3f}, Mean: {weights.mean():.3f}, Max: {weights.max():.3f}")

    return np.asarray(weights, dtype=float)


# Pre-defined feature sets for ablation studies
LOW_MI_FEATURES = [
    "gender", "region", "city", "age",
    "has_unresolved_ticket", "has_marketing_click_30d",
    "avg_csat_score_missing", "num_tickets_90d", "avg_csat_score",
    "open_rate_30d", "is_declining_engagement", "reactivation_flag",
]

COLLINEAR_PAIRS_DROP = [
    "orders_last_90d",               # Duplicates orders_roll3m_sum (r=0.994)
    "days_since_last_login",          # Duplicates days_since_last_usage_event (r=0.965)
    "total_session_time_30d",         # Duplicates num_usage_events_30d (r=0.988)
    "num_usage_events_60d",          # Redundant with roll3m_sum (r=0.971)
    "total_active_days_60d",          # Highly collinear with 30d/90d (r=0.970)
    "total_active_days_90d",          # Highly collinear with num_usage_events_roll3m_sum (r=0.982)
]


STATIC_TIER_FEATURES = [
    "is_paid_tier", "subscription_tier", "gender", "region", "city", "age", "plan_tier"
]


def apply_feature_filters(
    feature_cols: List[str],
    drop_low_mi: bool = False,
    drop_collinear: bool = False,
    behavioral_only: bool = False,
    custom_drop_features: Optional[List[str]] = None,
    df_train: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    mi_threshold: float = 0.001,
) -> List[str]:
    """Filter feature columns based on Mutual Information (MI), Multicollinearity, and Behavioral settings.

    If `behavioral_only` is True, drops static tier and demographic features (Strategy 4).
    If `df_train` and `target_col` are provided, computes Mutual Information dynamically
    using `make_mi_scores` (replicating Untitled.ipynb with mutual_info_regression/classif).
    Otherwise, applies pre-computed feature ablation sets.
    """
    dropped = set()

    if behavioral_only:
        for f in STATIC_TIER_FEATURES:
            if f in feature_cols:
                dropped.add(f)

    if drop_low_mi:
        if df_train is not None and target_col is not None and target_col in df_train.columns:
            avail_cols = [c for c in feature_cols if c in df_train.columns and c not in dropped]
            X_sub = df_train[avail_cols]
            y_sub = df_train[target_col]
            retained_mi, _ = filter_features_by_mi(
                X_sub, y_sub, mi_threshold=mi_threshold, method="classif", verbose=True
            )
            low_mi = [c for c in avail_cols if c not in retained_mi]
            dropped.update(low_mi)
        else:
            for f in LOW_MI_FEATURES:
                if f in feature_cols:
                    dropped.add(f)

    if drop_collinear:
        if df_train is not None:
            avail_cols = [c for c in feature_cols if c in df_train.columns and c not in dropped]
            X_sub = df_train[avail_cols]
            retained_corr, _ = filter_multicollinear_features(X_sub, threshold=0.90, verbose=True)
            collinear_dropped = [c for c in avail_cols if c not in retained_corr]
            dropped.update(collinear_dropped)
        else:
            for f in COLLINEAR_PAIRS_DROP:
                if f in feature_cols:
                    dropped.add(f)

    if custom_drop_features:
        for f in custom_drop_features:
            if f in feature_cols:
                dropped.add(f)

    remaining = [f for f in feature_cols if f not in dropped]
    if dropped:
        print(f"[FEATURE SELECTION] Total Dropped {len(dropped)} features: {sorted(list(dropped))}")
        print(f"[FEATURE SELECTION] Final Retained {len(remaining)} features (from original {len(feature_cols)})")
    return remaining


def compute_exponential_decay_weights(
    dates_or_df: Any,
    half_life_months: Optional[float] = None,
    decay_half_life: Optional[float] = None,
    snapshot_month_col: str = "snapshot_month",
    **kwargs: Any,
) -> np.ndarray:
    """Backwards-compatible exponential recency decay weights calculator.
    
    Accepts either a DataFrame or an array/Series of snapshot dates, and supports
    both `decay_half_life` and legacy `half_life_months` argument names.
    """
    hl = decay_half_life if decay_half_life is not None else half_life_months
    if isinstance(dates_or_df, pd.DataFrame):
        return compute_dynamic_sample_weights(
            df=dates_or_df,
            snapshot_month_col=snapshot_month_col,
            decay_half_life=hl,
            **kwargs,
        )
    if dates_or_df is None or len(dates_or_df) == 0:
        return np.array([], dtype=float)

    dates = pd.to_datetime(dates_or_df)
    max_date = dates.max()
    if hasattr(dates, "dt"):
        month_diffs = (max_date.year - dates.dt.year) * 12 + (max_date.month - dates.dt.month)
    else:
        month_diffs = np.array([(max_date.year - d.year) * 12 + (max_date.month - d.month) for d in dates])

    if hl and float(hl) > 0:
        weights = np.power(2.0, -month_diffs.astype(float) / float(hl))
    else:
        weights = np.ones(len(dates_or_df), dtype=float)

    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w
    return np.asarray(weights, dtype=float)


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

        # Apply ChurnFeaturePreprocessor (Dynamic behavioral & financial features only, merge_static_master=False)
        preprocessor = ChurnFeaturePreprocessor(merge_static_master=False, include_stock_features=stock_features)
        df = preprocessor.transform(df)

        # Exclude ID, raw date strings, and target column
        exclude_cols = ["customer_id", "snapshot_month", "snapshot_date", "account_status", "label_churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        train_groups = None

        if split_strategy in ["group_stratified", "stratified_group", "customer_stratified"]:
            # Phương Án 1: Stratified Group Split theo Customer (Zero Customer Leakage)
            from sklearn.model_selection import StratifiedGroupKFold

            # Compute customer-level churn indicator for balanced stratification
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
                val_mask = test_mask  # In CV mode, validation is performed across group CV folds
                split_type_name = "group_stratified_cv"
            elif val_size > 0:
                # Split train_full into train and val using customer group stratification
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
            # Standard Temporal / Out-Of-Time Split
            test_mask = df["snapshot_month"] >= test_start_month

            if use_cv:
                # All historical months before test set are used in Walk-Forward Cross-Validation
                train_mask = df["snapshot_month"] < test_start_month
                val_mask = (df["snapshot_month"] > train_end_month) & (df["snapshot_month"] <= val_end_month)
                split_type_name = "walk_forward_cv"
            else:
                train_mask = df["snapshot_month"] <= train_end_month
                val_mask = (df["snapshot_month"] > train_end_month) & (df["snapshot_month"] <= val_end_month)
                split_type_name = "time_based_holdout"

        behavioral_only = kwargs.get("behavioral_only", False) or kwargs.get("drop_static_tiers", False)

        # Apply Feature Selection Ablations dynamically on training data
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

        # Dynamic Sample Weights
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


class StaticDataset(BaseDataset):
    """Static cross-sectional dataset (dataset02_fixed.csv) with 1 row per customer.
    
    Split is aligned with train_model_sklearn.ipynb:
    - Test set: 20% holdout (test_size=0.2), random_state=42, stratify=None.
    - Val set: split from the remaining 80% train set for Optuna HPO & threshold tuning.
    """

    def __init__(
        self,
        data_path: str = "data/processed/dataset02_fixed.csv",
        test_size: float = 0.2,
        val_size: float = 0.15,
        stratify: bool = False,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        data_path = kwargs.get("data_path", self.data_path)
        test_size = kwargs.get("test_size", self.test_size)
        val_size = kwargs.get("val_size", self.val_size)
        stratify_flag = kwargs.get("stratify", self.stratify)
        seed = kwargs.get("seed", self.seed)

        print(f"[DATASET] Loading static dataset from '{data_path}'...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        print(f"[DATASET] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Exclude index column and target
        exclude_cols = ["Unnamed: 0", "churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df["churn"].astype(int)

        # 1. First Split: Exact 20% holdout test set matching train_model_sklearn.ipynb (random_state=42, no stratify)
        strat_y = y if stratify_flag else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat_y
        )

        # 2. Validation Split from remaining train set for Optuna HPO & threshold tuning
        if val_size > 0:
            strat_val = y_train_full if stratify_flag else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=strat_val
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = X_test, y_test

        result = SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            metadata={
                "dataset_name": "static",
                "data_path": data_path,
                "split_type": "aligned_with_notebook",
                "test_size": test_size,
                "val_size": val_size,
                "stratify": stratify_flag,
                "seed": seed,
            },
        )
        result.print_summary()
        return result


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

        # Exclude all churn targets from features to prevent leakage
        all_churn_targets = [
            "churn_30d", "churn_60d", "churn_case1_30d",
            "churn_case2_30d", "churn_case2a_30d", "churn_case2b_30d"
        ]
        non_feature_cols = ["customer_id", "snapshot_month", "snapshot_date", "account_status"] + all_churn_targets
        cat_cols = ["gender", "region", "city"]

        train_groups = None

        if split_strategy in ["group_stratified", "stratified_group", "customer_stratified"]:
            # Concat all 3 files and apply customer-level StratifiedGroupKFold
            from sklearn.model_selection import StratifiedGroupKFold

            df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
            print(f"[DATASET] Combined Pool: {len(df_all):,} snapshots across {df_all['customer_id'].nunique():,} unique customers")

            # Apply ChurnFeaturePreprocessor (Combine static customer master + clean latest time-series)
            preprocessor = ChurnFeaturePreprocessor(merge_static_master=True, include_stock_features=stock_features)
            df_all = preprocessor.transform(df_all)

            feature_cols = [c for c in df_all.columns if c not in non_feature_cols]

            # Compute customer-level churn indicator for balanced stratification
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

            # Feature selection filters dynamically on training data
            feature_cols = apply_feature_filters(
                feature_cols=feature_cols,
                drop_low_mi=drop_low_mi,
                drop_collinear=drop_collinear,
                behavioral_only=behavioral_only,
                df_train=df_all.loc[train_mask],
                target_col=target_col,
            )

            # Handle Categorical Columns
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
            # Default Pre-split behavior
            preprocessor = ChurnFeaturePreprocessor(merge_static_master=True, include_stock_features=stock_features)
            df_train = preprocessor.transform(df_train)
            df_val = preprocessor.transform(df_val)
            df_test = preprocessor.transform(df_test)

            feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

            df_train_eval = pd.concat([df_train, df_val], ignore_index=True) if use_cv else df_train

            # Feature selection filters dynamically on training data
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

        # Dynamic Sample Weights
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


class PointInTimeTimeSeriesDataset(BaseDataset):
    """Chiến Lược 4: Point-in-Time Customer-Level Dataset with Dynamic Time-Series Rolling Features.

    Each unique customer is represented by exactly 1 row (their most recent historical snapshot point),
    retaining all 42 high-dimensional rolling, trend, lag, and slope features.

    Features:
    - Eliminates multi-snapshot customer duplication.
    - Preserves all time-series dynamic features (activity_slope_3m, usage_trend_30d, session_duration_trend...).
    - Uses stratified or time-based splitting with balanced cross-validation.
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

        # Apply ChurnFeaturePreprocessor (Dynamic behavioral & financial features only, merge_static_master=False)
        preprocessor = ChurnFeaturePreprocessor(merge_static_master=False)
        df = preprocessor.transform(df)

        exclude_cols = ["customer_id", "snapshot_month", "label_churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Apply Feature Selection Ablations
        feature_cols = apply_feature_filters(
            feature_cols=feature_cols,
            drop_low_mi=drop_low_mi,
            drop_collinear=drop_collinear,
        )

        if test_start_month:
            # Temporal Point-in-Time:
            # Train pool: Latest snapshot of each customer in historical period (< test_start_month)
            # Test pool: Latest snapshot of each customer in test period (>= test_start_month)
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
            # Cross-sectional Point-in-Time (All-time latest snapshot per customer)
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

        # Dynamic Sample Weights
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


class DatasetRegistry:
    """Registry pattern for discovering and instantiating datasets."""
    _registry: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
        """Decorator to register a new dataset class."""
        def decorator(subclass: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseDataset:
        """Instantiate a registered dataset by name."""
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Dataset '{name}' is not registered. Available datasets: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of all registered dataset names."""
        return sorted(list(cls._registry.keys()))


class GroupStratifiedTimeSeriesDataset(TimeSeriesDataset):
    """TimeSeries dataset using Customer-Level Stratified Group Splitting (Zero Customer Leakage)."""
    def __init__(self, **kwargs: Any):
        kwargs["split_strategy"] = "group_stratified"
        super().__init__(**kwargs)


class GroupStratifiedLatestDataset(PreSplitLatestDataset):
    """Latest dataset (Static Master + Dynamic TimeSeries) using Customer-Level Stratified Group Splitting (Zero Customer Leakage)."""
    def __init__(self, **kwargs: Any):
        kwargs["split_strategy"] = "group_stratified"
        super().__init__(**kwargs)


# Register built-in datasets
DatasetRegistry.register("timeseries")(TimeSeriesDataset)
DatasetRegistry.register("timeseries_group")(GroupStratifiedTimeSeriesDataset)
DatasetRegistry.register("group_stratified")(GroupStratifiedTimeSeriesDataset)
DatasetRegistry.register("static")(StaticDataset)
DatasetRegistry.register("latest")(PreSplitLatestDataset)
DatasetRegistry.register("latest_group")(GroupStratifiedLatestDataset)
DatasetRegistry.register("presplit")(PreSplitLatestDataset)
DatasetRegistry.register("churn_team")(PreSplitLatestDataset)
DatasetRegistry.register("pit")(PointInTimeTimeSeriesDataset)
DatasetRegistry.register("point_in_time")(PointInTimeTimeSeriesDataset)
DatasetRegistry.register("timeseries_pit")(PointInTimeTimeSeriesDataset)

