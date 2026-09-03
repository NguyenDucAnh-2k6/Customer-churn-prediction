"""
Sample weighting and feature filtering utilities.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from src.features.selection import (
    make_mi_scores,
    filter_features_by_mi,
    filter_multicollinear_features,
)
from src.features.statistical_filtering import (
    filter_features_by_kde,
    filter_features_by_categorical,
    filter_features_by_boxplot,
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


LOW_MI_FEATURES = [
    "gender", "region", "city", "age",
    "has_unresolved_ticket", "has_marketing_click_30d",
    "avg_csat_score_missing", "num_tickets_90d", "avg_csat_score",
    "open_rate_30d", "is_declining_engagement", "reactivation_flag",
]

COLLINEAR_PAIRS_DROP = [
    "orders_last_90d",
    "days_since_last_login",
    "total_session_time_30d",
    "num_usage_events_60d",
    "total_active_days_60d",
    "total_active_days_90d",
]

STATIC_TIER_FEATURES = [
    "is_paid_tier", "subscription_tier", "gender", "region", "city", "age", "plan_tier"
]


def apply_feature_filters(
    feature_cols: List[str],
    drop_low_mi: bool = False,
    drop_collinear: bool = False,
    behavioral_only: bool = False,
    filter_kde: bool = False,
    filter_categorical: bool = False,
    filter_boxplot: bool = False,
    ks_threshold: float = 0.05,
    cramers_v_threshold: float = 0.03,
    iv_threshold: float = 0.02,
    cohens_d_threshold: float = 0.08,
    iqr_overlap_threshold: float = 0.90,
    custom_drop_features: Optional[List[str]] = None,
    df_train: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    mi_threshold: float = 0.001,
) -> List[str]:
    """Filter feature columns based on Mutual Information (MI), Multicollinearity, Statistical Divergence (KDE, Categorical, Boxplot), and Behavioral settings."""
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

    # 1. Statistical Filtering: KDE Distribution (Kolmogorov-Smirnov Test)
    if filter_kde and df_train is not None and target_col is not None and target_col in df_train.columns:
        avail_cols = [c for c in feature_cols if c in df_train.columns and c not in dropped]
        X_sub = df_train[avail_cols]
        y_sub = df_train[target_col]
        _, kde_dropped, _ = filter_features_by_kde(
            X_sub, y_sub, ks_threshold=ks_threshold, verbose=True
        )
        dropped.update(kde_dropped)

    # 2. Statistical Filtering: Categorical Churn Homogeneity (Cramér's V & Information Value)
    if filter_categorical and df_train is not None and target_col is not None and target_col in df_train.columns:
        avail_cols = [c for c in feature_cols if c in df_train.columns and c not in dropped]
        X_sub = df_train[avail_cols]
        y_sub = df_train[target_col]
        _, cat_dropped, _ = filter_features_by_categorical(
            X_sub, y_sub, cramers_v_threshold=cramers_v_threshold, iv_threshold=iv_threshold, verbose=True
        )
        dropped.update(cat_dropped)

    # 3. Statistical Filtering: Boxplot Overlap (Cohen's d & IQR Overlap)
    if filter_boxplot and df_train is not None and target_col is not None and target_col in df_train.columns:
        avail_cols = [c for c in feature_cols if c in df_train.columns and c not in dropped]
        X_sub = df_train[avail_cols]
        y_sub = df_train[target_col]
        _, box_dropped, _ = filter_features_by_boxplot(
            X_sub, y_sub, cohens_d_threshold=cohens_d_threshold, iqr_overlap_threshold=iqr_overlap_threshold, verbose=True
        )
        dropped.update(box_dropped)

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
    """Backwards-compatible exponential recency decay weights calculator."""
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
