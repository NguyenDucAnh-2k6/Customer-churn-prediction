"""
Data Cleaning, Encoding, and Imputation Preprocessing Module.

Handles boolean standardization, auto_renew business rules correction,
ordinal encoding of subscription tiers, missing indicator flags, and
Strict Train-Only Statistical Imputation (Zero Data Leakage).
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize boolean columns to 0/1 integers."""
    df = df.copy()
    bool_cols = [
        "is_declining_engagement", "reactivation_flag",
        "has_marketing_click_30d", "has_unresolved_ticket", "is_paid_tier",
        "is_free_tier", "has_any_activity_7d", "has_any_activity_14d", "has_any_activity_30d",
        "free_and_inactive_14d", "free_and_inactive_21d", "paid_weak_engagement",
        "recent_downgrade_and_quiet", "auto_renew_off_paid"
    ]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == object or df[col].dtype == bool:
                df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)
    return df


def clean_subscription_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce business logic on auto_renew and subscription_tier.
    
    Free tier accounts without an active subscription must have auto_renew = -1, not True.
    """
    df = df.copy()

    if "auto_renew" in df.columns:
        if df["auto_renew"].dtype == object or df["auto_renew"].dtype == bool:
            df["auto_renew"] = df["auto_renew"].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0, -1: -1}).fillna(-1).astype(int)
        else:
            df["auto_renew"] = df["auto_renew"].fillna(-1).astype(int)

        if "is_paid_tier" in df.columns:
            df.loc[df["is_paid_tier"] == 0, "auto_renew"] = -1

    if "subscription_tier" in df.columns:
        if df["subscription_tier"].dtype == object:
            tier_map = {"Free": 0, "Plus": 1, "Premium": 2}
            df["subscription_tier"] = df["subscription_tier"].map(tier_map).fillna(0).astype(int)
        else:
            df["subscription_tier"] = df["subscription_tier"].fillna(0).astype(int)

    return df


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing indicator flags and impute missing behavioral columns with neutral defaults."""
    df = df.copy()

    if "payments_success_rate" in df.columns:
        df["payments_success_rate_missing"] = df["payments_success_rate"].isna().astype(int)
        df["payments_success_rate"] = df["payments_success_rate"].fillna(1.0)

    if "session_duration_trend" in df.columns:
        df["session_duration_trend_missing"] = df["session_duration_trend"].isna().astype(int)
        df["session_duration_trend"] = df["session_duration_trend"].fillna(0.0)

    if "avg_csat_score" in df.columns:
        df["avg_csat_score_missing"] = df["avg_csat_score"].isna().astype(int)
        df["avg_csat_score"] = df["avg_csat_score"].fillna(3.5)

    if "days_since_last_order" in df.columns:
        df["days_since_last_order"] = df["days_since_last_order"].fillna(999.0)

    return df


class DataCleaningTransformer:
    """Standardizes types, cleans business rules, and imputes missing indicator values.
    
    Supports Scikit-Learn .fit() and .transform() to guarantee strict zero-leakage imputation.
    """

    def __init__(self, strategy: str = "domain_defaults"):
        self.strategy = strategy
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, Any] = {}
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame, y=None) -> "DataCleaningTransformer":
        """Learn statistical medians and modes strictly on X_train."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isna().any():
                med = float(df[col].median())
                self.numeric_medians_[col] = 0.0 if np.isnan(med) else med

        cat_cols = df.select_dtypes(include=[object, "string"]).columns
        for col in cat_cols:
            if df[col].isna().any():
                modes = df[col].mode(dropna=True)
                self.categorical_modes_[col] = modes.iloc[0] if len(modes) > 0 else "Unknown"

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full data hygiene pipeline using learned statistics."""
        df = clean_boolean_columns(df)
        df = clean_subscription_rules(df)
        df = create_missing_indicators(df)

        if self.is_fitted_:
            for col, med in self.numeric_medians_.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(med)
            for col, mode in self.categorical_modes_.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(mode)

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)
