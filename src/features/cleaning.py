"""
Data Cleaning, Encoding, and Imputation Preprocessing Module.

Handles boolean standardization, auto_renew business rules correction,
ordinal encoding of subscription tiers, and missing indicator flags.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize boolean columns to 0/1 integers."""
    df = df.copy()
    bool_cols = [
        "is_declining_engagement", "reactivation_flag",
        "has_marketing_click_30d", "has_unresolved_ticket", "is_paid_tier"
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
    """Standardizes types, cleans business rules, and imputes missing indicator values."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full data hygiene pipeline."""
        df = clean_boolean_columns(df)
        df = clean_subscription_rules(df)
        df = create_missing_indicators(df)
        return df
