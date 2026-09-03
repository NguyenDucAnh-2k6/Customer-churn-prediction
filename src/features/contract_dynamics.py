"""
Contract, Payment Cycle, Recency, and Dynamic Velocity Feature Engineering Module.

Extracts domain-specific subscription renewal intervals, payment delays,
and 30d vs 60d activity velocity acceleration.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_contract_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute specialized contract renewal, payment cycle, and activity velocity features."""
    df = df.copy()

    # 1. Activity Velocity Ratios (30d vs 60d recent momentum)
    if "total_usage_30d" in df.columns and "total_usage_60d" in df.columns:
        prev_usage_30d = np.maximum(0.0, df["total_usage_60d"] - df["total_usage_30d"])
        df["usage_velocity_30d_60d"] = df["total_usage_30d"] / (prev_usage_30d + 1.0)

    if "total_orders_30d" in df.columns and "total_orders_60d" in df.columns:
        prev_orders_30d = np.maximum(0.0, df["total_orders_60d"] - df["total_orders_30d"])
        df["orders_velocity_30d_60d"] = df["total_orders_30d"] / (prev_orders_30d + 1.0)

    if "total_payments_30d" in df.columns and "total_payments_60d" in df.columns:
        prev_pay_30d = np.maximum(0.0, df["total_payments_60d"] - df["total_payments_30d"])
        df["payments_velocity_30d_60d"] = df["total_payments_30d"] / (prev_pay_30d + 1.0)

    # 2. Spend & Session Velocity
    if "total_order_amounts_30d" in df.columns and "total_order_amounts_60d" in df.columns:
        prev_spend_30d = np.maximum(0.0, df["total_order_amounts_60d"] - df["total_order_amounts_30d"])
        df["order_amount_velocity_30d_60d"] = df["total_order_amounts_30d"] / (prev_spend_30d + 1.0)

    if "avg_usage_duration_30d" in df.columns and "avg_usage_duration_60d" in df.columns:
        df["usage_duration_velocity_30d_60d"] = (df["avg_usage_duration_30d"] + 1.0) / (df["avg_usage_duration_60d"] + 1.0)

    # 3. Contract & Renewal Risk Indicators
    if "is_auto_renew" in df.columns and "is_downgrade" in df.columns:
        auto_renew_val = df["is_auto_renew"].fillna(0).astype(float)
        downgrade_val = df["is_downgrade"].fillna(0).astype(float)
        expired_val = df["subscription_expired"].fillna(0).astype(float) if "subscription_expired" in df.columns else 0.0
        df["contract_churn_risk_score"] = (1.0 - auto_renew_val) * 2.0 + downgrade_val * 2.0 + expired_val * 3.0

    # 4. Renewal Urgency (Proximity to contract expiration)
    if "days_until_end_from_snapshot" in df.columns:
        # High risk if renewal is within next 15-30 days
        days_left = df["days_until_end_from_snapshot"].fillna(-999).astype(float)
        df["is_renewal_imminent_30d"] = ((days_left >= 0) & (days_left <= 30)).astype(int)
        df["is_subscription_past_due"] = (days_left < 0).astype(int)

    # 5. Interaction to Usage Alignment
    if "total_interactions_60d" in df.columns and "total_usage_60d" in df.columns:
        df["interaction_to_usage_ratio_60d"] = df["total_interactions_60d"] / (df["total_usage_60d"] + 1.0)

    return df
