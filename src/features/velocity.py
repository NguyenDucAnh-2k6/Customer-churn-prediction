"""
Velocity, Acceleration, and Engagement Share Feature Engineering Module.

Extracts dynamic speed of engagement decline, ratio relative to historical baseline,
and activity acceleration.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_velocity_drop_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute drop ratios comparing recent 30-day activity with 3-month rolling averages."""
    df = df.copy()

    # 1. Usage Event Drop Ratio (30d vs 3m baseline)
    if "num_usage_events_30d" in df.columns and "num_usage_events_roll3m_sum" in df.columns:
        df["usage_drop_ratio_3m"] = df["num_usage_events_30d"] / (df["num_usage_events_roll3m_sum"] / 3.0 + 1.0)

    # 2. Session Duration Drop Ratio & Delta
    if "avg_session_duration_30d" in df.columns and "avg_session_duration_roll3m_mean" in df.columns:
        df["session_duration_drop_ratio_3m"] = df["avg_session_duration_30d"] / (df["avg_session_duration_roll3m_mean"] + 1.0)
        df["usage_duration_change"] = df["avg_session_duration_30d"] - df["avg_session_duration_roll3m_mean"]

    # 3. Active Days Share of 90d window
    if "total_active_days_30d" in df.columns and "total_active_days_90d" in df.columns:
        df["active_days_share_90d"] = df["total_active_days_30d"] / (df["total_active_days_90d"] + 1.0)

    # 4. Order Share of 90d window
    if "orders_last_30d" in df.columns and "orders_roll3m_sum" in df.columns:
        df["orders_share_90d"] = df["orders_last_30d"] / (df["orders_roll3m_sum"] + 1.0)

    # 5. Activity Acceleration (Interaction between 3m slope and 30d trend)
    if "activity_slope_3m" in df.columns and "usage_trend_30d" in df.columns:
        df["activity_acceleration"] = df["activity_slope_3m"] * df["usage_trend_30d"]

    # 6. Lifetime Share (if total lifetime usage sessions is available)
    if "num_usage_events_30d" in df.columns and "total_usage_sessions" in df.columns:
        df["usage_30d_share_lifetime"] = df["num_usage_events_30d"] / (df["total_usage_sessions"].fillna(0) + 1.0)

    return df


class VelocityFeatureGenerator:
    """Transformer for engagement velocity, shares, and acceleration features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity and acceleration features."""
        return compute_velocity_drop_ratios(df)
