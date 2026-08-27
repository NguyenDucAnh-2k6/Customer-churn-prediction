"""
Financial & Stock Market Technical Indicators for Customer Behavior Time Series.

Adapts quantitative trading concepts (Volatility, MACD Momentum, Beta/Cohort Z-score, Maximum Drawdown)
to customer engagement time series.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_peer_beta_zscore(
    df: pd.DataFrame,
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "num_usage_events_30d",
) -> pd.Series:
    """Compute Relative Cohort Strength (Beta Z-Score) for each snapshot.
    
    Measures how many standard deviations a customer's monthly activity is above/below
    the entire active customer base in that snapshot month. Removes platform-wide seasonality.
    
    Formula:
        Z = (Usage_i,t - Mean_t) / (Std_t + 1e-6)
    """
    if snapshot_month_col not in df.columns or usage_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    month_mean = df.groupby(snapshot_month_col)[usage_col].transform("mean")
    month_std = df.groupby(snapshot_month_col)[usage_col].transform("std").fillna(1.0).replace(0, 1.0)
    return ((df[usage_col] - month_mean) / month_std).fillna(0.0)


def compute_engagement_macd(
    df: pd.DataFrame,
    short_col: str = "num_usage_events_30d",
    long_roll_col: str = "num_usage_events_roll3m_sum",
) -> pd.Series:
    """Compute Moving Average Convergence Divergence (MACD Momentum).
    
    Measures absolute divergence between short-term engagement (30-day) and medium-term average (3-month).
    
    Formula:
        MACD = Usage_30d - (Usage_roll3m / 3.0)
    """
    if short_col not in df.columns or long_roll_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    return (df[short_col] - (df[long_roll_col] / 3.0)).fillna(0.0)


def compute_usage_drawdown(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "num_usage_events_30d",
) -> pd.Series:
    """Compute Maximum Drawdown (MDD) from personal historical peak activity.
    
    Measures how far a user's current engagement has collapsed relative to their
    own all-time peak usage level.
    
    Formula:
        Drawdown = (CumMax_Usage - Current_Usage) / (CumMax_Usage + 1.0)
    """
    if customer_id_col not in df.columns or usage_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    orig_index = df.index
    if snapshot_month_col in df.columns:
        df_sorted = df.sort_values([customer_id_col, snapshot_month_col])
    else:
        df_sorted = df.sort_values(customer_id_col)
    
    cummax = df_sorted.groupby(customer_id_col)[usage_col].cummax()
    drawdown = (cummax - df_sorted[usage_col]) / (cummax + 1.0)
    
    return drawdown.reindex(orig_index).fillna(0.0)


def compute_behavioral_volatility(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    col: str = "total_active_days_30d",
    window: int = 3,
) -> pd.Series:
    """Compute Historical Volatility (Rolling Standard Deviation of Engagement).
    
    Measures instability and erraticness in a customer's activity before churn.
    """
    if customer_id_col not in df.columns or col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    orig_index = df.index
    if snapshot_month_col in df.columns:
        df_sorted = df.sort_values([customer_id_col, snapshot_month_col])
    else:
        df_sorted = df.sort_values(customer_id_col)
    
    vol = (
        df_sorted.groupby(customer_id_col)[col]
        .rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    return vol.reindex(orig_index).fillna(0.0)


class FinancialFeatureGenerator:
    """Transformer that generates stock/financial-inspired technical indicators on customer time series."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        snapshot_month_col: str = "snapshot_month",
    ):
        self.customer_id_col = customer_id_col
        self.snapshot_month_col = snapshot_month_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial/stock-inspired technical features to the DataFrame."""
        df = df.copy()

        # 1. Peer Relative Strength / Beta Z-Score
        if "num_usage_events_30d" in df.columns:
            df["peer_usage_zscore"] = compute_peer_beta_zscore(
                df, snapshot_month_col=self.snapshot_month_col, usage_col="num_usage_events_30d"
            )

        # 2. MACD Momentum (Usage divergence)
        if "num_usage_events_30d" in df.columns and "num_usage_events_roll3m_sum" in df.columns:
            df["engagement_macd"] = compute_engagement_macd(
                df, short_col="num_usage_events_30d", long_roll_col="num_usage_events_roll3m_sum"
            )

        # 3. Maximum Drawdown (MDD from personal peak)
        if "num_usage_events_30d" in df.columns and self.customer_id_col in df.columns:
            df["usage_drawdown_from_peak"] = compute_usage_drawdown(
                df,
                customer_id_col=self.customer_id_col,
                snapshot_month_col=self.snapshot_month_col,
                usage_col="num_usage_events_30d",
            )

        # 4. Behavioral Volatility (Rolling 3m std of active days)
        if "total_active_days_30d" in df.columns and self.customer_id_col in df.columns:
            df["active_days_volatility_3m"] = compute_behavioral_volatility(
                df,
                customer_id_col=self.customer_id_col,
                snapshot_month_col=self.snapshot_month_col,
                col="total_active_days_30d",
                window=3,
            )

        return df
