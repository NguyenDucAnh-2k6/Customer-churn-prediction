"""
Financial & Stock Market Technical Indicators for Customer Behavior Time Series.

Adapts quantitative trading concepts (Volatility, MACD Momentum, Beta/Cohort Z-score, 
Maximum Drawdown, RSI, Stochastic Oscillator, Downside Volatility, and Relative Strength)
to customer engagement and churn prediction.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

STOCK_FEATURE_COLS = [
    "peer_usage_zscore",
    "engagement_macd",
    "engagement_macd_signal",
    "engagement_macd_hist",
    "usage_drawdown_from_peak",
    "usage_drawdown_ratio",
    "active_days_volatility_3m",
    "usage_downside_volatility_3m",
    "RSI_usage",
    "RSI_dist_neutral",
    "RSI_oversold_smooth",
    "stoch_k_usage",
    "stoch_oversold_smooth",
    "cohort_relative_strength_30d",
]


def compute_peer_beta_zscore(
    df: pd.DataFrame,
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
) -> pd.Series:
    """Compute Relative Cohort Strength (Beta Z-Score) for each snapshot."""
    col = usage_col if usage_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    if not col or snapshot_month_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    month_mean = df.groupby(snapshot_month_col)[col].transform("mean")
    month_std = df.groupby(snapshot_month_col)[col].transform("std").fillna(1.0).replace(0, 1.0)
    return ((df[col] - month_mean) / month_std).fillna(0.0)


def compute_engagement_macd(
    df: pd.DataFrame,
    short_col: str = "total_usage_30d",
    long_roll_col: str = "total_usage_60d",
) -> pd.DataFrame:
    """Compute Moving Average Convergence Divergence (MACD Momentum) and Signal/Histogram."""
    s_col = short_col if short_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    l_col = long_roll_col if long_roll_col in df.columns else ("num_usage_events_roll3m_sum" if "num_usage_events_roll3m_sum" in df.columns else None)
    
    if not s_col or not l_col:
        macd = pd.Series(0.0, index=df.index)
    else:
        divisor = 3.0 if "3m" in l_col or "90" in l_col else 2.0
        macd = (df[s_col] - (df[l_col] / divisor)).fillna(0.0)
    
    signal = macd.rolling(3, min_periods=1).mean().fillna(0.0)
    hist = macd - signal
    
    return pd.DataFrame({
        "engagement_macd": macd,
        "engagement_macd_signal": signal,
        "engagement_macd_hist": hist,
    }, index=df.index)


def compute_usage_drawdown(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
) -> pd.DataFrame:
    """Compute Maximum Drawdown (MDD) from personal historical peak activity."""
    col = usage_col if usage_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    if not col or customer_id_col not in df.columns:
        return pd.DataFrame({
            "usage_drawdown_from_peak": pd.Series(0.0, index=df.index),
            "usage_drawdown_ratio": pd.Series(0.0, index=df.index),
        }, index=df.index)
    
    df_sorted = df.sort_values([customer_id_col, snapshot_month_col] if snapshot_month_col in df.columns else customer_id_col)
    cummax = df_sorted.groupby(customer_id_col)[col].cummax()
    drawdown_abs = (cummax - df_sorted[col])
    drawdown_ratio = drawdown_abs / (cummax + 1.0)
    
    return pd.DataFrame({
        "usage_drawdown_from_peak": drawdown_abs.loc[df.index].fillna(0.0),
        "usage_drawdown_ratio": drawdown_ratio.loc[df.index].fillna(0.0),
    }, index=df.index)


def compute_behavioral_volatility(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    col: str = "total_active_days_30d",
    window: int = 3,
) -> pd.DataFrame:
    """Compute Historical Volatility and Downside Volatility using vectorized transform."""
    c = col if col in df.columns else ("total_usage_30d" if "total_usage_30d" in df.columns else None)
    if not c or customer_id_col not in df.columns:
        return pd.DataFrame({
            "active_days_volatility_3m": pd.Series(0.0, index=df.index),
            "usage_downside_volatility_3m": pd.Series(0.0, index=df.index),
        }, index=df.index)
    
    df_sorted = df.sort_values([customer_id_col, snapshot_month_col] if snapshot_month_col in df.columns else customer_id_col)
    g = df_sorted.groupby(customer_id_col)[c]
    vol = g.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0.0)
    
    diff = g.diff().fillna(0.0).clip(upper=0.0)
    downside_vol = df_sorted.groupby(customer_id_col)[c].transform(lambda s: s.diff().fillna(0.0).clip(upper=0.0).rolling(window, min_periods=1).std()).fillna(0.0)
    
    return pd.DataFrame({
        "active_days_volatility_3m": vol.loc[df.index].fillna(0.0),
        "usage_downside_volatility_3m": downside_vol.loc[df.index].fillna(0.0),
    }, index=df.index)


def compute_engagement_rsi(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
    rsi_window: int = 3,
) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) on Customer Engagement (from stock/MOM.py)."""
    col = usage_col if usage_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    if not col or customer_id_col not in df.columns:
        return pd.DataFrame({
            "RSI_usage": pd.Series(50.0, index=df.index),
            "RSI_dist_neutral": pd.Series(0.0, index=df.index),
            "RSI_oversold_smooth": pd.Series(0.0, index=df.index),
        }, index=df.index)
    
    df_sorted = df.sort_values([customer_id_col, snapshot_month_col] if snapshot_month_col in df.columns else customer_id_col)
    g = df_sorted.groupby(customer_id_col)[col]
    delta = g.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    # Vectorized rolling mean per customer group
    avg_gain = df_sorted.groupby(customer_id_col)[col].transform(lambda s: s.diff().fillna(0.0).clip(lower=0.0).rolling(rsi_window, min_periods=1).mean())
    avg_loss = df_sorted.groupby(customer_id_col)[col].transform(lambda s: (-s.diff().fillna(0.0).clip(upper=0.0)).rolling(rsi_window, min_periods=1).mean())
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.loc[df.index].fillna(50.0)
    
    rsi_dist = (rsi - 50.0) / 50.0
    rsi_oversold = 1.0 / (1.0 + np.exp(0.2 * (rsi - 30.0)))
    
    return pd.DataFrame({
        "RSI_usage": rsi,
        "RSI_dist_neutral": rsi_dist,
        "RSI_oversold_smooth": rsi_oversold,
    }, index=df.index)


def compute_stochastic_oscillator(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
    window: int = 3,
) -> pd.DataFrame:
    """Compute Stochastic Oscillator on User Activity (from stock/MOM.py)."""
    col = usage_col if usage_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    if not col or customer_id_col not in df.columns:
        return pd.DataFrame({
            "stoch_k_usage": pd.Series(50.0, index=df.index),
            "stoch_oversold_smooth": pd.Series(0.0, index=df.index),
        }, index=df.index)
    
    df_sorted = df.sort_values([customer_id_col, snapshot_month_col] if snapshot_month_col in df.columns else customer_id_col)
    g = df_sorted.groupby(customer_id_col)[col]
    min_val = g.transform(lambda s: s.rolling(window, min_periods=1).min())
    max_val = g.transform(lambda s: s.rolling(window, min_periods=1).max())
    
    stoch_k = 100.0 * (df_sorted[col] - min_val) / (max_val - min_val + 1e-5)
    stoch_k = stoch_k.loc[df.index].fillna(50.0)
    stoch_oversold = 1.0 / (1.0 + np.exp(0.15 * (stoch_k - 20.0)))
    
    return pd.DataFrame({
        "stoch_k_usage": stoch_k,
        "stoch_oversold_smooth": stoch_oversold,
    }, index=df.index)


def compute_cohort_relative_strength(
    df: pd.DataFrame,
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
) -> pd.Series:
    """Compute Cohort-Relative Strength (from market/Beta.py)."""
    col = usage_col if usage_col in df.columns else ("num_usage_events_30d" if "num_usage_events_30d" in df.columns else None)
    if not col or snapshot_month_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    month_avg = df.groupby(snapshot_month_col)[col].transform("mean")
    diff_from_cohort = df[col] - month_avg
    return diff_from_cohort.fillna(0.0)


def add_all_financial_indicators(
    df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    snapshot_month_col: str = "snapshot_month",
    usage_col: str = "total_usage_30d",
    active_days_col: str = "total_active_days_30d",
) -> pd.DataFrame:
    """Convenience pipeline to calculate and append all quantitative stock/market features."""
    df = df.copy()
    
    # 1. Peer Beta Z-score
    df["peer_usage_zscore"] = compute_peer_beta_zscore(
        df, snapshot_month_col=snapshot_month_col, usage_col=usage_col
    )
    
    # 2. MACD Momentum
    macd_df = compute_engagement_macd(df, short_col=usage_col)
    for c in macd_df.columns:
        df[c] = macd_df[c]
    
    # 3. Maximum Drawdown
    dd_df = compute_usage_drawdown(
        df, customer_id_col=customer_id_col, snapshot_month_col=snapshot_month_col, usage_col=usage_col
    )
    for c in dd_df.columns:
        df[c] = dd_df[c]
    
    # 4. Behavioral Volatility & Downside Volatility
    vol_df = compute_behavioral_volatility(
        df, customer_id_col=customer_id_col, snapshot_month_col=snapshot_month_col, col=active_days_col
    )
    for c in vol_df.columns:
        df[c] = vol_df[c]
    
    # 5. Engagement RSI
    rsi_df = compute_engagement_rsi(
        df, customer_id_col=customer_id_col, snapshot_month_col=snapshot_month_col, usage_col=usage_col
    )
    for c in rsi_df.columns:
        df[c] = rsi_df[c]
    
    # 6. Stochastic Oscillator
    stoch_df = compute_stochastic_oscillator(
        df, customer_id_col=customer_id_col, snapshot_month_col=snapshot_month_col, usage_col=usage_col
    )
    for c in stoch_df.columns:
        df[c] = stoch_df[c]
    
    # 7. Cohort Relative Strength
    df["cohort_relative_strength_30d"] = compute_cohort_relative_strength(
        df, snapshot_month_col=snapshot_month_col, usage_col=usage_col
    )
    
    return df


class FinancialFeatureGenerator:
    """Scikit-Learn compatible transformer for generating financial and stock technical indicators."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        snapshot_month_col: str = "snapshot_month",
        usage_col: str = "total_usage_30d",
        active_days_col: str = "total_active_days_30d",
    ):
        self.customer_id_col = customer_id_col
        self.snapshot_month_col = snapshot_month_col
        self.usage_col = usage_col
        self.active_days_col = active_days_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_all_financial_indicators(
            df=X,
            customer_id_col=self.customer_id_col,
            snapshot_month_col=self.snapshot_month_col,
            usage_col=self.usage_col,
            active_days_col=self.active_days_col,
        )
