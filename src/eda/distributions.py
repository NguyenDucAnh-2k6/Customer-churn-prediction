"""
Distribution analysis module: Class balance, feature comparisons, and categorical churn breakdown.
Includes specialized visualizers for Financial Stock Indicators and Teammate Behavioral Dynamics.
"""

import os
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate a dual-view (Bar + Donut) visualization of the target class distribution."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    counts = df[target_col].value_counts().sort_index()
    percentages = (counts / len(df)) * 100

    labels = ["Active / Retained (0)", "Churned (1)"] if len(counts) == 2 else [str(c) for c in counts.index]
    colors = ["#2b5c8f", "#d9534f"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax1.set_title(f"Sample Counts by Class ({target_col})", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Samples", fontsize=11)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + (max(counts.values) * 0.015),
                 f"{int(height):,}", ha="center", fontsize=10, fontweight="bold")

    # Donut chart
    wedges, texts, autotexts = ax2.pie(
        percentages.values,
        labels=labels,
        autopct="%1.2f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")
    ax2.set_title(f"Class Proportion ({percentages.iloc[-1]:.2f}% Churn)", fontsize=12, fontweight="bold", pad=10)

    fig_title = title or f"Target Distribution Analysis: '{target_col}' (Total: {len(df):,} samples)"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Target Distribution Plot to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_feature_distributions_by_target(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    n_cols: int = 3,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate KDE distribution subplots comparing continuous features across churn classes."""
    valid_features = [f for f in features if f in df.columns and f != target_col]
    if not valid_features:
        return None

    n_cols_plot = min(n_cols, len(valid_features))
    n_rows = (len(valid_features) + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(n_cols_plot * 5, n_rows * 3.8))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(valid_features):
        ax = axes[i]
        
        # Clip outliers to 1st-99th percentile for clean plotting
        q_low, q_high = df[col].quantile(0.01), df[col].quantile(0.99)
        col_data = df[col].clip(lower=q_low, upper=q_high)

        sns.kdeplot(
            data=df,
            x=col_data,
            hue=target_col,
            palette=["#2b5c8f", "#d9534f"],
            common_norm=False,
            fill=True,
            alpha=0.35,
            linewidth=2,
            ax=ax,
            warn_singular=False,
        )
        ax.set_title(f"{col}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    for j in range(len(valid_features), len(axes)):
        fig.delaxes(axes[j])

    fig_title = title or "Feature Distributions by Churn Status"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Feature Distributions Plot to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_categorical_churn_rates(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    n_cols: int = 2,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate subplots of churn rates broken down by categorical variables."""
    valid_cols = [c for c in cat_cols if c in df.columns and c != target_col and df[c].nunique() <= 12]
    if not valid_cols:
        return None

    n_cols_plot = min(n_cols, len(valid_cols))
    n_rows = (len(valid_cols) + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(n_cols_plot * 6, n_rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        summary = df.groupby(col)[target_col].agg(["count", "mean"]).reset_index()
        summary["churn_pct"] = summary["mean"] * 100

        bars = ax.bar(
            summary[col].astype(str),
            summary["churn_pct"],
            color="#d9534f",
            edgecolor="black",
            width=0.55,
            alpha=0.85,
        )
        ax.set_title(f"Churn Rate by '{col}'", fontsize=11, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%",
                    ha="center", fontsize=8, fontweight="bold")

    for j in range(len(valid_cols), len(axes)):
        fig.delaxes(axes[j])

    fig_title = title or "Churn Rate by Key Categories"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Categorical Churn Rates Plot to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_stock_technical_indicators(
    df: pd.DataFrame,
    target_col: str,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate specialized multi-panel visualization of Financial & Stock Quantitative Indicators."""
    stock_cols = [
        "RSI_usage", "stoch_k_usage", "engagement_macd",
        "usage_drawdown_ratio", "active_days_volatility_3m", "peer_usage_zscore"
    ]
    present_cols = [c for c in stock_cols if c in df.columns]
    if len(present_cols) < 2:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(present_cols[:6]):
        ax = axes[idx]
        q_low, q_high = df[col].quantile(0.01), df[col].quantile(0.99)
        clipped = df[col].clip(q_low, q_high)

        sns.boxplot(
            data=df,
            x=target_col,
            y=clipped,
            hue=target_col,
            legend=False,
            palette=["#2b5c8f", "#d9534f"],
            ax=ax,
            width=0.45,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
            showfliers=False
        )
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained (0)", "Churned (1)"], fontsize=10)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    for j in range(len(present_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Quantitative Stock & Market Technical Indicators vs Churn Status", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Stock Technical Indicators Plot to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_behavioral_dynamics(
    df: pd.DataFrame,
    target_col: str,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate specialized multi-panel visualization of Multi-window Active Days, CSAT & Recency."""
    behavioral_cols = [
        "total_active_days_7d", "total_active_days_30d", "total_active_days_90d",
        "days_since_last_activity", "orders_roll3m_sum", "payments_success_rate"
    ]
    present_cols = [c for c in behavioral_cols if c in df.columns]
    if len(present_cols) < 2:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(present_cols[:6]):
        ax = axes[idx]
        q_low, q_high = df[col].quantile(0.01), df[col].quantile(0.99)
        clipped = df[col].clip(q_low, q_high)

        sns.boxplot(
            data=df,
            x=target_col,
            y=clipped,
            hue=target_col,
            legend=False,
            palette=["#2b5c8f", "#d9534f"],
            ax=ax,
            width=0.45,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
            showfliers=False

        )
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained (0)", "Churned (1)"], fontsize=10)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    for j in range(len(present_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Behavioral Dynamics, Multi-Window Activity & CSAT vs Churn Status", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Behavioral Dynamics Plot to: '{output_path}'")

    plt.close(fig)
    return fig
