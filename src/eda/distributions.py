"""
Distribution analysis module: Class balance, feature comparisons, and categorical churn breakdown.
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
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Target Distribution Chart to: '{output_path}'")

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
    """Generate subplots comparing continuous feature distributions between Churned (1) and Retained (0)."""
    valid_features = [f for f in features if f in df.columns and f != target_col]
    if not valid_features:
        raise ValueError("No valid features found to plot.")

    n_features = len(valid_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = np.array(axes).reshape(-1)

    colors = ["#2b5c8f", "#d9534f"]

    for i, col in enumerate(valid_features):
        ax = axes[i]
        # Use boxplot with list palette and hue parameter
        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            hue=target_col,
            palette=colors,
            legend=False,
            ax=ax,
            fliersize=2,
            linewidth=1.2,
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{target_col} (0=No, 1=Yes)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Hide extra unused subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    fig_title = title or f"Feature Distributions by Churn Status ({n_features} Key Features)"
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
