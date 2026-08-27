"""
Correlation analysis module: Heatmaps, target correlation ranking, and multicollinearity detection.
"""

import os
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    method: str = "pearson",
    top_k: Optional[int] = 25,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate and save a correlation matrix heatmap with upper-triangle masking.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, optional
        If specified and top_k is set, filters to top_k features most correlated with target.
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman').
    top_k : int, optional, default=25
        Maximum number of features to include for readability.
    output_path : str, optional
        Path to save the plot image (.png).
    title : str, optional
        Custom title for the figure.
    """
    # Select only numeric features
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Drop index-like columns
    drop_candidates = ["Unnamed: 0", "customer_id"]
    numeric_df = numeric_df.drop(columns=[c for c in drop_candidates if c in numeric_df.columns], errors="ignore")

    if target_col and target_col in numeric_df.columns and top_k and len(numeric_df.columns) > top_k:
        # Filter to top_k most correlated features with target
        corr_with_target = numeric_df.corr(method=method)[target_col].abs().sort_values(ascending=False)
        selected_cols = corr_with_target.head(top_k).index.tolist()
        numeric_df = numeric_df[selected_cols]

    corr = numeric_df.corr(method=method)

    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    n_features = len(numeric_df.columns)
    figsize = (max(10, n_features * 0.45), max(8, n_features * 0.40))
    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    annot = n_features <= 18  # Only show text numbers if feature count is manageable
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )

    chart_title = title or f"Correlation Matrix Heatmap ({n_features} Features)"
    ax.set_title(chart_title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Correlation Matrix Heatmap to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_target_correlations(
    df: pd.DataFrame,
    target_col: str,
    top_k: int = 20,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate a horizontal bar plot of features most correlated with the target variable."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_candidates = ["Unnamed: 0", "customer_id"]
    numeric_df = numeric_df.drop(columns=[c for c in drop_candidates if c in numeric_df.columns], errors="ignore")

    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    corrs = numeric_df.corr()[target_col].drop(labels=[target_col]).dropna()
    corrs_sorted = corrs.abs().sort_values(ascending=False).head(top_k)
    top_corrs = corrs[corrs_sorted.index].sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_corrs) * 0.35)))

    # Color code positive and negative correlations
    colors = ["#d9534f" if val > 0 else "#337ab7" for val in top_corrs.values]
    bars = ax.barh(top_corrs.index, top_corrs.values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(f"Correlation with '{target_col}'", fontsize=11, fontweight="bold")
    ax.set_title(title or f"Top {len(top_corrs)} Features Correlated with Target ({target_col})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Add data value labels
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{width:+.2f}",
                va="center", fontsize=8, color="#222222", fontweight="bold")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Target Correlation Barplot to: '{output_path}'")

    plt.close(fig)
    return fig


def detect_multicollinearity(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Detect and return highly correlated feature pairs (|correlation| >= threshold)."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_candidates = ["Unnamed: 0", "customer_id", "churn", "label_churn"]
    numeric_df = numeric_df.drop(columns=[c for c in drop_candidates if c in numeric_df.columns], errors="ignore")

    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    records = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] >= threshold]
        for row, val in high_corr.items():
            records.append({
                "feature_1": row,
                "feature_2": col,
                "abs_correlation": float(val),
            })

    df_multi = pd.DataFrame(records)
    if not df_multi.empty:
        df_multi = df_multi.sort_values(by="abs_correlation", ascending=False).reset_index(drop=True)
    return df_multi
