"""
Time-series EDA module: Monthly churn rate trends and activity drift over time.
"""

import os
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_churn_trend_over_time(
    df: pd.DataFrame,
    time_col: str = "snapshot_month",
    target_col: str = "label_churn",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate dual-axis chart: Monthly active customer volume (bars) and Churn rate % (line)."""
    if time_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Required columns ('{time_col}', '{target_col}') not in dataframe.")

    monthly = df.groupby(time_col)[target_col].agg(["count", "sum", "mean"]).reset_index()
    monthly["churn_rate_pct"] = monthly["mean"] * 100

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Primary axis: Sample counts
    color_bar = "#4a90e2"
    ax1.bar(
        monthly[time_col],
        monthly["count"],
        color=color_bar,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
        label="Total Active Customers",
    )
    ax1.set_ylabel("Customer Count", fontsize=11, color="#2b5c8f", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.tick_params(axis="x", rotation=60, labelsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)

    # Secondary axis: Churn rate %
    ax2 = ax1.twinx()
    color_line = "#d9534f"
    ax2.plot(
        monthly[time_col],
        monthly["churn_rate_pct"],
        color=color_line,
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Monthly Churn Rate (%)",
    )
    ax2.set_ylabel("Churn Rate (%)", fontsize=11, color=color_line, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, max(monthly["churn_rate_pct"].max() * 1.25, 35))

    # Add data point labels on the line
    for _, row in monthly.iterrows():
        ax2.annotate(
            f"{row['churn_rate_pct']:.1f}%",
            (row[time_col], row["churn_rate_pct"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color="#900",
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    fig_title = title or f"Customer Volume & Churn Rate Trend Over Time ({len(monthly)} Snapshot Months)"
    plt.title(fig_title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Monthly Churn Trend Plot to: '{output_path}'")

    plt.close(fig)
    return fig


def plot_activity_trends_over_time(
    df: pd.DataFrame,
    time_col: str = "snapshot_month",
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot monthly mean engagement metrics over time."""
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")

    default_features = [
        "total_active_days_30d",
        "avg_spend_to_date_per_month",
        "avg_session_duration_30d",
        "num_usage_events_30d",
    ]
    features_to_plot = [f for f in (feature_cols or default_features) if f in df.columns]

    if not features_to_plot:
        return None

    monthly_stats = df.groupby(time_col)[features_to_plot].mean().reset_index()

    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(14, len(features_to_plot) * 2.8), sharex=True)
    if len(features_to_plot) == 1:
        axes = [axes]

    colors = ["#2b5c8f", "#28a745", "#f0ad4e", "#6f42c1"]

    for i, col in enumerate(features_to_plot):
        ax = axes[i]
        c = colors[i % len(colors)]
        ax.plot(
            monthly_stats[time_col],
            monthly_stats[col],
            marker="s",
            color=c,
            linewidth=2,
            markersize=4,
            label=f"Mean {col}",
        )
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    plt.xticks(rotation=60, ha="right", fontsize=9)
    fig_title = title or "Monthly Average Engagement Metrics Over Time"
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[EDA] Saved Activity Trends Plot to: '{output_path}'")

    plt.close(fig)
    return fig
