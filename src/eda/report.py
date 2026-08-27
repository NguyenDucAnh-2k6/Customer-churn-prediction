"""
Automated EDA Suite Generator: Runs full visualization pipeline and exports summary report.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.eda.correlations import (
    detect_multicollinearity,
    plot_correlation_matrix,
    plot_target_correlations,
)
from src.eda.distributions import (
    plot_categorical_churn_rates,
    plot_feature_distributions_by_target,
    plot_target_distribution,
)
from src.eda.timeseries_eda import (
    plot_activity_trends_over_time,
    plot_churn_trend_over_time,
)


def generate_eda_suite(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    output_dir: str,
    time_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete EDA suite, produce all visual charts and write eda_summary.md report."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"📊 Running Complete EDA Suite for '{dataset_name.upper()}'")
    print(f"   Output Directory: '{output_dir}'")
    print(f"{'='*60}")

    generated_files = []

    # 1. Target Distribution
    target_dist_path = os.path.join(output_dir, "01_target_distribution.png")
    plot_target_distribution(df, target_col=target_col, output_path=target_dist_path)
    generated_files.append(target_dist_path)

    # 2. Correlation Matrix Heatmaps (Full & Top Correlated)
    corr_top_path = os.path.join(output_dir, "02_correlation_matrix_top.png")
    plot_correlation_matrix(df, target_col=target_col, top_k=20, output_path=corr_top_path,
                            title=f"Top 20 Correlated Features Correlation Heatmap ({dataset_name})")
    generated_files.append(corr_top_path)

    # 3. Target Correlation Ranking Barplot
    target_corr_path = os.path.join(output_dir, "03_target_correlations.png")
    plot_target_correlations(df, target_col=target_col, top_k=20, output_path=target_corr_path,
                             title=f"Top 20 Features Correlated with Target ({dataset_name})")
    generated_files.append(target_corr_path)

    # 4. Multicollinearity Detection
    df_multi = detect_multicollinearity(df, threshold=0.85)
    multi_csv_path = os.path.join(output_dir, "multicollinearity_pairs.csv")
    df_multi.to_csv(multi_csv_path, index=False)
    generated_files.append(multi_csv_path)

    # 5. Feature Distributions by Churn (Top continuous features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["Unnamed: 0", "customer_id", target_col]
    candidate_features = [c for c in numeric_cols if c not in exclude_cols]

    # Select top 9 features by correlation with target
    corrs = df[candidate_features].apply(lambda c: c.corr(df[target_col])).abs().sort_values(ascending=False)
    top_features = corrs.head(9).index.tolist()

    if top_features:
        feat_dist_path = os.path.join(output_dir, "04_feature_distributions.png")
        plot_feature_distributions_by_target(df, features=top_features, target_col=target_col,
                                            output_path=feat_dist_path,
                                            title=f"Distributions of Top {len(top_features)} Features by Churn Status ({dataset_name})")
        generated_files.append(feat_dist_path)

    # 6. Categorical Churn Rates
    cat_candidates = [c for c in candidate_features if df[c].nunique() <= 8]
    if cat_candidates:
        cat_churn_path = os.path.join(output_dir, "05_categorical_churn_rates.png")
        plot_categorical_churn_rates(df, cat_cols=cat_candidates[:6], target_col=target_col,
                                     output_path=cat_churn_path,
                                     title=f"Churn Rates by Category Breakdown ({dataset_name})")
        generated_files.append(cat_churn_path)

    # 7. Time-series EDA (if applicable)
    if time_col and time_col in df.columns:
        ts_trend_path = os.path.join(output_dir, "06_timeseries_churn_trend.png")
        plot_churn_trend_over_time(df, time_col=time_col, target_col=target_col, output_path=ts_trend_path)
        generated_files.append(ts_trend_path)

        ts_activity_path = os.path.join(output_dir, "07_timeseries_activity_trends.png")
        plot_activity_trends_over_time(df, time_col=time_col, output_path=ts_activity_path)
        generated_files.append(ts_activity_path)

    # 8. Write Markdown Summary Report
    report_md_path = os.path.join(output_dir, "eda_summary.md")
    _write_markdown_summary(
        df=df,
        dataset_name=dataset_name,
        target_col=target_col,
        time_col=time_col,
        df_multi=df_multi,
        top_corrs=corrs.head(10),
        output_path=report_md_path,
    )
    generated_files.append(report_md_path)

    print(f"\n[EDA] Successfully generated {len(generated_files)} artifacts in '{output_dir}'.")
    return {
        "dataset_name": dataset_name,
        "output_dir": output_dir,
        "generated_files": generated_files,
        "n_samples": len(df),
        "n_features": len(candidate_features),
        "churn_rate": float(df[target_col].mean()),
    }


def _write_markdown_summary(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    time_col: Optional[str],
    df_multi: pd.DataFrame,
    top_corrs: pd.Series,
    output_path: str,
) -> None:
    """Generate Markdown summary document with key statistics."""
    pos_count = int((df[target_col] == 1).sum())
    neg_count = int((df[target_col] == 0).sum())
    churn_rate = float(df[target_col].mean())

    lines = [
        f"# EDA Summary Report - {dataset_name.upper()} Dataset",
        "",
        "## 📌 1. Dataset Dimensions & Class Distribution",
        "",
        f"- **Total Samples:** `{len(df):,}` rows",
        f"- **Total Columns:** `{len(df.columns)}` columns",
        f"- **Target Column:** `{target_col}`",
        f"- **Positive (Churn = 1):** `{pos_count:,}` samples (`{churn_rate:.2%}`)",
        f"- **Negative (Active = 0):** `{neg_count:,}` samples (`{(1-churn_rate):.2%}`)",
        f"- **Class Imbalance Ratio:** `1 : {neg_count/pos_count:.2f}`",
        "",
        "## 🔗 2. Top Correlated Features with Target",
        "",
        "| Feature | Absolute Correlation | Direction |",
        "| :--- | :---: | :---: |",
    ]

    for feat, corr_abs in top_corrs.items():
        raw_corr = float(df[feat].corr(df[target_col]))
        direction = "Positive (+)" if raw_corr > 0 else "Negative (-)"
        lines.append(f"| `{feat}` | `{corr_abs:.4f}` | **{direction}** (`{raw_corr:+.4f}`) |")

    lines.extend([
        "",
        "## ⚠️ 3. Multicollinearity Detection (|r| >= 0.85)",
        "",
    ])

    if df_multi.empty:
        lines.append("✅ Không phát hiện cặp đặc trưng nào có tương quan quá cao ($|r| \\ge 0.85$).")
    else:
        lines.extend([
            f"Phát hiện **{len(df_multi)}** cặp đặc trưng có tương quan mạnh:",
            "",
            "| Feature 1 | Feature 2 | Correlation |",
            "| :--- | :--- | :---: |",
        ])
        for _, row in df_multi.head(10).iterrows():
            lines.append(f"| `{row['feature_1']}` | `{row['feature_2']}` | `{row['abs_correlation']:.4f}` |")

    lines.extend([
        "",
        "## 🖼️ 4. Visual Charts Generated",
        "",
        "- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.",
        "- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.",
        "- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.",
        "- `04_feature_distributions.png`: Boxplot so sánh phân phối giữa 2 nhóm Churned và Retained.",
        "- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.",
    ])

    if time_col and time_col in df.columns:
        lines.extend([
            "- `06_timeseries_churn_trend.png`: Biểu đồ xu hướng Churn Rate và số lượng KH qua các tháng.",
            "- `07_timeseries_activity_trends.png`: Xu hướng dịch chuyển các chỉ số tương tác theo thời gian.",
        ])

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[EDA] Written EDA Summary Report to: '{output_path}'")
