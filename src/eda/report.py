"""
Automated EDA Suite Generator: Runs full visualization pipeline and exports summary report.
Includes deep dive on Stock Quantitative Indicators and Multi-Horizon Behavioral Dynamics.
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
    plot_stock_technical_indicators,
    plot_behavioral_dynamics,
)
from src.eda.timeseries_eda import (
    plot_activity_trends_over_time,
    plot_churn_trend_over_time,
)
from src.features.statistical_filtering import generate_univariate_screening_report



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

    # 8. Financial & Stock Technical Indicators Analysis (if present)
    stock_fig_path = os.path.join(output_dir, "08_stock_technical_indicators.png")
    stock_res = plot_stock_technical_indicators(df, target_col=target_col, output_path=stock_fig_path)
    if stock_res is not None:
        generated_files.append(stock_fig_path)

    # 9. Multi-Window Behavioral Dynamics Analysis (if present)
    dyn_fig_path = os.path.join(output_dir, "09_teammate_behavioral_dynamics.png")
    dyn_res = plot_behavioral_dynamics(df, target_col=target_col, output_path=dyn_fig_path)
    if dyn_res is not None:
        generated_files.append(dyn_fig_path)

    # 10. Univariate Statistical Feature Screening (KDE, Categorical, Boxplot)
    screening_df = generate_univariate_screening_report(
        X_train=df[[c for c in df.columns if c not in ["customer_id", "Unnamed: 0", target_col]]],
        y_train=df[target_col],
    )
    screening_csv_path = os.path.join(output_dir, "10_univariate_feature_screening.csv")
    screening_df.to_csv(screening_csv_path, index=False)
    generated_files.append(screening_csv_path)

    # 11. Write Markdown Summary Report
    report_md_path = os.path.join(output_dir, "eda_summary.md")
    _write_markdown_summary(
        df=df,
        dataset_name=dataset_name,
        target_col=target_col,
        time_col=time_col,
        df_multi=df_multi,
        top_corrs=corrs.head(15),
        screening_df=screening_df,
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
    screening_df: pd.DataFrame,
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

    # Check for presence of stock indicators and compute their correlations
    stock_cols = [
        "RSI_usage", "stoch_k_usage", "engagement_macd",
        "usage_drawdown_ratio", "active_days_volatility_3m", "peer_usage_zscore", "cohort_relative_strength_30d"
    ]
    present_stock = [c for c in stock_cols if c in df.columns]
    if present_stock:
        lines.extend([
            "",
            "## 📈 3. Quantitative Stock & Market Technical Indicators Analysis",
            "",
            "| Stock Technical Indicator | Correlation with Churn | Business Signal Interpretation |",
            "| :--- | :---: | :--- |",
        ])
        for c in present_stock:
            c_val = float(df[c].corr(df[target_col]))
            c_dir = "Tăng nguy cơ Churn" if c_val > 0 else "Giảm nguy cơ Churn (Tích cực)"
            lines.append(f"| `{c}` | `{c_val:+.4f}` | {c_dir} |")

    lines.extend([
        "",
        "## ⚠️ 4. Multicollinearity Detection (|r| >= 0.85)",
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

    # 5. Statistical Screening Summary
    lines.extend([
        "",
        "## 🧪 5. Univariate Statistical Feature Screening (Separability)",
        "",
        "Đánh giá độ phân tách đơn biến giữa 2 nhóm Churned và Retained qua 3 tiêu chuẩn thống kê:",
        "- **KDE D_KS**: 2-Sample Kolmogorov-Smirnov ($D_{KS} \\ge 0.05$ là đạt chuẩn).",
        "- **Boxplot Cohen's d**: Standardized Mean Difference ($d \\ge 0.08$ hoặc $\\text{IQR Overlap} \\le 90\\%$).",
        "- **Categorical Cramér's V / IV**: Độ liên thuộc phân loại ($V \\ge 0.03$ hoặc $\\text{IV} \\ge 0.02$).",
        "",
        "| Feature | KS Stat | Cohen's d | IQR Overlap | Cramér's V / IV | Overall Signal |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    # Show top strong and weak features
    for _, row in screening_df.head(12).iterrows():
        ks_val = f"{row['ks_stat']:.4f}" if pd.notna(row['ks_stat']) else "-"
        cd_val = f"{row['cohens_d']:.4f}" if pd.notna(row['cohens_d']) else "-"
        iqr_val = f"{row['iqr_overlap_ratio']:.2%}" if pd.notna(row['iqr_overlap_ratio']) else "-"
        v_val = f"V={row['cramers_v']:.3f} / IV={row['information_value']:.3f}" if pd.notna(row['cramers_v']) else "-"
        sig_badge = "🟢 Strong" if row["overall_signal"] == "STRONG" else "🟡 Weak"
        lines.append(f"| `{row['feature']}` | `{ks_val}` | `{cd_val}` | `{iqr_val}` | `{v_val}` | {sig_badge} |")

    lines.extend([
        "",
        "## 🖼️ 6. Visual Charts Generated",
        "",
        "- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.",
        "- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.",
        "- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.",
        "- `04_feature_distributions.png`: Boxplot/KDE so sánh phân phối giữa 2 nhóm Churned và Retained.",
        "- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.",
    ])

    if time_col and time_col in df.columns:
        lines.extend([
            "- `06_timeseries_churn_trend.png`: Xu hướng tỷ lệ rời bỏ theo các mốc snapshot tháng.",
            "- `07_timeseries_activity_trends.png`: Xu hướng tương tác app trung bình theo thời gian.",
        ])

    if present_stock:
        lines.append("- `08_stock_technical_indicators.png`: Phân phối các chỉ báo tài chính (RSI, Stochastic, MACD, Volatility, Drawdown) theo nhãn rời bỏ.")

    behavioral_cols = ["total_active_days_7d", "total_active_days_30d", "days_since_last_activity", "orders_roll3m_sum"]
    if any(c in df.columns for c in behavioral_cols):
        lines.append("- `09_teammate_behavioral_dynamics.png`: Phân phối các chỉ số hoạt động đa khung thời gian (7d/30d/90d) và CSAT.")

    lines.append("- `10_univariate_feature_screening.csv`: Bảng tổng hợp chẩn đoán toàn diện chỉ số KS, Cohen's d, IQR Overlap và Cramér's V cho mọi đặc trưng.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[EDA] Saved Markdown Summary Report to: '{output_path}'")

