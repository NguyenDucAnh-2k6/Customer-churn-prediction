# EDA Summary Report - ROUND3_POINT_IN_TIME Dataset

## 📌 1. Dataset Dimensions & Class Distribution

- **Total Samples:** `10,000` rows
- **Total Columns:** `142` columns
- **Target Column:** `churn`
- **Positive (Churn = 1):** `4,848` samples (`48.48%`)
- **Negative (Active = 0):** `5,152` samples (`51.52%`)
- **Class Imbalance Ratio:** `1 : 1.06`

## 🔗 2. Top Correlated Features with Target

| Feature | Absolute Correlation | Direction |
| :--- | :---: | :---: |
| `label_churn` | `1.0000` | **Positive (+)** (`+1.0000`) |
| `is_free_tier` | `0.8385` | **Positive (+)** (`+0.8385`) |
| `free_and_inactive_14d` | `0.4150` | **Positive (+)** (`+0.4150`) |
| `total_payments_90d` | `0.3904` | **Negative (-)** (`-0.3904`) |
| `is_paid_tier` | `0.3864` | **Negative (-)** (`-0.3864`) |
| `free_and_inactive_21d` | `0.3783` | **Positive (+)** (`+0.3783`) |
| `total_payments_60d` | `0.3720` | **Negative (-)** (`-0.3720`) |
| `total_payments_30d` | `0.3295` | **Negative (-)** (`-0.3295`) |
| `is_renewal_imminent_30d` | `0.2544` | **Positive (+)** (`+0.2544`) |
| `snapshot_month_ord` | `0.2142` | **Negative (-)** (`-0.2142`) |
| `paid_weak_engagement` | `0.1923` | **Negative (-)** (`-0.1923`) |
| `auto_renew` | `0.1880` | **Negative (-)** (`-0.1880`) |
| `is_auto_renew` | `0.1880` | **Negative (-)** (`-0.1880`) |
| `had_upgrade_90d` | `0.1722` | **Negative (-)** (`-0.1722`) |
| `payments_success_rate_missing` | `0.1619` | **Positive (+)** (`+0.1619`) |

## 📈 3. Quantitative Stock & Market Technical Indicators Analysis

| Stock Technical Indicator | Correlation with Churn | Business Signal Interpretation |
| :--- | :---: | :--- |
| `RSI_usage` | `-0.0280` | Giảm nguy cơ Churn (Tích cực) |
| `stoch_k_usage` | `+0.0173` | Tăng nguy cơ Churn |
| `engagement_macd` | `+0.0121` | Tăng nguy cơ Churn |
| `usage_drawdown_ratio` | `-0.0206` | Giảm nguy cơ Churn (Tích cực) |
| `active_days_volatility_3m` | `-0.0278` | Giảm nguy cơ Churn (Tích cực) |
| `peer_usage_zscore` | `-0.0782` | Giảm nguy cơ Churn (Tích cực) |
| `cohort_relative_strength_30d` | `-0.0763` | Giảm nguy cơ Churn (Tích cực) |

## ⚠️ 4. Multicollinearity Detection (|r| >= 0.85)

Phát hiện **101** cặp đặc trưng có tương quan mạnh:

| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :---: |
| `RSI_usage` | `RSI_dist_neutral` | `1.0000` |
| `has_any_activity_30d` | `session_duration_trend_missing` | `1.0000` |
| `subscription_age_days` | `days_on_current_tier` | `1.0000` |
| `tenure_days` | `customer_tenure` | `1.0000` |
| `customer_age` | `age` | `1.0000` |
| `num_usage_events_30d` | `total_usage_30d` | `1.0000` |
| `avg_session_duration_30d` | `avg_usage_duration_30d` | `1.0000` |
| `usage_velocity_30d_60d` | `usage_trend_30d` | `1.0000` |
| `session_duration_trend` | `usage_duration_change` | `1.0000` |
| `num_usage_events_60d` | `total_usage_60d` | `1.0000` |

## 🧪 5. Univariate Statistical Feature Screening (Separability)

Đánh giá độ phân tách đơn biến giữa 2 nhóm Churned và Retained qua 3 tiêu chuẩn thống kê:
- **KDE D_KS**: 2-Sample Kolmogorov-Smirnov ($D_{KS} \ge 0.05$ là đạt chuẩn).
- **Boxplot Cohen's d**: Standardized Mean Difference ($d \ge 0.08$ hoặc $\text{IQR Overlap} \le 90\%$).
- **Categorical Cramér's V / IV**: Độ liên thuộc phân loại ($V \ge 0.03$ hoặc $\text{IV} \ge 0.02$).

| Feature | KS Stat | Cohen's d | IQR Overlap | Cramér's V / IV | Overall Signal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gender` | `-` | `-` | `-` | `V=0.001 / IV=0.000` | 🟡 Weak |
| `region` | `-` | `-` | `-` | `V=0.010 / IV=0.000` | 🟡 Weak |
| `city` | `-` | `-` | `-` | `V=0.035 / IV=0.005` | 🟢 Strong |
| `snapshot_month` | `-` | `-` | `-` | `V=0.281 / IV=0.617` | 🟢 Strong |
| `snapshot_date` | `-` | `-` | `-` | `V=0.281 / IV=0.617` | 🟢 Strong |
| `snapshot_month_ord` | `0.1425` | `0.4387` | `100.00%` | `-` | 🟢 Strong |
| `is_free_tier` | `-` | `-` | `-` | `V=0.838 / IV=4.113` | 🟢 Strong |
| `label_churn` | `-` | `-` | `-` | `V=1.000 / IV=18.416` | 🟢 Strong |
| `customer_age` | `0.0114` | `0.0099` | `100.00%` | `-` | 🟡 Weak |
| `age` | `0.0114` | `0.0099` | `100.00%` | `-` | 🟡 Weak |
| `tenure_days` | `0.0434` | `0.0613` | `96.07%` | `-` | 🟡 Weak |
| `customer_tenure` | `0.0434` | `0.0613` | `96.07%` | `-` | 🟡 Weak |

## 🖼️ 6. Visual Charts Generated

- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.
- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.
- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.
- `04_feature_distributions.png`: Boxplot/KDE so sánh phân phối giữa 2 nhóm Churned và Retained.
- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.
- `08_stock_technical_indicators.png`: Phân phối các chỉ báo tài chính (RSI, Stochastic, MACD, Volatility, Drawdown) theo nhãn rời bỏ.
- `09_teammate_behavioral_dynamics.png`: Phân phối các chỉ số hoạt động đa khung thời gian (7d/30d/90d) và CSAT.
- `10_univariate_feature_screening.csv`: Bảng tổng hợp chẩn đoán toàn diện chỉ số KS, Cohen's d, IQR Overlap và Cramér's V cho mọi đặc trưng.