# EDA Summary Report - ROUND3_TIMESERIES Dataset

## 📌 1. Dataset Dimensions & Class Distribution

- **Total Samples:** `184,479` rows
- **Total Columns:** `141` columns
- **Target Column:** `label_churn`
- **Positive (Churn = 1):** `30,641` samples (`16.61%`)
- **Negative (Active = 0):** `153,838` samples (`83.39%`)
- **Class Imbalance Ratio:** `1 : 5.02`

## 🔗 2. Top Correlated Features with Target

| Feature | Absolute Correlation | Direction |
| :--- | :---: | :---: |
| `free_and_inactive_14d` | `0.4501` | **Positive (+)** (`+0.4501`) |
| `free_and_inactive_21d` | `0.4463` | **Positive (+)** (`+0.4463`) |
| `is_free_tier` | `0.3939` | **Positive (+)** (`+0.3939`) |
| `total_payments_90d` | `0.2844` | **Negative (-)** (`-0.2844`) |
| `payments_success_rate_missing` | `0.2803` | **Positive (+)** (`+0.2803`) |
| `total_payments_60d` | `0.2730` | **Negative (-)** (`-0.2730`) |
| `days_since_last_usage_event` | `0.2724` | **Positive (+)** (`+0.2724`) |
| `total_payments_30d` | `0.2390` | **Negative (-)** (`-0.2390`) |
| `is_auto_renew` | `0.2342` | **Negative (-)** (`-0.2342`) |
| `auto_renew` | `0.2342` | **Negative (-)** (`-0.2342`) |
| `contract_churn_risk_score` | `0.1859` | **Positive (+)** (`+0.1859`) |
| `is_paid_tier` | `0.1854` | **Negative (-)** (`-0.1854`) |
| `has_any_activity_7d` | `0.1826` | **Negative (-)** (`-0.1826`) |
| `activity_gap_ratio` | `0.1806` | **Positive (+)** (`+0.1806`) |
| `total_usage_all_time` | `0.1804` | **Positive (+)** (`+0.1804`) |

## 📈 3. Quantitative Stock & Market Technical Indicators Analysis

| Stock Technical Indicator | Correlation with Churn | Business Signal Interpretation |
| :--- | :---: | :--- |
| `RSI_usage` | `-0.1042` | Giảm nguy cơ Churn (Tích cực) |
| `stoch_k_usage` | `-0.0310` | Giảm nguy cơ Churn (Tích cực) |
| `engagement_macd` | `-0.1144` | Giảm nguy cơ Churn (Tích cực) |
| `usage_drawdown_ratio` | `+0.0437` | Tăng nguy cơ Churn |
| `active_days_volatility_3m` | `-0.0456` | Giảm nguy cơ Churn (Tích cực) |
| `peer_usage_zscore` | `-0.0339` | Giảm nguy cơ Churn (Tích cực) |
| `cohort_relative_strength_30d` | `-0.0334` | Giảm nguy cơ Churn (Tích cực) |

## ⚠️ 4. Multicollinearity Detection (|r| >= 0.85)

Phát hiện **201** cặp đặc trưng có tương quan mạnh:

| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :---: |
| `RSI_usage` | `RSI_dist_neutral` | `1.0000` |
| `has_any_activity_30d` | `session_duration_trend_missing` | `1.0000` |
| `total_orders_30d` | `orders_last_30d` | `1.0000` |
| `tenure_days` | `customer_tenure` | `1.0000` |
| `is_auto_renew` | `auto_renew` | `1.0000` |
| `num_usage_events_30d` | `total_usage_30d` | `1.0000` |
| `avg_session_duration_30d` | `avg_usage_duration_30d` | `1.0000` |
| `subscription_age_days` | `days_on_current_tier` | `1.0000` |
| `customer_age` | `age` | `1.0000` |
| `opened_rate_30d` | `open_rate_30d` | `1.0000` |

## 🖼️ 5. Visual Charts Generated

- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.
- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.
- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.
- `04_feature_distributions.png`: Boxplot/KDE so sánh phân phối giữa 2 nhóm Churned và Retained.
- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.
- `06_timeseries_churn_trend.png`: Xu hướng tỷ lệ rời bỏ theo các mốc snapshot tháng.
- `07_timeseries_activity_trends.png`: Xu hướng tương tác app trung bình theo thời gian.
- `08_stock_technical_indicators.png`: Phân phối các chỉ báo tài chính (RSI, Stochastic, MACD, Volatility, Drawdown) theo nhãn rời bỏ.
- `09_teammate_behavioral_dynamics.png`: Phân phối các chỉ số hoạt động đa khung thời gian (7d/30d/90d) và CSAT.