# EDA Summary Report - TIMESERIES Dataset

## 📌 1. Dataset Dimensions & Class Distribution

- **Total Samples:** `166,084` rows
- **Total Columns:** `45` columns
- **Target Column:** `label_churn`
- **Positive (Churn = 1):** `31,340` samples (`18.87%`)
- **Negative (Active = 0):** `134,744` samples (`81.13%`)
- **Class Imbalance Ratio:** `1 : 4.30`

## 🔗 2. Top Correlated Features with Target

| Feature | Absolute Correlation | Direction |
| :--- | :---: | :---: |
| `is_paid_tier` | `0.4064` | **Negative (-)** (`-0.4064`) |
| `subscription_tier` | `0.2721` | **Negative (-)** (`-0.2721`) |
| `avg_session_duration_30d` | `0.2401` | **Negative (-)** (`-0.2401`) |
| `event_type_diversity_30d` | `0.2373` | **Negative (-)** (`-0.2373`) |
| `total_active_days_30d` | `0.2344` | **Negative (-)** (`-0.2344`) |
| `avg_session_duration_roll3m_mean` | `0.2265` | **Negative (-)** (`-0.2265`) |
| `total_active_days_60d` | `0.2262` | **Negative (-)** (`-0.2262`) |
| `num_usage_events_30d` | `0.2253` | **Negative (-)** (`-0.2253`) |
| `total_session_time_30d` | `0.2225` | **Negative (-)** (`-0.2225`) |
| `total_active_days_90d` | `0.2176` | **Negative (-)** (`-0.2176`) |

## ⚠️ 3. Multicollinearity Detection (|r| >= 0.85)

Phát hiện **57** cặp đặc trưng có tương quan mạnh:

| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :---: |
| `orders_last_90d` | `orders_roll3m_sum` | `0.9940` |
| `num_usage_events_30d` | `total_session_time_30d` | `0.9873` |
| `num_usage_events_roll3m_sum` | `total_active_days_90d` | `0.9823` |
| `num_usage_events_60d` | `total_active_days_60d` | `0.9811` |
| `num_usage_events_30d` | `total_active_days_30d` | `0.9730` |
| `num_usage_events_60d` | `num_usage_events_roll3m_sum` | `0.9709` |
| `total_active_days_60d` | `total_active_days_90d` | `0.9705` |
| `days_since_last_login` | `days_since_last_usage_event` | `0.9675` |
| `total_active_days_30d` | `total_session_time_30d` | `0.9606` |
| `num_usage_events_roll3m_sum` | `total_active_days_60d` | `0.9547` |

## 🖼️ 4. Visual Charts Generated

- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.
- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.
- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.
- `04_feature_distributions.png`: Boxplot so sánh phân phối giữa 2 nhóm Churned và Retained.
- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.
- `06_timeseries_churn_trend.png`: Biểu đồ xu hướng Churn Rate và số lượng KH qua các tháng.
- `07_timeseries_activity_trends.png`: Xu hướng dịch chuyển các chỉ số tương tác theo thời gian.
