# EDA Summary Report - STATIC Dataset

## 📌 1. Dataset Dimensions & Class Distribution

- **Total Samples:** `7,950` rows
- **Total Columns:** `33` columns
- **Target Column:** `churn`
- **Positive (Churn = 1):** `554` samples (`6.97%`)
- **Negative (Active = 0):** `7,396` samples (`93.03%`)
- **Class Imbalance Ratio:** `1 : 13.35`

## 🔗 2. Top Correlated Features with Target

| Feature | Absolute Correlation | Direction |
| :--- | :---: | :---: |
| `usage_60d_share` | `0.3452` | **Positive (+)** (`+0.3452`) |
| `avg_usage_duration_60d` | `0.3318` | **Positive (+)** (`+0.3318`) |
| `total_usage_60d` | `0.3068` | **Positive (+)** (`+0.3068`) |
| `avg_usage_duration_all_time` | `0.2692` | **Positive (+)** (`+0.2692`) |
| `total_usage_all_time` | `0.1535` | **Positive (+)** (`+0.1535`) |
| `days_until_end_from_snapshot` | `0.1513` | **Positive (+)** (`+0.1513`) |
| `total_interactions_all_time` | `0.1247` | **Negative (-)** (`-0.1247`) |
| `subscription_age_days` | `0.1134` | **Negative (-)** (`-0.1134`) |
| `is_auto_renew` | `0.1022` | **Negative (-)** (`-0.1022`) |
| `subscription_expired` | `0.1012` | **Negative (-)** (`-0.1012`) |

## ⚠️ 3. Multicollinearity Detection (|r| >= 0.85)

Phát hiện **7** cặp đặc trưng có tương quan mạnh:

| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :---: |
| `days_since_last_completed_order` | `has_completed_order` | `0.9004` |
| `total_order_amounts_60d` | `total_payment_amounts_60d` | `0.8990` |
| `is_auto_renew` | `avg_usage_duration_all_time` | `0.8744` |
| `avg_usage_duration_60d` | `usage_60d_share` | `0.8654` |
| `opened_rate_60d` | `opened_rate_change` | `0.8646` |
| `avg_order_amount_60d` | `avg_payment_amount_60d` | `0.8632` |
| `subscription_expired` | `usage_duration_change` | `0.8574` |

## 🖼️ 4. Visual Charts Generated

- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.
- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.
- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.
- `04_feature_distributions.png`: Boxplot so sánh phân phối giữa 2 nhóm Churned và Retained.
- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.
