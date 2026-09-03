# EDA Summary Report - ROUND3 Dataset

## 📌 1. Dataset Dimensions & Class Distribution

- **Total Samples:** `10,002` rows
- **Total Columns:** `66` columns
- **Target Column:** `churn`
- **Positive (Churn = 1):** `692` samples (`6.92%`)
- **Negative (Active = 0):** `9,310` samples (`93.08%`)
- **Class Imbalance Ratio:** `1 : 13.45`

## 🔗 2. Top Correlated Features with Target

| Feature | Absolute Correlation | Direction |
| :--- | :---: | :---: |
| `is_auto_renew` | `0.4409` | **Negative (-)** (`-0.4409`) |
| `avg_usage_duration_60d` | `0.3721` | **Negative (-)** (`-0.3721`) |
| `usage_duration_change` | `0.3713` | **Negative (-)** (`-0.3713`) |
| `total_usage_60d` | `0.3639` | **Negative (-)** (`-0.3639`) |
| `contract_churn_risk_score` | `0.3562` | **Positive (+)** (`+0.3562`) |
| `avg_usage_duration_30d` | `0.3549` | **Negative (-)** (`-0.3549`) |
| `days_since_last_usage` | `0.3162` | **Positive (+)** (`+0.3162`) |
| `total_usage_30d` | `0.3125` | **Negative (-)** (`-0.3125`) |
| `subscription_expired` | `0.3051` | **Positive (+)** (`+0.3051`) |
| `usage_60d_share` | `0.2815` | **Negative (-)** (`-0.2815`) |

## ⚠️ 3. Multicollinearity Detection (|r| >= 0.85)

Phát hiện **22** cặp đặc trưng có tương quan mạnh:

| Feature 1 | Feature 2 | Correlation |
| :--- | :--- | :---: |
| `avg_usage_duration_60d` | `usage_duration_change` | `0.9863` |
| `total_order_amount_all_time` | `total_payment_amount_all_time` | `0.9799` |
| `total_orders_all_time` | `total_order_amount_all_time` | `0.9654` |
| `subscription_expired` | `contract_churn_risk_score` | `0.9556` |
| `total_orders_all_time` | `total_payment_amount_all_time` | `0.9441` |
| `total_order_amounts_60d` | `total_payment_amounts_60d` | `0.9216` |
| `days_until_end_from_snapshot` | `subscription_expired` | `0.9172` |
| `converted_rate_60d` | `converted_rate_change` | `0.9062` |
| `total_order_amounts_30d` | `total_payment_amounts_30d` | `0.9054` |
| `clicked_rate_60d` | `clicked_rate_change` | `0.9038` |

## 🖼️ 4. Visual Charts Generated

- `01_target_distribution.png`: Biểu đồ phân phối nhãn Churn vs Active.
- `02_correlation_matrix_top.png`: Heatmap ma trận tương quan giữa Top đặc trưng.
- `03_target_correlations.png`: Xếp hạng đặc trưng tương quan với biến mục tiêu.
- `04_feature_distributions.png`: Boxplot so sánh phân phối giữa 2 nhóm Churned và Retained.
- `05_categorical_churn_rates.png`: Tỷ lệ Churn theo các biến phân loại.
