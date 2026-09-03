# Customer Churn Prediction - XGBoost & Optuna Comparison Walkthrough

Chúng ta đã hoàn thành việc xây dựng, tối ưu siêu tham số bằng **Optuna** (lưu trữ đồng thời vào SQLite Database [`tracking/optuna_study.db`](file:///d:/ML_intern/tracking/optuna_study.db)), và đánh giá hiệu năng mô hình **XGBoost** trên cả **2 bộ dữ liệu** trong `data/processed/`:

1. **Bộ 1 (Time-Series Approach):** [`data/processed/churn_feature_dataset_processed.csv`](file:///d:/ML_intern/data/processed/churn_feature_dataset_processed.csv) (166,084 mẫu snapshot tháng, 42 features).
2. **Bộ 2 (Static Cross-Sectional Approach):** [`data/processed/dataset02_fixed.csv`](file:///d:/ML_intern/data/processed/dataset02_fixed.csv) (7,950 khách hàng tĩnh, 31 features tổng hợp).

---

## ⚖️ 1. Bảng So Sánh Đối Chiếu Toàn Diện (Side-by-Side Comparison)

| Tiêu Chí So Sánh | 📈 Bộ 1: Chuỗi Thời Gian (Time-Series) | 🏷️ Bộ 2: Tĩnh (Static Cross-Sectional) |
| :--- | :--- | :--- |
| **File Dữ Liệu** | `churn_feature_dataset_processed.csv` | `dataset02_fixed.csv` |
| **Bản chất dữ liệu** | Đa mốc thời gian (Monthly Snapshots) | 1 dòng = 1 khách hàng (Static Summary) |
| **Kích thước mẫu** | `166,084` dòng $\times$ `45` cột | `7,950` dòng $\times$ `33` cột |
| **Số lượng Features** | `42` đặc trưng (Rolling, Lags, Trends, Slopes) | `31` đặc trưng (All-time, 60d, Changes, Share) |
| **Tỷ lệ nhãn Churn (Positive)** | `18.87%` tổng thể (Train: 25.8%, Test: 1.3%) | `6.97%` (Khớp Ground Truth tỷ lệ Closed) |
| **Chiến lược Split** | **Time-based Split** (Quá khứ $\to$ Tương lai) | **Stratified Split** (70% Train, 15% Val, 15% Test) |
| **Optuna Study Name** | `xgb_churn_feature_timeseries` | `xgb_churn_fixed_static` |
| **Validation ROC-AUC** | **`0.9129`** | **`0.9794`** |
| **Validation PR-AUC** | **`0.6726`** | **`0.8129`** |
| **Validation Recall (Churn=1)** | **`92.76%`** (th=0.50) / **`90.69%`** (th=0.53) | **`91.57%`** (th=0.50) / **`63.86%`** (th=0.76) |
| **Validation F1-Score** | **`0.6979`** (th=0.50) / **`0.6989`** (th=0.53) | **`0.6387`** (th=0.50) / **`0.6839`** (th=0.76) |
| **Test Set ROC-AUC** | **`0.8434`** (Tập tương lai 2026) | **`0.9710`** (Tập holdout ngẫu nhiên) |
| **Test Set PR-AUC** | **`0.2380`** | **`0.7611`** |
| **Test Set Recall (Churn=1)** | **`45.61%`** (th=0.50) / **`44.21%`** (th=0.53) | **`85.54%`** (th=0.50) / **`59.04%`** (th=0.76) |
| **Test Set F1-Score** | **`0.1892`** (th=0.50) / **`0.1919`** (th=0.53) | **`0.5941`** (th=0.50) / **`0.6405`** (th=0.76) |

---

## 🔍 2. Kiểm Chứng Đặc Trưng & So Sánh Tầm Quan Trọng (Feature Importance)

### A. Kiểm chứng tính chất "Tĩnh" của Bộ 2 (`dataset02_fixed.csv`):
* Không chứa bất kỳ cột ID chuỗi thời gian nào (`snapshot_month`, `snapshot_date`, `ord`).
* Các yếu tố hành vi được nén thành các chỉ số tĩnh tổng hợp:
  * Tổng / trung bình trong 60 ngày: `total_usage_60d`, `total_order_amounts_60d`, `total_payments_60d`.
  * Thay đổi tốc độ tương tác: `opened_rate_change`, `clicked_rate_change`, `usage_duration_change`.
  * Trạng thái subscription: `subscription_expired`, `days_until_end_from_snapshot`, `is_auto_renew`.

### B. So Sánh Top Features Giữa 2 Cách Tiếp Cận:
```
📈 Bộ 1 (Time-Series Approach)              🏷️ Bộ 2 (Static Approach)
1. is_paid_tier (62.24%)                    1. subscription_expired (28.64%)
2. auto_renew (7.49%)                       2. days_until_end_from_snapshot (25.79%)
3. subscription_tier (7.10%)                3. avg_usage_duration_all_time (3.46%)
4. total_active_days_60d (5.99%)            4. usage_60d_share (3.35%)
5. total_active_days_90d (5.34%)            5. is_auto_renew (3.23%)
```

---

## 💡 3. Phân Tích & Đánh Giá Chuyên Sâu (Key Takeaways)

> [!NOTE]
> **1. Bộ dữ liệu Tĩnh (`dataset02_fixed.csv`) đạt điểm số ROC-AUC rất cao (0.9710):**
> Do có sự hiện diện của các đặc trưng trực tiếp như `subscription_expired` (hết hạn dịch vụ) và `days_until_end_from_snapshot` (số ngày đến ngày kết thúc hợp đồng), mô hình có tín hiệu phân tách cực kỳ rõ ràng giữa khách hàng hủy và duy trì.

> [!TIP]
> **2. Bộ dữ liệu Chuỗi Thời Gian (`churn_feature_dataset_processed.csv`) phản ánh bài toán thực tế (Real-world Monitoring):**
> * Cho phép theo dõi xu hướng suy giảm (`activity_slope_3m`, `usage_trend_30d`, `lag1m`) theo từng tháng để đưa ra cảnh báo sớm trước khi khách hàng chính thức rời bỏ.
> * Việc kiểm tra trên tập Test 2026 (Out-of-time evaluation) đo lường khả năng thích ứng của mô hình khi phân phối churn thay đổi theo thời gian.

---

## 📁 4. Cấu Trúc File & Model Artifacts Hoàn Chỉnh

```
d:/ML_intern/
├── src/
│   └── models/
│       ├── __init__.py
│       ├── evaluate.py                  # Module đánh giá, threshold tuning, feature importance
│       ├── train_xgb_timeseries.py      # Pipeline huấn luyện Bộ 1 (Time-series)
│       ├── train_xgb_static.py          # Pipeline huấn luyện Bộ 2 (Static)
│       └── artifacts/
│           ├── xgb_best_model.json      # Best model Bộ 1
│           ├── best_params.json         # Best params Bộ 1
│           ├── feature_importance.csv   # Feature importance Bộ 1
│           ├── evaluation_summary.json  # Metrics summary Bộ 1
│           └── static/
│               ├── xgb_best_model.json      # Best model Bộ 2
│               ├── best_params.json         # Best params Bộ 2
│               ├── feature_importance.csv   # Feature importance Bộ 2
│               └── evaluation_summary.json  # Metrics summary Bộ 2
└── tracking/
    └── optuna_study.db                  # SQLite DB chứa cả 2 studies:
                                         #  - 'xgb_churn_feature_timeseries' (33 trials)
                                         #  - 'xgb_churn_fixed_static' (30 trials)
```
