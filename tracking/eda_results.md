# Customer Churn Machine Learning & EDA System Walkthrough

Chúng ta đã hoàn thành xuất sắc việc xây dựng hệ thống **Exploratory Data Analysis (EDA)** chuyên sâu cùng kiến trúc **Modular ML Registry & Training Pipeline** cho bài toán Customer Churn Prediction trên cả 2 bộ dữ liệu:
1. **Bộ Chuỗi Thời Gian:** [`data/processed/churn_feature_dataset_processed.csv`](file:///d:/ML_intern/data/processed/churn_feature_dataset_processed.csv) (166,084 mẫu, 42 features)
2. **Bộ Dữ Liệu Tĩnh:** [`data/processed/dataset02_fixed.csv`](file:///d:/ML_intern/data/processed/dataset02_fixed.csv) (7,950 mẫu, 31 features)

---

## 📊 1. Kết Quả Khám Phá Dữ Liệu (EDA Insights)

Toàn bộ biểu đồ hình ảnh phân giải cao (High DPI) và báo cáo phân tích đã được lưu tự động tại thư mục [`reports/eda/`](file:///d:/ML_intern/reports/eda/):

### A. So Sánh Thống Kê Giữa 2 Bộ Dữ Liệu:
| Tiêu chí | 📈 Bộ Chuỗi Thời Gian (`timeseries`) | 🏷️ Bộ Dữ Liệu Tĩnh (`static`) |
| :--- | :--- | :--- |
| **Tổng số mẫu** | `166,084` dòng | `7,950` dòng |
| **Số đặc trưng** | `42` features | `31` features |
| **Tỷ lệ Churn (1)** | `18.87%` (31,340 mẫu) | `6.97%` (554 mẫu) |
| **Tỷ lệ Active (0)** | `81.13%` (134,744 mẫu) | `93.03%` (7,396 mẫu) |
| **Tỷ lệ mất cân bằng** | `1 : 4.30` | `1 : 13.35` |
| **Top tương quan nghịch (-)** | `is_paid_tier` (-0.406), `subscription_tier` (-0.272), `avg_session_duration_30d` (-0.240) | `total_interactions_all_time` (-0.125), `is_auto_renew` (-0.102) |
| **Top tương quan thuận (+)** | `payments_success_rate_missing` (+0.045) | `usage_60d_share` (+0.345), `avg_usage_duration_60d` (+0.332), `days_until_end_from_snapshot` (+0.151) |
| **Cặp đa cộng tuyến ($|r| \ge 0.85$)** | **57** cặp (chủ yếu giữa các rolling 30d/60d/90d) | **7** cặp (`avg_order_amount_60d` $\leftrightarrow$ `avg_payment_amount_60d`,...) |

### B. Danh Mục Biểu Đồ Đã Xuất:
1. **Phân phối nhãn Churn (`01_target_distribution.png`):** Biểu đồ Bar + Donut hiển thị tỷ lệ % và số lượng mẫu.
2. **Ma trận tương quan (`02_correlation_matrix_top.png`):** Heatmap ma trận tương quan giữa Top đặc trưng (ẩn tam giác trên).
3. **Xếp hạng tương quan với nhãn (`03_target_correlations.png`):** Barplot ngang xếp hạng đặc trưng ảnh hưởng mạnh nhất đến khả năng churn.
4. **Phân phối theo Churn (`04_feature_distributions.png`):** Boxplot so sánh phân phối các đặc trưng giữa nhóm Churned vs Retained.
5. **Tỷ lệ Churn theo phân loại (`05_categorical_churn_rates.png`):** Tỷ lệ Churn (%) theo từng gói cước và phân khúc.
6. **Xu hướng Churn theo thời gian (`06_timeseries_churn_trend.png`):** Biểu đồ đường kép (Dual-axis) theo 35 tháng (`snapshot_month`).
7. **Xu hướng hoạt động theo thời gian (`07_timeseries_activity_trends.png`):** Biến động trung bình của các chỉ số tương tác qua các tháng.

---

## 🛠️ 2. Cấu Trúc Các Module EDA (`src/eda/`)

```
d:/ML_intern/src/eda/
├── __init__.py
├── correlations.py      # plot_correlation_matrix, plot_target_correlations, detect_multicollinearity
├── distributions.py     # plot_target_distribution, plot_feature_distributions_by_target, plot_categorical_churn_rates
├── timeseries_eda.py    # plot_churn_trend_over_time, plot_activity_trends_over_time
└── report.py            # generate_eda_suite (Sinh toàn bộ plots và eda_summary.md)
```

### Cách chạy nhanh công cụ EDA qua CLI:
```bash
# 1. Chạy phân tích EDA cho tất cả datasets:
python -m src.run_eda --all

# 2. Chạy riêng bộ chuỗi thời gian:
python -m src.run_eda --dataset timeseries

# 3. Chạy riêng bộ tĩnh:
python -m src.run_eda --dataset static
```

---

## 🤖 3. Tổng Hợp Kiến Trúc Huấn Luyện ML & Tối Ưu Hóa (Registry Pattern)

```
d:/ML_intern/
├── src/
│   ├── data/
│   │   ├── base.py              # BaseDataset & SplitResult
│   │   └── datasets.py          # TimeSeriesDataset, StaticDataset & DatasetRegistry
│   ├── models/
│   │   ├── base.py              # BaseModelWrapper
│   │   ├── registry.py          # ModelRegistry (xgboost, random_forest, logistic_regression)
│   │   └── evaluate.py          # Evaluation metrics, Optimal Threshold Tuning, Feature Importance
│   ├── training/
│   │   └── trainer.py           # OptunaTrainer thống nhất (HPO Loop, SQLite DB, Artifacts Logging)
│   └── train.py                 # Unified CLI Runner
└── tracking/
    └── optuna_study.db          # Cơ sở dữ liệu Optuna SQLite
```

### Chạy huấn luyện và so sánh mô hình:
```bash
# Huấn luyện XGBoost trên bộ Time-Series:
python -m src.train --dataset timeseries --model xgboost --n_trials 30

# Huấn luyện XGBoost trên bộ Static:
python -m src.train --dataset static --model xgboost --n_trials 30

# Huấn luyện Baseline Random Forest trên bộ Static:
python -m src.train --dataset static --model random_forest --n_trials 20
```
