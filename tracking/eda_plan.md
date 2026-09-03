# Kế hoạch Xây dựng Module EDA (Exploratory Data Analysis)

## 📌 1. Mục tiêu & Tổng quan

Mục tiêu là phát triển một bộ công cụ **EDA chuyên sâu, trực quan và có tính module hóa cao (`src/eda/`)** phục vụ cho việc khám phá, phân tích tương quan, phân phối đặc trưng và xu hướng chuỗi thời gian trên cả 2 bộ dữ liệu:
1. **Bộ chuỗi thời gian:** [`data/processed/churn_feature_dataset_processed.csv`](file:///d:/ML_intern/data/processed/churn_feature_dataset_processed.csv) (166k dòng, 42 features)
2. **Bộ dữ liệu tĩnh:** [`data/processed/dataset02_fixed.csv`](file:///d:/ML_intern/data/processed/dataset02_fixed.csv) (7.9k dòng, 31 features)

---

## 🏗️ 2. Cấu Trúc Các Module EDA (`src/eda/`)

```
d:/ML_intern/
├── src/
│   └── eda/
│       ├── __init__.py
│       ├── correlations.py      # Phân tích tương quan (Correlation Matrix Heatmap, Target Correlation Barplot, Multicollinearity Detector)
│       ├── distributions.py     # Phân tích phân phối (Class Imbalance, Churn vs Non-Churn KDE/Boxplots, Categorical Churn Rates)
│       ├── timeseries_eda.py    # Phân tích xu hướng chuỗi thời gian (Monthly Churn Rate Trend, Engagement Drift theo tháng)
│       └── report.py            # Automated EDA Reporter (Tự động chạy và xuất hình ảnh + báo cáo Markdown)
├── src/
│   └── run_eda.py               # CLI runner để chạy EDA tự động
└── reports/
    └── eda/
        ├── timeseries/          # Thư mục lưu biểu đồ & báo cáo cho bộ Timeseries
        └── static/              # Thư mục lưu biểu đồ & báo cáo cho bộ Static
```

---

## 📊 3. Chi Tiết Các Biểu Đồ & Chức Năng Cụ Thể

### A. Module Phân tích Tương quan ([`src/eda/correlations.py`](file:///d:/ML_intern/src/eda/correlations.py))
1. **Correlation Matrix Heatmap (`plot_correlation_matrix`)**:
   * Vẽ ma trận tương quan (Pearson / Spearman), ẩn tam giác trên (mask upper triangle), bảng màu tương phản cao (`coolwarm` / `vlag`).
   * Tự động lọc Top K đặc trưng có tương quan cao nhất với nhãn để heatmap rõ ràng, dễ đọc.
2. **Target Correlation Ranking (`plot_target_correlations`)**:
   * Biểu đồ cột ngang xếp hạng các đặc trưng tương quan dương (+) và tương quan âm (-) mạnh nhất với `churn`.
3. **Multicollinearity Detector (`detect_multicollinearity`)**:
   * Tự động phát hiện các cặp đặc trưng bị đa cộng tuyến ($|r| > 0.85$) giúp loại bỏ đặc trưng dư thừa.

### B. Module Phân tích Phân phối ([`src/eda/distributions.py`](file:///d:/ML_intern/src/eda/distributions.py))
1. **Target Imbalance Chart (`plot_target_distribution`)**:
   * Tỷ lệ % và số lượng Churn (1) vs Active (0).
2. **Feature Distributions by Churn (`plot_feature_distributions_by_target`)**:
   * So sánh phân phối KDE / Boxplot giữa 2 nhóm khách hàng (Churned vs Retained) cho các đặc trưng quan trọng nhất (như `tenure_days`, `usage_duration`, `active_days`, `orders_last_30d`).
3. **Categorical Churn Rates (`plot_categorical_churn_rates`)**:
   * Tỷ lệ rời bỏ theo từng hạng gói dịch vụ (`subscription_tier`, `plan_tier`), hình thức gia hạn (`auto_renew`), kênh thanh toán, khu vực.

### C. Module Phân tích Chuỗi Thời Gian ([`src/eda/timeseries_eda.py`](file:///d:/ML_intern/src/eda/timeseries_eda.py))
1. **Monthly Churn Rate Trend (`plot_churn_trend_over_time`)**:
   * Biểu đồ đường kép (Dual-axis): Thể hiện tổng số lượng khách hàng hoạt động (Active Customers) và Tỷ lệ Churn (%) qua 35 tháng (`2023-07` $\to$ `2026-05`).
2. **Engagement Drift (`plot_activity_trends_over_time`)**:
   * Xu hướng biến động trung bình của các chỉ số hành vi (`total_active_days_30d`, `avg_spend_to_date_per_month`, `avg_session_duration`) theo mốc thời gian.

### D. Automated EDA Pipeline & CLI ([`src/eda/report.py`](file:///d:/ML_intern/src/eda/report.py) & [`src/run_eda.py`](file:///d:/ML_intern/src/run_eda.py))
* Cho phép sinh toàn bộ biểu đồ và báo cáo tóm tắt chỉ bằng 1 câu lệnh:
  ```bash
  # Chạy EDA cho bộ chuỗi thời gian:
  python -m src.run_eda --dataset timeseries

  # Chạy EDA cho bộ tĩnh:
  python -m src.run_eda --dataset static

  # Chạy EDA cho tất cả datasets:
  python -m src.run_eda --all
  ```

---

## 🧪 4. Kế Hoạch Kiểm Tra (Verification Plan)

1. Chạy `python -m src.run_eda --all` và kiểm tra exit code 0.
2. Kiểm tra các biểu đồ `.png` phân giải cao được xuất đầy đủ vào `reports/eda/timeseries/` và `reports/eda/static/`.
3. Kiểm tra báo cáo tổng quan `eda_summary.md` được sinh ra tự động với các insight thống kê chính xác.
