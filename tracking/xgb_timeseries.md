# Customer Churn Prediction - XGBoost & Optuna HPO Walkthrough

Chúng ta đã hoàn thành việc xây dựng toàn bộ pipeline huấn luyện mô hình **XGBoost (XGBClassifier)** kết hợp **Optuna Hyperparameter Optimization** (lưu trữ trực tiếp vào Optuna SQLite Database) trên bộ dữ liệu chuỗi thời gian [`data/processed/churn_feature_dataset_processed.csv`](file:///d:/ML_intern/data/processed/churn_feature_dataset_processed.csv).

---

## 🎯 1. Chiến Lược Phân Chia Dữ Liệu Thời Gian (Time-based Split)

Bộ dữ liệu chứa **166,084 dòng** và **42 features** từ `2023-07` đến `2026-05`:
* **Train Set ($\le$ 2025-09):** `99,459` mẫu | Tỷ lệ Churn: **25.83%** (`25,688` positive).
* **Validation Set (2025-10 $\to$ 2025-12):** `23,140` mẫu | Tỷ lệ Churn: **21.96%** (`5,082` positive).
* **Holdout Test Set ($\ge$ 2026-01):** `43,485` mẫu | Tỷ lệ Churn: **1.31%** (`570` positive).

---

## ⚙️ 2. Quá Trình Tối Ưu Siêu Tham Số (Optuna HPO)

* **Database Storage:** SQLite DB tại [`tracking/optuna_study.db`](file:///d:/ML_intern/tracking/optuna_study.db) (Study Name: `xgb_churn_feature_timeseries`).
* **Tổng số Trials:** `33` trials hoàn thành.
* **Thời gian thực thi:** ~1.26 phút (sử dụng `tree_method='hist'`, `n_jobs=-1` và `early_stopping_rounds=30`).
* **Best Trial (#31) Metric (ROC-AUC):** **`0.91286`**

### 🏆 Bộ Siêu Tham Số Tối Ưu Nhất:
| Siêu tham số | Giá trị tối ưu | Ý nghĩa |
| :--- | :--- | :--- |
| `n_estimators` | `400` (dừng sớm tại cây `#383`) | Số lượng cây Boosting |
| `max_depth` | `5` | Độ sâu tối đa mỗi cây |
| `learning_rate` | `0.04788` | Tốc độ học (Shrinkage) |
| `min_child_weight` | `4` | Trọng số tối thiểu mẫu trên lá |
| `subsample` | `0.7359` | Tỷ lệ lấy mẫu dòng ngẫu nhiên |
| `colsample_bytree` | `0.9012` | Tỷ lệ lấy mẫu đặc trưng cho mỗi cây |
| `gamma` | `4.9192` | Ngưỡng phạt phân nhánh (Pruning) |
| `reg_alpha` (L1) | `8.6926` | L1 Regularization giúp loại bỏ nhiễu |
| `reg_lambda` (L2) | `0.0023` | L2 Regularization kiểm soát độ mượt |
| `scale_pos_weight` | `2.0275` | Trọng số phạt nhãn Churn (Imbalance control) |

---

## 📊 3. Hiệu Năng Mô Hình (Model Evaluation)

### A. Kết Quả Trên Tập Validation Set (`23,140` mẫu)
| Metric | Ngưỡng Mặc Định (`0.5000`) | Ngưỡng Tối Ưu (`0.5346`) |
| :--- | :---: | :---: |
| **ROC-AUC** | **`0.9129`** | **`0.9129`** |
| **PR-AUC (Avg Precision)** | **`0.6726`** | **`0.6726`** |
| **Accuracy** | `82.36%` | `82.84%` |
| **Precision (Churn=1)** | `55.94%` | `56.85%` |
| **Recall (Churn=1)** | **`92.76%`** | **`90.69%`** |
| **F1-Score (Churn=1)** | **`0.6979`** | **`0.6989`** |
| **Macro F1-Score** | `0.7867` | `0.7895` |

> [!TIP]
> Mô hình bắt được **> 90% khách hàng sắp rời bỏ (Recall 90.69% - 92.76%)** trên tập Validation với ROC-AUC vượt trội **0.9129**.

---

### B. Kết Quả Trên Tập Test Set Tương Lai (`43,485` mẫu)
| Metric | Ngưỡng Mặc Định (`0.5000`) | Ngưỡng Tối Ưu (`0.5346`) |
| :--- | :---: | :---: |
| **ROC-AUC** | **`0.8434`** | **`0.8434`** |
| **PR-AUC (Avg Precision)** | `0.2380` | `0.2380` |
| **Accuracy** | `94.88%` | `95.12%` |
| **Precision (Churn=1)** | `11.94%` | `12.26%` |
| **Recall (Churn=1)** | `45.61%` | `44.21%` |
| **F1-Score (Churn=1)** | `0.1892` | `0.1919` |

---

## 🌟 4. Top 15 Đặc Trưng Quan Trọng Nhất (Feature Importances by Gain)

| Thứ hạng | Tên Đặc Trưng | Gain Score | Tỷ lệ (%) | Nhận xét Nghiệp Vụ |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `is_paid_tier` | **`3,906.41`** | **62.24%** | Khách dùng gói trả phí hay miễn phí quyết định phần lớn xu hướng churn. |
| **2** | `auto_renew` | **`470.20`** | **7.49%** | Tính năng tự động gia hạn gói cước. |
| **3** | `subscription_tier` | **`445.72`** | **7.10%** | Hạng gói cước dịch vụ (Basic/Pro/Enterprise). |
| **4** | `total_active_days_60d` | **`375.74`** | **5.99%** | Tần suất hoạt động trong 60 ngày gần nhất. |
| **5** | `total_active_days_90d` | **`335.05`** | **5.34%** | Tần suất hoạt động trong 90 ngày gần nhất. |
| **6** | `total_active_days_30d` | `113.06` | 1.80% | Tần suất hoạt động trong 30 ngày gần nhất. |
| **7** | `days_since_last_usage_event` | `74.74` | 1.19% | Số ngày kể từ lần tương tác sản phẩm gần nhất. |
| **8** | `avg_spend_to_date_per_month` | `65.51` | 1.04% | Chi tiêu trung bình mỗi tháng. |
| **9** | `activity_slope_3m` | `53.75` | 0.86% | Độ dốc/xu hướng suy giảm hoạt động trong 3 tháng. |
| **10** | `num_usage_events_30d_lag1m` | `50.16` | 0.80% | Sự kiện sử dụng tháng trước đó (Lag 1 month). |
| **11** | `orders_last_90d` | `45.89` | 0.73% | Tổng đơn hàng trong 90 ngày. |
| **12** | `avg_session_duration_roll3m_mean` | `32.51` | 0.52% | Thời lượng phiên trung bình rolling 3 tháng. |
| **13** | `orders_roll3m_sum` | `32.36` | 0.52% | Tổng đơn hàng rolling 3 tháng. |
| **14** | `days_since_last_activity` | `29.00` | 0.46% | Khoảng cách ngày từ hoạt động gần nhất. |
| **15** | `num_usage_events_roll3m_sum` | `23.14` | 0.37% | Tổng số sự kiện sử dụng rolling 3 tháng. |

---

## 📁 5. Danh Sách Các File & Artifacts Đã Tạo

1. **Mã nguồn:**
   * [`src/models/__init__.py`](file:///d:/ML_intern/src/models/__init__.py): Khởi tạo module.
   * [`src/models/evaluate.py`](file:///d:/ML_intern/src/models/evaluate.py): Module tính toán metric, tìm ngưỡng tối ưu và tính Feature Importance.
   * [`src/models/train_xgb_timeseries.py`](file:///d:/ML_intern/src/models/train_xgb_timeseries.py): Script hoàn chỉnh chạy pipeline, kết nối Optuna SQLite DB, huấn luyện và lưu artifacts.
2. **Cơ sở dữ liệu HPO:**
   * [`tracking/optuna_study.db`](file:///d:/ML_intern/tracking/optuna_study.db): SQLite Database lưu trữ toàn bộ 33 trials của Optuna.
3. **Model & Báo cáo Artifacts:**
   * [`src/models/artifacts/xgb_best_model.json`](file:///d:/ML_intern/src/models/artifacts/xgb_best_model.json): Checkpoint mô hình XGBoost tốt nhất.
   * [`src/models/artifacts/best_params.json`](file:///d:/ML_intern/src/models/artifacts/best_params.json): Cấu hình siêu tham số và ngưỡng tối ưu.
   * [`src/models/artifacts/feature_importance.csv`](file:///d:/ML_intern/src/models/artifacts/feature_importance.csv): Bảng xếp hạng 42 đặc trưng theo Gain & Weight.
   * [`src/models/artifacts/evaluation_summary.json`](file:///d:/ML_intern/src/models/artifacts/evaluation_summary.json): Chi tiết các chỉ số đánh giá trên cả tập Val và Test.
