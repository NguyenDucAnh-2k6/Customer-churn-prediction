# 🎯 Customer Churn Prediction ML System

Hệ thống Machine Learning toàn diện dự báo hành vi Rời bỏ của Khách hàng (Customer Churn Prediction) được xây dựng trên nền tảng **Lakehouse (MinIO S3 / PyArrow)**, hỗ trợ tối ưu siêu tham số tự động (**Optuna HPO**), kiểm định chéo bảo đảm không rò rỉ khách hàng (**Stratified Group K-Fold Cross-Validation**), và tối ưu ngưỡng quyết định phục vụ kinh doanh (**Threshold Tuning & Top-K Lift Evaluation**).

---

## 📁 1. Cấu Trúc Thư Mục Dự Án (Project Structure)

```bash
d:/ML_intern/
├── data/                                 # Kho lưu trữ dữ liệu (Lakehouse & Processed Datasets)
│   ├── churn_customers.csv               # Bảng Silver: 10,002 hồ sơ khách hàng
│   ├── churn_orders.csv                  # Bảng Silver: 118,839 đơn hàng
│   ├── churn_payments.csv                # Bảng Silver: 199,916 giao dịch thanh toán
│   ├── churn_product_usage.csv           # Bảng Silver: 770,082 log truy cập và sử dụng app
│   ├── churn_subscriptions.csv           # Bảng Silver: 12,615 lịch sử gói thuê bao
│   ├── churn_support_tickets.csv         # Bảng Silver: 23,600 phiếu khiếu nại CSKH
│   ├── churn_marketing_interactions.csv  # Bảng Silver: 186,884 tương tác tiếp thị
│   └── processed/                        # Dữ liệu đã xử lý theo từng phiên bản
│       ├── dataset02_fixed.csv           # Bộ Static: 7,950 dòng (tương thích train_model_sklearn)
│       ├── churn_feature_dataset_processed.csv # Bộ Time-series Panel: 166,084 dòng
│       ├── latest/                       # Bộ Pre-split (train/val/test) của nhóm
│       └── round3/                       # Bộ Round 3 Point-in-Time sạch 100% (34 cột, 0% missing)
│           ├── churn_master.csv          # Master Dataset (10,002 KH x 34 cột)
│           ├── churn_train.csv           # Train Set 80% (8,002 KH x 35 cột, kèm cv_fold 0-4)
│           ├── churn_test.csv            # Test Set 20% (2,000 KH x 34 cột)
│           └── schema_data_dictionary.md # Từ điển dữ liệu tiếng Việt có dấu
├── reports/                              # Báo cáo phân tích EDA & Từ điển schema
│   ├── eda/                              # Biểu đồ và báo cáo EDA chi tiết cho từng bộ dữ liệu
│   └── schema_data_dictionary.md         # Bản sao từ điển schema Round 3
├── tracking/                             # Lưu trữ trạng thái thử nghiệm & HPO
│   └── optuna_study.db                   # SQLite DB lưu vết toàn bộ Optuna Trials
├── src/                                  # Mã nguồn module ML Pipeline
│   ├── data/                             # [DATA MODULE] Quản lý & nạp các bộ dữ liệu
│   │   ├── base.py                       # Lớp trừu tượng BaseDataset & Dataclass SplitResult
│   │   ├── registry.py                   # DatasetRegistry: Đăng ký & khởi tạo dataset
│   │   ├── datasets.py                   # Orchestrator & re-exporter tương thích ngược
│   │   ├── timeseries.py                 # TimeSeriesDataset (Panel dữ liệu chuỗi thời gian)
│   │   ├── static.py                     # StaticDataset (Dataset02 Cross-Sectional)
│   │   ├── latest.py                     # PreSplitLatestDataset (Bộ train/val/test nhóm cấp)
│   │   ├── pit.py                        # PointInTimeTimeSeriesDataset (Chiến Lược 4)
│   │   ├── round3.py                     # Round3Dataset (Point-in-Time 34 cột sạch)
│   │   └── weights.py                    # Tính trọng số mẫu động (Recency Decay) & Lọc features
│   ├── features/                         # [FEATURE ENGINEERING MODULE]
│   │   ├── preprocessor.py               # Tiền xử lý dữ liệu và gắn nhãn
│   │   ├── financial_indicators.py       # Tính chỉ số tài chính & kỹ thuật (MACD, Volatility, Z-Score)
│   │   ├── velocity.py                   # Tính gia tốc tương tác & độ dốc hoạt động
│   │   ├── cleaning.py                   # Làm sạch và điền khuyết thiếu
│   │   └── selection.py                  # Lọc Mutual Information & Đa cộng tuyến
│   ├── models/                           # [MODEL WRAPPER MODULE]
│   │   ├── base.py                       # Lớp trừu tượng BaseModelWrapper
│   │   ├── registry.py                   # ModelRegistry: Đăng ký & quản lý các mô hình
│   │   ├── xgboost_model.py              # XGBoost Classifier với HPO scale_pos_weight
│   │   ├── lightgbm.py                   # LightGBM Classifier
│   │   ├── catboost.py                   # CatBoost Classifier
│   │   ├── random_forest.py              # Random Forest Classifier (Scikit-Learn)
│   │   ├── logistic_regression.py        # Logistic Regression Classifier
│   │   ├── tabnet.py                     # PyTorch TabNet Classifier
│   │   ├── lstm.py                       # PyTorch LSTM Sequence Classifier
│   │   ├── evaluate.py                   # Tính toán Top-K Lift, PR-AUC, ROC-AUC, Brier Score
│   │   └── artifacts/                    # Thư mục lưu Model Weights, Plots & Metrics JSON
│   ├── training/                         # [TRAINING MODULE]
│   │   └── trainer.py                    # OptunaTrainer: Điều phối HPO, CV, Tuning Threshold
│   ├── train.py                          # CLI Entrypoint chính để huấn luyện mô hình
│   └── run_eda.py                        # CLI Entrypoint sinh báo cáo & biểu đồ EDA
├── collect_data.py                       # Script tải 7 bảng Silver từ MinIO S3 Lakehouse
├── build_dataset.py                      # Script xây dựng dataset Round 3 sạch 0% null
└── README.md                             # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## ⚡ 2. Quy Trình Chuẩn Bị Dữ Liệu (Data Pipeline)

### Bước 1: Tải dữ liệu Silver từ MinIO S3
Kết nối tới MinIO (`s3://lqminh/silver/devdb/`), tự động đọc các phân vùng `year=.../month=.../*.parquet` bằng PyArrow và lưu về `data/`:
```bash
python collect_data.py
```

### Bước 2: Xây dựng bộ dữ liệu Round 3 Point-in-Time
Tạo bộ dataset 1 dòng / 1 khách hàng sạch 100% không rò rỉ thông tin (Zero Customer & Target Leakage), loại bỏ cột khuyết thiếu > 15%, phân chia Train (80%)/Test (20%) bằng StratifiedGroupKFold kèm `cv_fold` (0-4):
```bash
python build_dataset.py
```

---

## 💻 3. Hướng Dẫn Sử Dụng CLI & Toàn Bộ Tham Số (CLI Reference)

Tất cả các tác vụ huấn luyện mô hình được thực thi qua lệnh:
```bash
python -m src.train [CÁC THAM SỐ]
```

### 📋 Bảng Tham Số Chi Tiết

| Tham Số CLI | Kiểu Dữ Liệu | Giá Trị Mặc Định | Mô Tả & Tác Dụng Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| **`--dataset`** | `str` | `timeseries` | Bộ dữ liệu huấn luyện. Lựa chọn: `round3` (hoặc `r3`), `static`, `timeseries`, `latest`, `pit`, `timeseries_group`. |
| **`--model`** | `str` | `xgboost` | Kiến trúc mô hình. Lựa chọn: `xgboost` (`xgb`), `lightgbm` (`lgbm`), `catboost` (`cb`), `random_forest` (`rf`), `logistic_regression` (`lr`), `tabnet`, `lstm`. |
| **`--n_trials`** | `int` | `30` | Số lượng vòng thử nghiệm tối ưu siêu tham số của Optuna. |
| **`--timeout`** | `int` | `None` | Thời gian tối đa (giây) chạy Optuna. |
| **`--metric`** | `str` | `roc_auc` | Hàm mục tiêu tối ưu trong Optuna HPO: `roc_auc`, `pr_auc`, `precision_at_5`, `recall_at_10`, `lift_top_5`, `f1_macro`, `f1`. |
| **`--cv`** | `int` | `None` | Số lượng Folds kiểm định chéo (ví dụ: `--cv 5`). Chạy 5-Fold Stratified Group CV trên Train set. |
| **`--decay_half_life`** | `float` | `None` | Chu kỳ bán rã (tháng) cho kỹ thuật gán trọng số suy giảm hàm mũ (Recency Decay Weighting) theo thời gian. |
| **`--customer_weight_power`** | `float` | `0.0` | Số mũ $\alpha$ cân bằng tần suất khách hàng ($w = 1 / N_{cust}^\alpha$). |
| **`--use_usage_weight`** | `flag` | `False` | Bật gán trọng số mẫu theo mức độ hoạt động tương tác thực tế (active days). |
| **`--drop_low_mi`** | `flag` | `False` | **Ablation Study:** Tự động loại bỏ các đặc trưng có Mutual Information gần bằng 0 (nhân khẩu học, CSAT thiếu, mở mail). |
| **`--drop_collinear`** | `flag` | `False` | **Ablation Study:** Tự động loại bỏ các đặc trưng đa cộng tuyến mạnh ($|r| \ge 0.90$). |
| **`--behavioral_only`** | `flag` | `False` | **Strategy 4:** Bỏ các biến phân loại gói dịch vụ cố định để buộc mô hình chỉ học từ chuỗi hành vi thực. |
| **`--stock_features`** | `flag` | `True` | Tích hợp các chỉ báo kỹ thuật tài chính (MACD tương tác, Biến động hoạt động, Độ sụt giảm đỉnh Drawdown). |
| **`--no_stock_features`** | `flag` | `False` | **Ablation Study:** Loại bỏ các chỉ báo tài chính kỹ thuật. |
| **`--split_strategy`** | `str` | `time_based` | Chiến lược chia tập: `group_stratified` (Zero Leakage theo khách hàng), `time_based` (Out-of-Time), `pit` (Point-in-Time). |
| **`--group_stratified`** | `flag` | `False` | Shorthand cho `--split_strategy group_stratified`. |
| **`--data_path`** | `str` | `None` | Đường dẫn file CSV tùy chỉnh nếu muốn override dữ liệu mặc định. |
| **`--artifacts_dir`** | `str` | `None` | Thư mục lưu artifacts (mô hình, đồ thị feature importance, metrics JSON). |
| **`--db_url`** | `str` | `sqlite:///tracking/optuna_study.db` | Đường dẫn SQLite Database lưu vết Optuna Study. |
| **`--study_name`** | `str` | `None` | Tên study trong Optuna (Mặc định: `{model}_{dataset}`). |
| **`--seed`** | `int` | `42` | Random seed tái lập kết quả. |
| **`--list`** | `flag` | `False` | Liệt kê toàn bộ danh sách Dataset và Model đã đăng ký rồi thoát. |

---

## 🚀 4. Các Lệnh Huấn Luyện Mẫu Phổ Biến (Examples & Recipes)

### 1. Huấn luyện mô hình XGBoost trên bộ Dataset Round 3 (Kèm 5-Fold Cross-Validation)
```bash
python -m src.train --dataset round3 --model xgboost --n_trials 30 --cv 5
```

### 2. Huấn luyện mô hình LightGBM trên bộ Static (`dataset02_fixed.csv`)
```bash
python -m src.train --dataset static --model lightgbm --n_trials 25 --cv 5
```

### 3. Huấn luyện CatBoost trên chuỗi thời gian kèm Trọng số suy giảm Recency Decay
```bash
python -m src.train --dataset timeseries --model catboost --decay_half_life 12 --n_trials 20
```

### 4. Huấn luyện Random Forest / Logistic Regression Baseline
```bash
python -m src.train --dataset round3 --model random_forest --cv 5
python -m src.train --dataset round3 --model logistic_regression --cv 5
```

### 5. Chạy thử nghiệm Ablation Study (Loại bỏ Đa cộng tuyến & Low MI)
```bash
python -m src.train --dataset round3 --model xgboost --drop_collinear --drop_low_mi --cv 5
```

### 6. Liệt kê toàn bộ Model & Dataset có sẵn trong hệ thống
```bash
python -m src.train --list
```

### 7. Sinh báo cáo Phân tích Dữ liệu Khám phá (EDA) tự động
```bash
python -m src.run_eda --dataset static
python -m src.run_eda --dataset timeseries
```

---

## 📈 5. Kết Quả Đánh Giá & Đo Lường (Evaluation Metrics)

Hệ thống tự động xuất bảng đánh giá chi tiết sau mỗi phiên huấn luyện vào `src/models/artifacts/{dataset}_{model}/`:
* **ROC-AUC & PR-AUC:** Đánh giá năng lực phân biệt và độ chính xác trung bình trên dữ liệu mất cân bằng.
* **Top-K Lift & Capture Rate (Top 1%, 2%, 5%, 10%, 20%):** Đo lường hiệu quả nhắm mục tiêu tiếp thị giữ chân khách hàng so với ngẫu nhiên (Lift thường đạt $10\times - 15\times$ ở Top 5%).
* **Tuned Threshold:** Tự động dò ngưỡng xác suất tối ưu hóa F1-Score / Lift thay vì mặc định $0.50$.
* **Feature Importance:** Xuất bảng xếp hạng và đồ thị PNG thể hiện các đặc trưng đóng góp lớn nhất vào quyết định rời bỏ.
