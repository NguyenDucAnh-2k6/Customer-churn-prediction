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

## 📊 2. Danh Sách & Bản Đồ Các Bộ Dữ Liệu (Dataset Catalog)

Hệ thống hỗ trợ đa dạng các chiến lược và cấu trúc dữ liệu khách hàng:

| Tên Dataset (`--dataset`) | Tên viết tắt (Aliases) | Định Dạng & Kích Thước | Chiến Lược Chia Tập | Mục Đích & Đặc Điểm Nghiệp Vụ |
| :--- | :--- | :--- | :--- | :--- |
| **`round3`** *(Khuyến nghị)* | `r3`, `round_3` | **Master Point-in-Time:** 10,002 KH x 141 features (1 dòng / 1 khách hàng) | 5-Fold Stratified Group CV (Train 80% / Test 20%) | Bộ dữ liệu snapshot chuẩn hóa sạch 100% không rò rỉ dữ liệu (Zero Leakage), tích hợp đầy đủ đặc trưng chu kỳ, động học và chỉ báo tài chính. |
| **`round3_timeseries`** | `r3_timeseries`, `timeseries_round3` | **Monthly Panel:** 184,479 dòng x 141 features (10,002 KH qua các tháng) | 5-Fold Customer Stratified Group CV | Chuỗi thời gian panel sinh trực tiếp từ MinIO Lakehouse (lqminh) với nhãn mục tiêu đa thành phần (Account Deletion, Free/Downgrade Inactivity). |
| **`timeseries`** | `timeseries_group`, `group_stratified` | **Monthly Panel:** 166,084 dòng x 56 features | Time-based (OOT) hoặc Group Stratified | Bảng dữ liệu chuỗi thời gian panel hàng tháng kinh điển từ `churn_feature_dataset_processed.csv`. |
| **`static`** | `dataset02` | **Cross-Sectional:** 7,950 dòng x 34 features (1 dòng / 1 khách hàng) | Random Holdout 80/20 hoặc K-Fold CV | Bộ dữ liệu tĩnh ban đầu từ `dataset02_fixed.csv`, tương thích với notebook baseline. |
| **`latest`** | `presplit`, `churn_team`, `latest_group` | **Pre-split Panel:** Train (115k), Val (24k), Test (26k) | Pre-split sẵn hoặc Group Stratified | Bộ dữ liệu chuỗi thời gian do nhóm phân tách sẵn trong `data/processed/latest/`. |
| **`pit`** | `point_in_time`, `timeseries_pit` | **Single Snapshot PIT:** 1 dòng mới nhất của mỗi khách hàng | Time-based hoặc Group Stratified | Rút trích snapshot điểm thời gian mới nhất từ chuỗi thời gian panel. |

---

## 🧩 3. Ma Trận Tương Thích Flag CLI $\times$ Dataset (Compatibility Matrix)

Bảng dưới đây xác định chính xác các flag CLI được áp dụng cho từng bộ dữ liệu:

| Nhóm Tính Năng | Flag CLI | `round3` (PIT) | `round3_timeseries` | `timeseries` | `latest` / `pit` | `static` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Lọc Thống Kê EDA** *(Mới)* | `--filter_kde` (KS-Test $D_{KS}$) | ✅ | ✅ | ✅ | ✅ | ❌ |
| | `--filter_categorical` (Cramér's V, IV) | ✅ | ✅ | ✅ | ✅ | ❌ |
| | `--filter_boxplot` (Cohen's d, IQR overlap) | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Ablation Feature** | `--drop_low_mi` (Mutual Information) | ✅ | ✅ | ✅ | ✅ | ❌ |
| | `--drop_collinear` (Đa cộng tuyến $|r| \ge 0.9$) | ✅ | ✅ | ✅ | ✅ | ❌ |
| | `--behavioral_only` (Bỏ static tiers) | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Làm Giàu Đặc Trưng** | `--stock_features` / `--no_stock_features` | ✅ | ✅ | ✅ | ✅ | ❌ |
| | `--dynamic_contract_features` / `--no_dynamic_contract_features` | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Trọng Số Mẫu (Weights)** | `--decay_half_life` (Recency Decay) | ❌ *(PIT)* | ✅ | ✅ | ✅ | ❌ |
| | `--customer_weight_power` (Cân bằng KH) | ❌ *(PIT)* | ✅ | ✅ | ✅ | ❌ |
| | `--use_usage_weight` (Active days) | ❌ *(PIT)* | ✅ | ✅ | ✅ | ❌ |
| **Chia Tập & CV** | `--cv [N]` (Stratified Group CV) | ✅ *(Fold 0-4)* | ✅ *(Group CV)* | ✅ | ✅ | ✅ |
| | `--split_strategy` / `--group_stratified` | ❌ *(Cố định 0-leak)* | ✅ | ✅ | ✅ | ❌ |
| **Optuna & Mô Hình** | `--model`, `--n_trials`, `--metric`, `--seed` | ✅ | ✅ | ✅ | ✅ | ✅ |
| | `--db_url`, `--study_name`, `--artifacts_dir` | ✅ | ✅ | ✅ | ✅ | ✅ |

*Chú thích:*
* ✅: Hỗ trợ và hoạt động đầy đủ.
* ❌: Không áp dụng do cấu trúc dữ liệu không phù hợp (ví dụ: bộ PIT chỉ có 1 snapshot nên không tính Recency Decay).

---

## 💻 4. Hướng Dẫn Sử Dụng CLI & Toàn Bộ Tham Số (CLI Reference)

Tất cả các tác vụ huấn luyện mô hình được thực thi qua lệnh:
```bash
python -m src.train [CÁC THAM SỐ]
```

### 📋 Bảng Tham Số Chi Tiết

| Tham Số CLI | Kiểu Dữ Liệu | Mặc Định | Dataset Áp Dụng | Mô Tả & Tác Dụng Nghiệp Vụ |
| :--- | :---: | :---: | :---: | :--- |
| **`--dataset`** | `str` | `timeseries` | *Tất cả* | Bộ dữ liệu huấn luyện: `round3` (`r3`), `round3_timeseries` (`r3_timeseries`), `timeseries`, `static`, `latest`, `pit`. |
| **`--model`** | `str` | `xgboost` | *Tất cả* | Kiến trúc mô hình: `xgboost` (`xgb`), `lightgbm` (`lgbm`), `catboost` (`cb`), `random_forest` (`rf`), `logistic_regression` (`lr`), `tabnet`, `lstm`. |
| **`--n_trials`** | `int` | `30` | *Tất cả* | Số lượng vòng thử nghiệm tối ưu siêu tham số của Optuna. |
| **`--timeout`** | `int` | `None` | *Tất cả* | Thời gian tối đa (giây) chạy Optuna. |
| **`--metric`** | `str` | `roc_auc` | *Tất cả* | Hàm mục tiêu tối ưu: `roc_auc`, `pr_auc`, `precision_at_5`, `recall_at_10`, `lift_top_5`, `f1_macro`, `f1`. |
| **`--cv`** | `int` | `None` | *Tất cả* | Số lượng Fold kiểm định chéo (ví dụ: `--cv 5`). Chạy Customer-Stratified Group CV trên tập Train. |
| **`--filter_kde`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Lọc KDE Phân Phối Trùng Nhau:** Tự động loại bỏ các biến liên tục có Kolmogorov-Smirnov $D_{KS} < \text{ks\_threshold}$ giữa Churn vs Retained. |
| **`--ks_threshold`** | `float` | `0.05` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Ngưỡng kiểm định 2-Sample KS test tối thiểu cho `--filter_kde`. |
| **`--filter_categorical`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Lọc Biểu Đồ Cột Đẳng Tỷ Lệ:** Loại bỏ biến danh mục có Cramér's V $< \text{threshold}$ hoặc IV $< \text{threshold}$. |
| **`--cramers_v_threshold`** | `float` | `0.03` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Ngưỡng Cramér's V tối thiểu cho `--filter_categorical`. |
| **`--iv_threshold`** | `float` | `0.02` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Ngưỡng Information Value (IV) tối thiểu cho `--filter_categorical`. |
| **`--filter_boxplot`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Lọc Boxplot Trùng Lặp Mean/IQR:** Loại bỏ các biến có Cohen's d $< \text{threshold}$ và độ chồng lấn IQR $> \text{threshold}$. |
| **`--cohens_d_threshold`** | `float` | `0.08` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Ngưỡng chênh lệch trung bình chuẩn hóa Cohen's d cho `--filter_boxplot`. |
| **`--iqr_overlap_threshold`** | `float` | `0.90` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Ngưỡng tỷ lệ chồng lấn hộp IQR tối đa cho `--filter_boxplot`. |
| **`--drop_low_mi`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Ablation:** Tự động loại bỏ các đặc trưng có Mutual Information gần bằng 0. |
| **`--drop_collinear`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Ablation:** Tự động loại bỏ các đặc trưng đa cộng tuyến mạnh ($|r| \ge 0.90$). |
| **`--behavioral_only`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Strategy 4:** Bỏ các biến phân loại gói dịch vụ tĩnh để mô hình chỉ học từ chuỗi hành vi. |
| **`--stock_features`** | `flag` | `True` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Tích hợp chỉ báo tài chính (MACD, Biến động hoạt động, Drawdown sụt giảm). |
| **`--no_stock_features`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Ablation:** Tắt các chỉ báo kỹ thuật tài chính. |
| **`--dynamic_contract_features`** | `flag` | `True` | `round3`, `round3_timeseries`, `timeseries`, `latest` | Tích hợp đặc trưng động học chu kỳ thanh toán & hợp đồng (velocity 30d vs 60d, khoảng cách tái ký, trễ hạn). |
| **`--no_dynamic_contract_features`** | `flag` | `False` | `round3`, `round3_timeseries`, `timeseries`, `latest` | **Ablation:** Tắt các đặc trưng động học chu kỳ hợp đồng. |
| **`--decay_half_life`** | `float` | `None` | `round3_timeseries`, `timeseries`, `latest`, `pit` | Chu kỳ bán rã (tháng) cho trọng số suy giảm hàm mũ (Recency Decay Weighting). |
| **`--customer_weight_power`** | `float` | `0.0` | `round3_timeseries`, `timeseries`, `latest`, `pit` | Số mũ $\alpha$ cân bằng tần suất khách hàng ($w = 1 / N_{cust}^\alpha$). |
| **`--use_usage_weight`** | `flag` | `False` | `round3_timeseries`, `timeseries`, `latest` | Bật gán trọng số mẫu theo mức độ hoạt động tương tác (active days). |
| **`--split_strategy`** | `str` | `time_based` | `timeseries`, `latest`, `pit` | Chiến lược chia tập: `group_stratified` (Zero Leakage theo khách hàng), `time_based` (Out-of-Time). |
| **`--group_stratified`** | `flag` | `False` | `timeseries`, `latest`, `pit` | Shorthand cho `--split_strategy group_stratified`. |
| **`--data_path`** | `str` | `None` | *Tất cả* | Đường dẫn file CSV tùy chỉnh nếu muốn override dữ liệu mặc định. |
| **`--artifacts_dir`** | `str` | `None` | *Tất cả* | Thư mục lưu artifacts (mô hình, đồ thị feature importance, metrics JSON). |
| **`--db_url`** | `str` | `sqlite:///tracking/optuna_study.db` | *Tất cả* | Đường dẫn SQLite Database lưu vết Optuna Study. |
| **`--study_name`** | `str` | `None` | *Tất cả* | Tên study trong Optuna (Mặc định: `{model}_{dataset}`). |
| **`--seed`** | `int` | `42` | *Tất cả* | Random seed tái lập kết quả. |
| **`--list`** | `flag` | `False` | *Tất cả* | Liệt kê toàn bộ danh sách Dataset và Model đã đăng ký rồi thoát. |

---

## 🚀 5. Các Lệnh Huấn Luyện Mẫu Phổ Biến (Examples & Recipes)

### 1. Huấn luyện XGBoost trên Round 3 Point-in-Time (Kèm 5-Fold Stratified Group CV)
```bash
python -m src.train --dataset round3 --model xgboost --n_trials 30 --cv 5
```

### 2. Huấn luyện XGBoost trên Round 3 Time-Series Panel (184k dòng)
```bash
python -m src.train --dataset round3_timeseries --model xgboost --n_trials 30 --cv 5
```

### 3. Thực nghiệm Lọc Thống Kê EDA (Ablation: KDE / Cột Danh Mục / Boxplot)
```bash
# 1. Lọc các đặc trưng có KDE phân phối trùng nhau (2-Sample KS test):
python -m src.train --dataset round3 --model xgboost --filter_kde --n_trials 20 --cv 5

# 2. Lọc các đặc trưng danh mục có tỷ lệ Churn/Retained sát nhau (Cramér's V & IV):
python -m src.train --dataset round3 --model xgboost --filter_categorical --n_trials 20 --cv 5

# 3. Lọc các đặc trưng có Boxplot trùng khớp mean/IQR:
python -m src.train --dataset round3 --model xgboost --filter_boxplot --n_trials 20 --cv 5

# 4. Kết hợp cả 3 bộ lọc thống kê tối ưu:
python -m src.train --dataset round3 --model xgboost --filter_kde --filter_categorical --filter_boxplot --n_trials 20 --cv 5
```

### 4. Huấn luyện CatBoost trên chuỗi thời gian kèm Trọng số suy giảm Recency Decay
```bash
python -m src.train --dataset timeseries --model catboost --decay_half_life 12 --n_trials 20
```

### 5. Huấn luyện LightGBM trên bộ Static (`dataset02_fixed.csv`)
```bash
python -m src.train --dataset static --model lightgbm --n_trials 25 --cv 5
```

### 6. Thử nghiệm Ablation Bỏ Chỉ Báo Tài Chính (No Stock Features)
```bash
python -m src.train --dataset round3 --model xgboost --no_stock_features --cv 5
```

### 7. Liệt kê toàn bộ Model & Dataset có sẵn trong hệ thống
```bash
python -m src.train --list
```

### 8. Sinh báo cáo Phân tích Dữ liệu Khám phá (EDA) & Báo cáo Sàng lọc Univariate
```bash
# EDA & Báo cáo Sàng lọc đặc trưng thống kê (10_univariate_feature_screening.csv) cho Round 3:
python -m src.run_eda --dataset round3

# EDA cho bộ Round 3 TimeSeries Panel:
python -m src.run_eda --dataset round3_timeseries

# Chạy EDA tự động trên toàn bộ các bộ dữ liệu:
python -m src.run_eda --all
```

---

## 📈 6. Kết Quả Đánh Giá & Đo Lường (Evaluation Metrics)

Hệ thống tự động xuất bảng đánh giá chi tiết sau mỗi phiên huấn luyện vào `src/models/artifacts/{dataset}_{model}/`:
* **ROC-AUC & PR-AUC:** Đánh giá năng lực phân biệt và độ chính xác trung bình trên dữ liệu mất cân bằng.
* **Top-K Lift & Capture Rate (Top 1%, 2%, 5%, 10%, 20%):** Đo lường hiệu quả nhắm mục tiêu tiếp thị giữ chân khách hàng so với ngẫu nhiên (Lift thường đạt $2.06\times$ trên PIT và $6.04\times$ trên Timeseries ở Top 5%).
* **Tuned Threshold:** Tự động dò ngưỡng xác suất tối ưu hóa F1-Score / Lift thay vì mặc định $0.50$.
* **Feature Importance:** Xuất bảng xếp hạng và đồ thị PNG thể hiện các đặc trưng đóng góp lớn nhất vào quyết định rời bỏ.

