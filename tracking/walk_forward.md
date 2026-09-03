# Customer Churn Machine Learning & Multi-Dataset Benchmark Walkthrough

Hệ thống hiện đã hỗ trợ đầy đủ cờ `--cv <k>` (ví dụ: `--cv 5`) linh hoạt cho **tất cả các dataset**:
1. **Dataset Chuỗi Thời Gian (`timeseries` / `latest`):** Tự động phát hiện và áp dụng **Walk-Forward TimeSeries Cross-Validation (Expanding Window theo tháng)** để chống rò rỉ dữ liệu qua thời gian.
2. **Dataset Tĩnh (`static`):** Áp dụng **Stratified K-Fold Cross-Validation** trên 80% tập Train và kiểm thử trên 20% Holdout Test.

---

## 📈 1. Cơ Chế 5-Fold Walk-Forward TimeSeries Validation

Khi chạy lệnh `python -m src.train --dataset timeseries --model xgboost --cv 5`, hệ thống tự động nhận diện mốc thời gian `snapshot_month` và chia 30 tháng lịch sử (`2023-07` đến `2025-12`, 122,599 mẫu) thành 5 chu kỳ cuộn tịnh tiến mở rộng:

```
                            30 Tháng Quá Khứ (122,599 mẫu)                      5 Tháng Tương Lai
   ┌────────────────────────────────────────────────────────────────────────┐ ┌─────────────────┐
   │ 2023-07 ..................................................... 2025-12  │ │ 2026-01..2026-05│
   └────────────────────────────────────────────────────────────────────────┘ └─────────────────┘
                                                                               (Holdout Test Set)
   Fold 1: [== Train (5 tháng: 2023-07..11) ==] [Val (2023-12..2024-04)]
   Fold 2: [==== Train (10 tháng: 2023-07..2024-04) ====] [Val (2024-05..09)]
   Fold 3: [====== Train (15 tháng: 2023-07..2024-09) ======] [Val (2024-10..2025-02)]
   Fold 4: [======== Train (20 tháng: 2023-07..2025-02) ========] [Val (2025-03..07)]
   Fold 5: [========== Train (25 tháng: 2023-07..2025-07) ==========] [Val (2025-08..12)]
```

### ✅ Ưu điểm của Walk-Forward CV:
* **Tuyệt đối không có Lookahead Bias:** Mô hình ở mỗi fold luôn chỉ được học dữ liệu quá khứ và dự đoán dữ liệu của các tháng tương lai liền kề.
* **Đo lường độ ổn định qua thời gian:** Điểm số Optuna là trung bình cộng của 5 chu kỳ kinh doanh khác nhau.
* **Tối ưu hóa ngưỡng Out-Of-Fold (OOF):** Tổng hợp dự đoán OOF trên 119,564 mẫu để tìm ra Decision Threshold tối ưu nhất trước khi deploy vào tập Test 2026.

---

## 📊 2. Kết Quả Huấn Luyện XGBoost Với 5-Fold Walk-Forward CV

| Tiêu chí | 🔄 **5-Fold Walk-Forward CV (Out-Of-Fold)** | 🔮 **Holdout Test Set (Năm 2026)** |
| :--- | :--- | :--- |
| **Kích thước dữ liệu** | `119,564` mẫu OOF (`29,923` Churned) | `43,485` mẫu tương lai (`570` Churned) |
| **ROC-AUC** | **`0.8933`** *(Best Trial CV: 0.8867)* | **`0.8578`** |
| **PR-AUC (Average Precision)** | **`0.6493`** | **`0.2235`** |
| **Decision Threshold Tối ưu** | `0.4852` | `0.4852` (Áp dụng từ OOF) |
| **Accuracy** | `80.28%` | **`95.29%`** |
| **Precision (Churn=1)** | `56.50%` | `12.30%` |
| **Recall (Churn=1)** | **`92.22%`** | **`42.28%`** |
| **F1-Score (Churn=1)** | **`0.7007`** | `0.1906` |
| **Top Features** | 1. `is_paid_tier` (58.1%)<br/>2. `subscription_tier` (22.7%)<br/>3. `auto_renew` (6.7%)<br/>4. `total_active_days_90d` (3.1%) | |

---

## 💻 3. Hướng Dẫn Sử Dụng Cờ `--cv` Cho Mọi Dataset

```bash
# 1. Walk-Forward 5-Fold CV trên bộ Chuỗi Thời Gian:
python -m src.train --dataset timeseries --model xgboost --n_trials 30 --cv 5

# 2. Walk-Forward 5-Fold CV trên bộ Dữ liệu Team (churn_train + churn_val):
python -m src.train --dataset latest --model xgboost --n_trials 30 --cv 5

# 3. Stratified 5-Fold CV trên bộ Dữ liệu Tĩnh:
python -m src.train --dataset static --model xgboost --n_trials 30 --cv 5
```
