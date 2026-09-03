# 🚀 Báo Cáo Triển Khai Chiến Lược Nâng Cao Cho Time-Series Churn

Chúng tôi đã hoàn thành việc nghiên cứu, xây dựng và kiểm thử thực tế **3 chiến lược then chốt** theo đúng yêu cầu:
1. **Bộ chỉ số xếp hạng rủi ro:** `Precision@Top 5%`, `Recall@Top 10%`, và **Bảng Phân Vị Lợi Thế Tích Lũy (Cumulative Lift Analysis Table)**.
2. **Chiến lược Trọng số Thời gian Lũy thừa (Exponential Recency Decay Sample Weights)**.
3. **Chiến Lược 4: Bộ dữ liệu Khách hàng Ảnh chụp Thời điểm với Đặc trưng Chuỗi Thời gian Động (Point-in-Time Customer-Level Dataset with Dynamic Rolling Features)** (`--dataset pit`).

---

## 📊 1. Bộ Chỉ Số Đánh Giá Mới: Top-K Lift & Ranking Metrics

Trong bài toán mất cân bằng dữ liệu cao (Test Churn $\approx 1.31\%$), mô hình không đánh giá theo ngưỡng đơn lẻ `0.5`, mà đo lường **khả năng sàng lọc chính xác Top khách hàng rủi ro cao nhất**:

### 🎯 Công thức tính:
* **Precision@Top K%:** Tỷ lệ khách hàng thực sự Churn trong nhóm $K\%$ có xác suất cao nhất.
* **Recall@Top K%:** Tỷ lệ Churners bắt trúng trên tổng số lượng Churners toàn bộ tập dữ liệu.
* **Cumulative Lift@Top K%:** Độ hiệu quả của mô hình so với lựa chọn ngẫu nhiên:
  $$\text{Lift@K} = \frac{\text{Precision@Top K\%}}{\text{Baseline Churn Rate}}$$

### 📈 Bảng Thống Kê Thực Tế Trên Tập Test 2026 (43,485 mẫu | 570 ca Churn):

| Phân vị Nguy cơ | 👥 Số KH Can Thiệp | 🎯 Churners Bắt Được | 🎯 Precision@K | 📈 Recall@K (Coverage) | 🚀 Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1% rủi ro cao nhất** | `434` | `148 / 570` | **`34.10%`** | **`25.96%`** | **`26.02x`** (Gấp 26 lần ngẫu nhiên) |
| **Top 2% rủi ro cao nhất** | `869` | `185 / 570` | **`21.29%`** | **`32.46%`** | **`16.24x`** |
| **Top 5% rủi ro cao nhất** | `2,174` | `248 / 570` | **`11.41%`** | **`43.51%`** | **`8.70x`** |
| **Top 10% rủi ro cao nhất** | `4,348` | `314 / 570` | **`7.22%`** | **`55.09%`** | **`5.51x`** |
| **Top 20% rủi ro cao nhất** | `8,697` | `387 / 570` | **`4.45%`** | **`67.89%`** | **`3.39x`** |

👉 **Tối ưu hóa Optuna:** Có thể chọn trực tiếp `--metric precision_at_5` hoặc `--metric recall_at_10` làm mục tiêu HPO.

---

## ⏳ 2. Chiến Lược Trọng Số Thời Gian (Exponential Recency Decay)

Để khắc phục hiện tượng dữ liệu quá khứ xa (năm 2023) bị thiếu log tương tác, thuật toán gán trọng số học giảm dần theo chu kỳ bán rã (Half-Life):

$$w_i = 2^{-\frac{T_{\text{max}} - t_i}{\text{half\_life\_months}}}$$

* Trọng số được chuẩn hóa sao cho $\bar{w} = 1.0$.
* **Phân phối trọng số thực tế khi `--decay_half_life 12`:**
  * **Năm 2023:** Trọng số trung bình = **`0.30`** (giảm thiểu nhiễu từ giai đoạn log thưa thớt)
  * **Năm 2024:** Trọng số trung bình = **`0.51`**
  * **Năm 2025:** Trọng số trung bình = **`0.99`**
  * **Năm 2026:** Trọng số trung bình = **`1.55`** (ưu tiên cao nhất cho dữ liệu gần hiện tại)

---

## 💎 3. Chiến Lược 4: Point-in-Time (PIT) Customer-Level Dataset (`--dataset pit`)

* **Bản chất:** Mỗi khách hàng duy nhất được đại diện bởi **đúng 1 dòng** tại mốc ảnh chụp lịch sử gần nhất (Point-in-Time).
* **Bảo toàn 100% Đặc trưng Động:** Giữ lại toàn bộ 42 đặc trưng chuỗi thời gian cực mạnh (`activity_slope_3m`, `usage_trend_30d`, `avg_session_duration_roll3m_mean`, `orders_roll3m_sum`, `payments_success_rate`...).
* **Kiểm định:** Chạy 5-Fold Stratified Cross-Validation trên 8,297 khách hàng lịch sử độc lập.

### 📊 Kết Quả Thực Nghiệm XGBoost trên PIT Dataset:
* **5-Fold Stratified CV (Out-Of-Fold trên 8,297 khách hàng):**
  * **OOF ROC-AUC:** **`0.9309`**
  * **OOF PR-AUC:** **`0.7867`**
  * **Precision@Top 5%:** **`94.44%`** (Lift: **`3.86x`**)
  * **Recall@Top 10%:** **`33.78%`**
  * **OOF Recall (Churn=1):** **`88.82%`**
  * **OOF F1-Score:** **`0.7453`**

---

## 💻 4. Hướng Dẫn Sử Dụng Các Lệnh Mới

```bash
# 1. Chạy Timeseries với Trọng số Thời gian Lũy thừa (Half-life = 12 tháng) và Tối ưu Precision@Top 5%:
python -m src.train --dataset timeseries --model xgboost --decay_half_life 12 --metric precision_at_5 --n_trials 30 --cv 5

# 2. Chạy Chiến Lược 4: Point-in-Time Customer-Level Dataset với 5-Fold CV:
python -m src.train --dataset pit --model xgboost --n_trials 30 --cv 5

# 3. Chạy Point-in-Time Dataset tối ưu Recall@Top 10%:
python -m src.train --dataset pit --model xgboost --metric recall_at_10 --n_trials 30 --cv 5

# 4. Chạy bộ dữ liệu Team cấp (latest) kết hợp Exponential Decay Weights:
python -m src.train --dataset latest --model xgboost --decay_half_life 12 --metric precision_at_5 --n_trials 30 --cv 5
```
