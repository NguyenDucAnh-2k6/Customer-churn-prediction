# 🚀 Báo Cáo Triển Khai Cơ Chế Trọng Số Đa Yếu Tố (Multi-Factor Dynamic Sample Weights) Cho Time-Series

Chúng tôi đã hoàn thành việc xây dựng và tích hợp **Hệ Thống Trọng Số Đa Yếu Tố (Dynamic Sample Weighting)** trực tiếp vào toàn bộ pipeline huấn luyện chuỗi thời gian (`timeseries` và `latest`).

---

## 🎯 1. Nguyên Lý Toán Học Của Cơ Chế Trọng Số

Trọng số mẫu tổng hợp $W_{i, t}$ được tính toán động cho từng dòng quan sát của khách hàng $i$ tại mốc tháng $t$:

$$W_{i, t} = w_{\text{time}}(t) \times w_{\text{cust}}(i) \times w_{\text{usage}}(i, t)$$

$$\tilde{W}_{i, t} = \frac{W_{i, t}}{\bar{W}} \quad \left(\text{Chuẩn hóa để } \bar{W} = 1.0\right)$$

### 🔹 3 Thành Phần Trọng Số:
1. **$w_{\text{time}}(t)$ (Trọng số Thời gian - Exponential Recency Decay):**
   $$w_{\text{time}}(t) = 2^{-\frac{T_{\text{max}} - t}{\text{decay\_half\_life}}}$$
   * *Tác dụng:* Giảm trọng số của dữ liệu quá khứ xa (năm 2023 còn $\approx 0.26$) và tăng mạnh trọng số của dữ liệu gần hiện tại (năm 2025 đạt $\approx 1.28$).
2. **$w_{\text{cust}}(i)$ (Trọng số Cân bằng Tần suất Khách hàng - Inverse Customer Frequency):**
   $$w_{\text{cust}}(i) = \frac{1}{(N_i)^\alpha} \quad (\text{với } \alpha = \text{customer\_weight\_power} \in [0, 1])$$
   * *Tác dụng:* Khách hàng có 25-30 tháng lịch sử sẽ không còn áp đảo khách hàng mới (với $\alpha = 0.5$, tỷ trọng đóng góp cả đời được cân bằng mềm theo căn bậc hai).
3. **$w_{\text{usage}}(i, t)$ (Trọng số Mức độ Tương tác - Engagement Activity):**
   $$w_{\text{usage}}(i, t) = 1 + \log(1 + \text{total\_active\_days\_30d}_{i, t})$$
   * *Tác dụng:* Ưu tiên học kỹ hơn ở những tháng khách hàng có hoạt động rõ rệt.

---

## 📊 2. Kết Quả Thực Nghiệm Trên Toàn Bộ Dữ Liệu Time-Series (122,599 Mẫu Train | 43,485 Mẫu Test 2026)

Lệnh thực thi:
```bash
python -m src.train --dataset timeseries --model xgboost \
    --decay_half_life 12 \
    --customer_weight_power 0.5 \
    --use_usage_weight \
    --metric precision_at_5 \
    --n_trials 30 \
    --cv 5
```

### 📈 Hiệu Năng Xếp Hạng Top-K Trên Tập Test Năm 2026 (570 Churners / 43,485 Users):

| Phân Vị Nguy Cơ ($K\%$) | 👥 Số KH Nhắm Tới ($n_K$) | 🎯 Số Churners Bắt Được | 🎯 Precision@K | 📈 Recall@K (Coverage) | 🚀 Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1% rủi ro cao nhất** | `434` khách hàng | **147** / 570 ca | **`33.87%`** | **`25.79%`** | **`25.84x`** (Gấp 26 lần ngẫu nhiên) |
| **Top 2% rủi ro cao nhất** | `869` khách hàng | **177** / 570 ca | **`20.37%`** | **`31.05%`** | **`15.54x`** |
| **Top 5% rủi ro cao nhất** | `2,174` khách hàng | **234** / 570 ca | **`10.76%`** | **`41.05%`** | **`8.21x`** |
| **Top 10% rủi ro cao nhất** | `4,348` khách hàng | **327** / 570 ca | **`7.52%`** | **`57.37%`** | **`5.74x`** |
| **Top 20% rủi ro cao nhất** | `8,697` khách hàng | **413** / 570 ca | **`4.75%`** | **`72.46%`** | **`3.62x`** |

* **Chỉ số Tổng Thể:**
  * **Test ROC-AUC:** **`0.8585`**
  * **OOF Walk-Forward ROC-AUC:** **`0.8889`**
  * **OOF Precision@Top 5%:** **`71.43%`** (Lift: **`2.85x`**)
  * **OOF Recall:** **`91.12%`**

---

## 💻 3. Tổng Hợp Các Tùy Chọn CLI Mới:

```bash
# 1. Chạy Time-Series với đầy đủ 3 thành phần trọng số + 5-Fold Walk-Forward CV:
python -m src.train --dataset timeseries --model xgboost --decay_half_life 12 --customer_weight_power 0.5 --use_usage_weight --metric precision_at_5 --n_trials 30 --cv 5

# 2. Chạy Time-Series chỉ với cân bằng khách hàng (alpha = 0.5):
python -m src.train --dataset timeseries --model xgboost --customer_weight_power 0.5 --metric precision_at_5 --n_trials 30 --cv 5

# 3. Chạy Time-Series tối ưu Recall@Top 10%:
python -m src.train --dataset timeseries --model xgboost --decay_half_life 12 --customer_weight_power 0.5 --metric recall_at_10 --n_trials 30 --cv 5

# 4. Chạy trên bộ dữ liệu Team cấp (churn_train + churn_val):
python -m src.train --dataset latest --model xgboost --decay_half_life 12 --customer_weight_power 0.5 --metric precision_at_5 --n_trials 30 --cv 5
```
