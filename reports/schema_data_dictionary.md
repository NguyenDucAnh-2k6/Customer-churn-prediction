# 📘 Từ Điển Dữ Liệu & Định Nghĩa Schema — Dataset Round 3

> **Mục đích:** Bộ dữ liệu Point-in-Time (1 dòng / 1 khách hàng) dùng để huấn luyện và đánh giá mô hình Dự báo Khách hàng Rời bỏ (Customer Churn Prediction).
> **Đặc điểm nổi bật:**
> 1. **Zero Customer Leakage:** Toàn bộ lịch sử của 1 khách hàng chỉ thuộc về tập Train hoặc Test.
> 2. **Zero Target/Temporal Leakage:** Snapshot quan sát hành vi trước khi churn, loại bỏ hoàn toàn các rò rỉ trạng thái sau khi đóng tài khoản.
> 3. **Chất lượng dữ liệu cao (0% Missing):** Đã loại bỏ 10 cột có tỷ lệ thiếu > 15%, toàn bộ 34 cột còn lại đều sạch 100%.
> 4. **Cân bằng Cross-Validation:** Tích hợp sẵn cột `cv_fold` (0 - 4) với tỷ lệ Churn đồng đều giữa các Folds.

---

## 📊 1. Thống Kê Tổng Quan Bộ Dữ Liệu
- **Tổng số khách hàng (Master):** `10,002` dòng × `34` cột
- **Tập Huấn luyện (Train Set - 80%):** `8,002` dòng × `35` cột (Tỷ lệ Churn: `6.72%`)
- **Tập Kiểm thử (Test Set - 20%):** `2,000` dòng × `34` cột (Tỷ lệ Churn: `7.70%`)
- **Số lượng Folds Cross-Validation:** `5 Folds` (`cv_fold` từ `0` đến `4`)
- **Tỷ lệ giá trị thiếu (Missing Rate):** `0.00%` trên toàn bộ 34 cột

---

## 📋 2. Chi Tiết Các Cột & Định Nghĩa Nghiệp Vụ (Data Dictionary)

### 1. Định danh & Nhân khẩu học

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `customer_id` | `int64` | `churn_customers` | Mã định danh khách hàng duy nhất (Primary Key) |
| `gender` | `str` | `churn_customers` | Giới tính khách hàng (Male, Female, Other) |
| `customer_age` | `float64` | `churn_customers` | Độ tuổi của khách hàng tính đến thời điểm snapshot |
| `customer_tenure` | `int64` | `churn_customers` | Số ngày gắn bó kể từ ngày đăng ký tài khoản đến snapshot |


### 2. Đơn hàng & Chi tiêu

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `total_order_amounts_60d` | `float64` | `churn_orders` | Tổng số tiền chi tiêu mua hàng trong 60 ngày gần nhất (VND) |
| `total_orders_60d` | `float64` | `churn_orders` | Tổng số lượng đơn hàng đặt trong 60 ngày gần nhất |
| `avg_order_amount_60d` | `float64` | `churn_orders` | Giá trị đơn hàng trung bình trong 60 ngày gần nhất (VND) |


### 3. Giao dịch Thanh toán

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `total_payment_amounts_60d` | `float64` | `churn_payments` | Tổng số tiền thanh toán thành công trong 60 ngày gần nhất (VND) |
| `total_payments_60d` | `float64` | `churn_payments` | Tổng số lượt giao dịch thanh toán trong 60 ngày gần nhất |
| `avg_payment_amount_60d` | `float64` | `churn_payments` | Giá trị thanh toán trung bình mỗi giao dịch trong 60 ngày gần nhất (VND) |
| `failed_payment_rate_60d` | `float64` | `churn_payments` | Tỷ lệ giao dịch thanh toán bị thất bại / lỗi thẻ trong 60 ngày gần nhất |


### 4. Gói dịch vụ Thuê bao

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `is_auto_renew` | `float64` | `churn_subscriptions` | Cờ bật tự động gia hạn gói thuê bao (1.0: Có, 0.0: Không) |
| `is_downgrade` | `float64` | `churn_subscriptions` | Cờ xác định có hành vi hạ cấp gói dịch vụ (1.0: Có, 0.0: Không) |
| `plan_tier` | `str` | `churn_subscriptions` | Hạng gói dịch vụ thuê bao hiện tại (Basic, Standard, Premium, None) |
| `subscription_expired` | `int64` | `churn_subscriptions` | Cờ xác định gói thuê bao đã hết hạn tại snapshot (1: Hết hạn, 0: Còn hạn) |


### 5. CSKH & Khiếu nại (CSAT)

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `total_tickets_60d` | `float64` | `churn_support_tickets` | Tổng số lượng phiếu khiếu nại gửi trong 60 ngày gần nhất |
| `missing_csat_rate_60d` | `float64` | `churn_support_tickets` | Tỷ lệ ticket trong 60 ngày không để lại đánh giá CSAT (0.0 - 1.0) |


### 6. Tiếp thị Marketing

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `total_interactions_all_time` | `float64` | `churn_marketing_interactions` | Tổng số thông điệp tiếp thị khách hàng nhận được toàn thời gian |
| `opened_rate_all_time` | `float64` | `churn_marketing_interactions` | Tỷ lệ mở thông điệp tiếp thị toàn thời gian (opened / total) |
| `clicked_rate_all_time` | `float64` | `churn_marketing_interactions` | Tỷ lệ nhấp link tiếp thị toàn thời gian (clicked / total) |
| `converted_rate_all_time` | `float64` | `churn_marketing_interactions` | Tỷ lệ chuyển đổi mua hàng từ tiếp thị toàn thời gian (converted / total) |
| `total_interactions_60d` | `float64` | `churn_marketing_interactions` | Số thông điệp tiếp thị nhận được trong 60 ngày gần nhất |
| `opened_rate_60d` | `float64` | `churn_marketing_interactions` | Tỷ lệ mở thông điệp tiếp thị trong 60 ngày gần nhất |
| `clicked_rate_60d` | `float64` | `churn_marketing_interactions` | Tỷ lệ nhấp link tiếp thị trong 60 ngày gần nhất |
| `converted_rate_60d` | `float64` | `churn_marketing_interactions` | Tỷ lệ chuyển đổi mua hàng tiếp thị trong 60 ngày gần nhất |
| `opened_rate_change` | `float64` | `Derived` | Mức độ thay đổi tỷ lệ mở mail (opened_rate_60d - opened_rate_all_time) |
| `clicked_rate_change` | `float64` | `Derived` | Mức độ thay đổi tỷ lệ click link (clicked_rate_60d - clicked_rate_all_time) |
| `converted_rate_change` | `float64` | `Derived` | Mức độ thay đổi tỷ lệ chuyển đổi (converted_rate_60d - converted_rate_all_time) |
| `interaction_60d_share` | `float64` | `Derived` | Tỷ trọng tương tác 60 ngày so với toàn thời gian (total_60d / total_all_time) |


### 7. Sử dụng Ứng dụng (App Usage)

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `total_usage_all_time` | `float64` | `churn_product_usage` | Tổng số phiên truy cập ứng dụng toàn thời gian |
| `total_usage_60d` | `float64` | `churn_product_usage` | Tổng số phiên truy cập ứng dụng trong 60 ngày gần nhất |
| `usage_60d_share` | `float64` | `Derived` | Tỷ trọng số phiên dùng app 60 ngày so với toàn thời gian |
| `usage_duration_change` | `float64` | `Derived` | Mức độ thay đổi thời lượng phiên (duration_60d - duration_all_time, giây) |


### 8. Nhãn mục tiêu (Target)

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `churn` | `int64` | `churn_customers` | Nhãn mục tiêu (Ground Truth): 1 nếu Rời bỏ (Closed), 0 nếu Hoạt động (Active) |


### 9. Phân đoạn Cross-Validation

| Tên Cột (Feature Name) | Kiểu Dữ Liệu | Bảng Nguồn (Silver Layer) | Mô Tả Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :---: | :--- |
| `cv_fold` | `int64` | `StratifiedGroupKFold (k=5)` | Mã Fold (0 - 4) định danh tập Validation trong 5-Fold Cross-Validation (Chỉ có trong churn_train.csv) |


---

## 🎯 3. Phân Phối Tỷ Lệ Churn Trên Từng Fold (Train Set)

| Fold ID | Số Lượng Khách Hàng | Số Lượng Churn (Positive) | Tỷ Lệ Churn (%) |
| :---: | :---: | :---: | :---: |
| **Fold 0** | `1,600` | `121` | `7.56%` |
| **Fold 1** | `1,600` | `108` | `6.75%` |
| **Fold 2** | `1,601` | `102` | `6.37%` |
| **Fold 3** | `1,601` | `113` | `7.06%` |
| **Fold 4** | `1,600` | `94` | `5.88%` |
| **Test Set** | `2,000` | `154` | `7.70%` |