# Customer Churn Dataset Join & Ground Truth Walkthrough

Chúng ta đã hoàn thành việc xác định **Ground Truth**, xây dựng pipeline làm sạch và thực hiện phép **SQL JOIN / Feature Aggregation** tự động từ 10 file CSV ban đầu để tạo ra bộ dataset huấn luyện ML hoàn chỉnh [`data/churn_ml_dataset.csv`](file:///d:/ML_intern/data/churn_ml_dataset.csv).

---

## 🎯 1. Vị trí và định nghĩa Ground Truth

Ground Truth được trích xuất từ cột `account_status` trong [`data/churn_customers.csv`](file:///d:/ML_intern/data/churn_customers.csv):

| Trạng thái (`account_status`) | Định nghĩa Churn | Nhãn `is_churn` | Số lượng | Tỷ lệ (%) |
| :--- | :--- | :---: | :---: | :---: |
| `Closed` | Khách hàng đã đóng/hủy tài khoản | **1** | 692 | **6.92%** |
| `Active` | Khách hàng đang hoạt động | **0** | 9,312 | **93.06%** |
| `Inactive` | Tạm ngưng hoạt động | **0** | 2 | **0.02%** |

---

## 🛠️ 2. Các thay đổi và công việc đã hoàn thành

### Script xử lý dữ liệu tự động: [`build_dataset.py`](file:///d:/ML_intern/build_dataset.py)

Script đã thực hiện các công đoạn:
1. **Khử trùng lặp (Deduplication):**
   * `churn_customers.csv`: Loại bỏ 820,164 dòng lặp $\rightarrow$ Giữ lại **10,006 khách hàng duy nhất**.
   * `churn_orders.csv`: Loại bỏ 9,003 dòng lặp $\rightarrow$ Giữ lại **3,001 đơn hàng**.
   * `churn_product_usage.csv`: Loại bỏ 3,006 dòng lặp $\rightarrow$ Giữ lại **1,003 bản ghi log**.
   * `churn_products.csv`: Loại bỏ 301 dòng lặp $\rightarrow$ Giữ lại **601 sản phẩm**.
2. **Xử lý lỗi Timestamp/Date Anomalies:**
   * Ép kiểu thời gian an toàn cho các ngày bị lỗi (ví dụ năm `20266`) bằng `pd.to_datetime(..., errors='coerce')`.
3. **Tính toán đặc trưng khách hàng (Feature Engineering):**
   * Tính `tenure_days` (Số ngày gắn bó), `age` (Tuổi khách hàng), `days_since_last_login`.
4. **SQL-like JOIN & Feature Aggregations:**
   * Tổng hợp dữ liệu giao dịch từ 8 bảng phụ (`orders`, `order_items`, `products`, `payments`, `subscriptions`, `tickets`, `usage`, `marketing`) theo `customer_id`.
   * Thực hiện **SQL LEFT JOIN** ghép tất cả tính năng vào bảng khách hàng master.
   * Xử lý Imputation điền giá trị `0` cho số đếm/tổng tiền và `'None'` cho biến danh mục đối với khách hàng không có giao dịch tương ứng.

---

## 📊 3. Kết quả bộ Dataset hoàn chỉnh ([`churn_ml_dataset.csv`](file:///d:/ML_intern/data/churn_ml_dataset.csv))

* **Số lượng mẫu (Rows):** `10,006` khách hàng duy nhất (**1 dòng = 1 khách hàng**).
* **Số lượng đặc trưng (Columns):** `52` cột (Bao gồm `customer_id`, `is_churn` và 50 features).

### Các nhóm cột chính:
* **Định danh & Nhãn:** `customer_id`, `is_churn`, `account_status`
* **Thông tin cá nhân & Nhân khẩu học:** `gender`, `city`, `region`, `age`, `tenure_days`, `days_since_last_login`, `province`, `crm_channel`, `has_national_id`, `has_phone`
* **Đơn hàng & Mua sắm:** `total_orders`, `completed_orders`, `returned_orders`, `cancelled_orders`, `total_spent`, `avg_order_value`, `max_order_value`, `days_since_last_order`, `total_items_purchased`, `distinct_products_bought`, `distinct_categories_bought`, `top_category`
* **Thanh toán:** `total_payments`, `successful_payments`, `failed_payments`, `total_payment_amount`, `primary_payment_method`
* **Gói Dịch Vụ Subscription:** `has_subscription`, `sub_status`, `sub_plan_tier`, `sub_auto_renew`, `sub_change_type`
* **Chăm sóc khách hàng (Tickets):** `total_support_tickets`, `avg_csat_score`, `avg_ticket_resolution_hours`, `urgent_tickets`, `account_tickets`
* **Tương tác App/Web (Usage):** `total_usage_sessions`, `total_usage_seconds`, `avg_session_seconds`, `primary_usage_device`
* **Marketing:** `mkt_total_interactions`, `mkt_opened_count`, `mkt_clicked_count`, `mkt_converted_count`, `mkt_open_rate`, `mkt_click_rate`, `mkt_conversion_rate`

---

## ✅ 4. Kiểm tra & Xác minh (Verification)

* **Script Execution:**
  ```bash
  python build_dataset.py
  ```
  Thành công (Exit code 0).
* **Kiểm tra File Dataset:** File được tạo thành công tại [`data/churn_ml_dataset.csv`](file:///d:/ML_intern/data/churn_ml_dataset.csv) với dung lượng ~1.7 MB, đảm bảo không có rò rỉ dữ liệu (data leakage) hoặc lặp dòng `customer_id`.
