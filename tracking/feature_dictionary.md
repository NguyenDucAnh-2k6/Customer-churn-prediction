# 📚 TỪ ĐIỂN VÀ Ý NGHĨA TOÀN BỘ ĐẶC TRƯNG (FEATURE DICTIONARY & BUSINESS MEANINGS)
**Dự án:** Dự Báo Khách Hàng Rời Bỏ (Customer Churn Prediction)  
**Tác giả:** ML / Data Science Team  
**Cập nhật lần cuối:** 2026-08-27  

---

## 🧭 1. Tổng Quan Phân Loại Đặc Trưng (Feature Taxonomy)

Toàn bộ hệ thống dữ liệu Churn Prediction được cấu trúc thành **5 nhóm đặc trưng chính** và **1 nhóm biến mục tiêu (Targets)**:

```
                                  HỆ THỐNG ĐẶC TRƯNG CHURN PREDICTION
                                                   │
  ┌──────────────────────┬─────────────────────────┼─────────────────────────┬──────────────────────┐
  ▼                      ▼                         ▼                         ▼                      ▼
[Nhóm 1]               [Nhóm 2]                  [Nhóm 3]                  [Nhóm 4]               [Nhóm 5]
Hành Vi Động           Tài Khoản & Gói           Customer 360              Chỉ Số Vận Tốc         Cờ Khuyết Thiếu
(Time-Series Activity) (Account / Demographics)  (Lifetime Macro)          (Velocity & Ratios)    (Missing Indicators)
```

---

## 📈 2. Nhóm 1: Đặc Trưng Hành Vi Động & Tương Tác Chuỗi Thời Gian (Time-Series Activity)

Nhóm đặc trưng cốt lõi phản ánh mức độ tương tác, thói quen và sự suy giảm hoạt động theo từng tháng snapshot.

| Tên Đặc Trưng (Feature Name) | Kiểu Dữ Liệu | Công Thức / Nguồn Tính | Ý Nghĩa Nghiệp Vụ & Dấu Hiệu Churn |
| :--- | :---: | :--- | :--- |
| `num_usage_events_30d` | `int / float` | Số lượng sự kiện sử dụng app/web trong 30 ngày gần nhất | Mức độ tương tác ngắn hạn. Nếu giảm mạnh $\implies$ Tín hiệu cảnh báo sớm nguy cơ Churn. |
| `num_usage_events_30d_lag1m` | `int / float` | Số sự kiện sử dụng ở tháng liền trước (tháng $t-1$) | Điểm tựa để so sánh đà tăng/giảm giữa 2 tháng liên tiếp. |
| `num_usage_events_60d` | `int / float` | Tổng sự kiện sử dụng trong 60 ngày | Mức độ tương tác trung hạn của khách hàng. |
| `num_usage_events_roll3m_sum` | `int / float` | Tổng sự kiện sử dụng lũy kế trong 3 tháng gần nhất | Thể hiện "sức nặng thói quen" của khách hàng trong quý. |
| `total_active_days_30d` | `int` | Số ngày duy nhất có phát sinh hoạt động trong 30 ngày | Tần suất xuất hiện (Frequency). Dù 1 ngày dùng 100 lần nhưng chỉ active 1 ngày thì rủi ro cao hơn người active đều 20 ngày. |
| `total_active_days_60d` | `int` | Số ngày duy nhất có hoạt động trong 60 ngày | Độ bền vững của thói quen sử dụng trong 2 tháng. |
| `total_active_days_90d` | `int` | Số ngày duy nhất có hoạt động trong 90 ngày | Mức độ gắn kết trung thành trong cả quý. |
| `avg_session_duration_30d` | `float` | Tổng thời lượng (giây) / Số session trong 30 ngày | Độ sâu tương tác (Engagement Depth). Thời lượng ngắn dần báo hiệu khách hàng đang mất dần hứng thú. |
| `avg_session_duration_roll3m_mean` | `float` | Trung bình thời lượng phiên của 3 tháng gần nhất | Mức thời lượng cơ sở (Baseline) của khách hàng để làm mốc đối chiếu. |
| `total_session_time_30d` | `float` | Tổng thời gian sử dụng app/web trong 30 ngày (giây) | Tổng thời lượng tương tác thực tế với sản phẩm. |
| `event_type_diversity_30d` | `int` | Số lượng loại sự kiện (Event Types) khác nhau trong 30 ngày | **Độ đa dạng tính năng**. Khách hàng sắp rời bỏ thường chỉ dùng 1-2 tính năng cơ bản trước khi ngừng hẳn. |
| `session_duration_trend` | `float` | Hệ số góc (Slope) thời lượng phiên theo thời gian | Nếu âm ($<0$) $\implies$ Phiên dùng ngày càng ngắn lại. |
| `days_since_last_usage_event` | `float` | Số ngày tính từ lần cuối cùng dùng sản phẩm đến snapshot | **Số ngày im lặng**. Càng lớn $\implies$ Khách hàng đã "chết lâm sàng". |
| `days_since_last_login` | `float` | Số ngày tính từ lần đăng nhập cuối cùng | Khoảng cách đăng nhập. Khách không login quá 30 ngày có xác suất Churn cực cao. |
| `days_since_last_activity` | `float` | Số ngày tính từ bất kỳ hành vi nào (click, login, xem) | Chỉ số đo độ "tươi" (Recency) tổng quát của tài khoản. |
| `days_since_last_activity_lag1m` | `float` | Số ngày không hoạt động tính tại snapshot tháng trước | Dùng để so sánh xem khoảng thời gian im lặng đang kéo dài thêm hay được rút ngắn. |
| `activity_slope_3m` | `float` | Độ dốc tuyến tính (Linear Trend) số lượt dùng qua 3 tháng | **Gia tốc suy giảm**. Giá trị âm càng sâu chứng tỏ khách hàng đang tụt giảm tần suất rất nhanh. |
| `usage_trend_30d` | `float` | Tỷ lệ tăng/giảm sử dụng trong 30 ngày gần nhất | Xu hướng tương tác vi mô trong tháng hiện tại. |
| `is_declining_engagement` | `int (0/1)` | Cờ đánh dấu hoạt động giảm liên tục 2 tháng | Cảnh báo tài khoản rơi vào vùng "nguy cơ suy thoái". |
| `reactivation_flag` | `int (0/1)` | Cờ đánh dấu tài khoản vừa quay lại sau $>60$ ngày im lặng | Nhận diện khách hàng "thức giấc" (cần chăm sóc để tránh Churn lại). |

---

## 🛒 3. Nhóm 2: Đơn Hàng, Doanh Thu & Thanh Toán (Orders, Revenue & Payments)

Phản ánh giá trị tiền tệ (Monetary) và các sự cố thanh toán (nguyên nhân hàng đầu gây Churn ngoài ý muốn).

| Tên Đặc Trưng | Kiểu Dữ Liệu | Nguồn Tính | Ý Nghĩa Nghiệp Vụ & Dấu Hiệu Churn |
| :--- | :---: | :--- | :--- |
| `orders_last_30d` | `int` | Số đơn hàng hoàn tất trong 30 ngày | Sức mua ngắn hạn của khách hàng. |
| `orders_last_90d` | `int` | Số đơn hàng hoàn tất trong 90 ngày | Tần suất mua sắm trong quý. |
| `orders_roll3m_sum` | `int` | Lũy kế đơn hàng 3 tháng gần nhất | Khối lượng giao dịch quý (tương quan cao với `orders_last_90d`). |
| `days_since_last_order` | `float` | Số ngày từ đơn hàng thành công gần nhất đến snapshot | Độ trễ mua sắm. Khách lâu không mua hàng là dấu hiệu chuẩn bị rời nền tảng. |
| `avg_spend_to_date_per_month` | `float` | Tổng chi tiêu / Số tháng hoạt động | Mức chi tiêu bình quân tháng của khách hàng (Giá trị khách hàng). |
| `payments_success_rate` | `float` | Giao dịch thành công / Tổng số giao dịch thanh toán | **Tỷ lệ thành công khi trả tiền**. Nếu thấp $\implies$ Lỗi thẻ, hết hạn thẻ (Involuntary Churn). |

---

## 🎫 4. Nhóm 3: Chăm Sóc Khách Hàng & Marketing (Support & Marketing)

Phản ánh sự hài lòng, các vướng mắc chưa được giải quyết và phản hồi với chiến dịch tiếp thị.

| Tên Đặc Trưng | Kiểu Dữ Liệu | Nguồn Tính | Ý Nghĩa Nghiệp Vụ & Dấu Hiệu Churn |
| :--- | :---: | :--- | :--- |
| `num_tickets_90d` | `int` | Số lượng ticket hỗ trợ tạo ra trong 90 ngày | Khách gặp nhiều sự cố kỹ thuật thường có tỷ lệ thất vọng và bỏ đi cao. |
| `avg_csat_score` | `float` | Điểm hài lòng trung bình (Customer Satisfaction, 1 - 5 sao) | Điểm $< 3.0$ là dấu hiệu rõ ràng của khách hàng bất mãn. |
| `has_unresolved_ticket` | `int (0/1)` | Cờ có ticket đang ở trạng thái 'Open' / 'Pending' tại snapshot | **Sự cố tồn đọng**. Khách hàng đang bực mình vì chưa được xử lý vấn đề. |
| `open_rate_30d` | `float` | Lượt mở email / Lượt email nhận trong 30 ngày | Mức độ chú ý đến thương hiệu qua kênh Email/Notification. |
| `has_marketing_click_30d` | `int (0/1)` | Cờ khách có bấm vào link khuyến mãi/email trong 30 ngày | Khách còn bấm link nghĩa là vẫn còn quan tâm đến sản phẩm. |

---

## 👤 5. Nhóm 4: Tài Khoản, Gói Dịch Vụ & Nhân Khẩu Học (Account & Demographics)

Nhóm các thuộc tính tĩnh hoặc bán tĩnh mô tả danh tính và gói thuê bao của khách hàng.

| Tên Đặc Trưng | Kiểu Dữ Liệu | Giá Trị / Encoding | Ý Nghĩa Nghiệp Vụ & Lưu Ý Kỹ Thuật |
| :--- | :---: | :--- | :--- |
| `tenure_days` | `float` | Số ngày kể từ ngày tạo tài khoản | Tuổi đời khách hàng. Khách mới ($< 90$ ngày) thường có tỷ lệ Churn cao hơn khách lâu năm. |
| `is_paid_tier` | `int (0/1)` | `1`: Trả phí (Plus, Premium), `0`: Miễn phí (Free) | Phân loại loại hình tài khoản. *(Cần cẩn trọng vì có thể gây Shortcut Bias nếu không kiểm soát)*. |
| `subscription_tier` | `int` | `0`: Free, `1`: Plus, `2`: Premium | Cấp bậc gói dịch vụ của khách hàng. |
| `auto_renew` | `int` | `1`: Tự gia hạn, `0`: Thủ công, `-1`: Không có gói | **Cờ tự động gia hạn**. Khách tắt auto-renew là tín hiệu 90% sẽ hủy gói khi hết hạn. |
| `gender` | `int / str` | `0`: Nam, `1`: Nữ | Giới tính khách hàng (MI score gần như bằng 0). |
| `region` / `city` | `category` | Vùng miền / Tỉnh thành | Địa lý khách hàng (Ít ảnh hưởng trực tiếp đến hành vi Churn). |
| `age` | `float` | Độ tuổi khách hàng | Tuổi khách hàng. |

---

## 🌐 6. Nhóm 5: Customer 360 Trọn Đời (Lifetime Macro Features - `churn_ml_dataset.csv`)

Tổng hợp toàn bộ lịch sử từ ngày đầu tiên đến hiện tại (chỉ áp dụng trong `--dataset latest` hoặc phân tích Macro).

| Tên Đặc Trưng | Kiểu Dữ Liệu | Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :--- |
| `total_orders` / `completed_orders` | `int` | Tổng số đơn hàng tích lũy cả đời khách hàng |
| `returned_orders` / `cancelled_orders` | `int` | Số đơn hàng bị hoàn trả hoặc hủy bỏ (chỉ số đo rủi ro trải nghiệm xấu) |
| `total_spent` / `avg_order_value` | `float` | Tổng giá trị chi tiêu trọn đời (LTV) và giá trị trung bình 1 đơn hàng |
| `total_items_purchased` | `int` | Tổng số lượng mặt hàng đã mua |
| `distinct_products_bought` | `int` | Số mã sản phẩm khác nhau từng mua |
| `total_payments` / `failed_payments` | `int` | Tổng số lượt thanh toán và số lần thanh toán thất bại cả đời |
| `total_support_tickets` / `urgent_tickets` | `int` | Tổng số lần yêu cầu hỗ trợ và số lần yêu cầu khẩn cấp |
| `total_usage_sessions` / `total_usage_seconds` | `float` | Tổng số phiên và tổng thời gian sử dụng tích lũy trọn đời |
| `mkt_total_interactions` | `int` | Tổng số lần tiếp xúc marketing (Email, Push, SMS) |
| `mkt_open_rate` / `mkt_click_rate` / `mkt_conversion_rate` | `float` | Tỷ lệ mở, bấm và chuyển đổi chiến dịch marketing trọn đời |

---

## ⚡ 7. Nhóm 6: Chỉ Số Vận Tốc & Tỷ Số Động Học (Hybrid Velocity & Acceleration Ratios)

Các đặc trưng do pipeline tự động sinh (`src/features/preprocessor.py`) để đo lường tốc độ suy giảm hành vi.

| Tên Đặc Trưng | Công Thức Toán Học | Ý Nghĩa Nghiệp Vụ |
| :--- | :--- | :--- |
| `usage_drop_ratio_3m` | $\frac{\text{num\_usage\_events\_30d}}{\text{num\_usage\_events\_roll3m\_sum} / 3.0 + 1.0}$ | **Tỷ số sụt giảm lượng dùng**. Nếu $< 0.3 \implies$ Tháng này chỉ dùng bằng $30\%$ mức trung bình quý $\to$ Báo động đỏ. |
| `session_duration_drop_ratio_3m` | $\frac{\text{avg\_session\_duration\_30d}}{\text{avg\_session\_duration\_roll3m\_mean} + 1.0}$ | **Tỷ số biến động thời lượng**. Phiên dùng ngắn lại so với thói quen cũ. |
| `active_days_share_90d` | $\frac{\text{total\_active\_days\_30d}}{\text{total\_active\_days\_90d} + 1.0}$ | **Tỷ trọng ngày hoạt động**. Tháng này chiếm bao nhiêu phần trong quý. |
| `orders_share_90d` | $\frac{\text{orders\_last\_30d}}{\text{orders\_roll3m\_sum} + 1.0}$ | **Tỷ trọng mua sắm gần đây** so với cả quý. |
| `usage_duration_change` | $\text{avg\_session\_duration\_30d} - \text{roll3m\_mean}$ | Độ chênh lệch tuyệt đối về thời gian tương tác (giây). |
| `activity_acceleration` | $\text{activity\_slope\_3m} \times \text{usage\_trend\_30d}$ | **Gia tốc tụt giảm**. Kết hợp cả độ dốc quý và xu hướng tháng. |
| `usage_30d_share_lifetime` | $\frac{\text{num\_usage\_events\_30d}}{\text{total\_usage\_sessions} + 1.0}$ | Đóng góp của tháng gần nhất so với toàn bộ lịch sử tài khoản. |
| `spent_30d_share_lifetime` | $\frac{\text{avg\_spend\_to\_date\_per\_month}}{\text{total\_spent} + 1.0}$ | Mức độ chi tiêu hiện tại so với tổng tài sản tích lũy. |

---

## 🚩 8. Nhóm 7: Cờ Đánh Dấu Khuyết Thiếu (Missing Value Indicators)

| Tên Đặc Trưng | Công Thức | Ý Nghĩa Nghiệp Vụ |
| :--- | :---: | :--- |
| `payments_success_rate_missing` | `payments_success_rate.isna().astype(int)` | Đánh dấu khách hàng **chưa từng có giao dịch thanh toán nào** (thường là Free User chưa mua hàng). |
| `session_duration_trend_missing` | `session_duration_trend.isna().astype(int)` | Khách dùng quá ít phiên để tính toán được hệ số góc (Slope). |
| `avg_csat_score_missing` | `avg_csat_score.isna().astype(int)` | Khách hàng **chưa từng gửi ticket hoặc không đánh giá sao**. |

---

## 🎯 9. Biến Mục Tiêu (Target Variables Matrix)

| Tên Nhãn Target | Định Nghĩa Nghiệp Vụ | Loại Bài Toán & Phạm Vi |
| :--- | :--- | :--- |
| **`label_churn` / `churn_30d`** | Khách hàng đóng tài khoản HOẶC không có hoạt động trong 30 ngày tiếp theo | **Primary Target** (Dự báo Churn tổng thể trong 30 ngày tới). |
| `churn_60d` | Khách hàng rời bỏ / không hoạt động trong 60 ngày tiếp theo | Phục vụ dự báo dài hạn (Long-term Churn). |
| `churn_case1_30d` | Khách hàng chủ động hủy gói trả phí (Paid Cancellation) | Phục vụ bài toán chuyên biệt cho nhóm Paid Subscription. |
| `churn_case2_30d` | Khách hàng ngừng hoạt động (Inactivity Churn) | Áp dụng cho nhóm ngừng tương tác nói chung. |
| `churn_case2a_30d` | Khách hàng Paid nhưng không tương tác | Nguy cơ lãng phí gói và sắp hủy gói. |
| `churn_case2b_30d` | Khách hàng Free không tương tác | Nguy cơ mất người dùng miễn phí (E-commerce / App Inactive). |

---

## 🧭 10. Khuyến Nghị Phân Loại Cho Các Chiến Lược ML

| Nhóm Đặc Trưng | Chiến Lược Toàn Diện (Full Model) | 🚀 Chiến Lược 4 (`--behavioral_only`) |
| :--- | :---: | :---: |
| **Hành vi động (Usage, Recency, CSAT)** | ✅ Giữ lại (Trọng tâm) | ✅ **Giữ lại 100% (Trọng tâm cốt lõi)** |
| **Chỉ số Vận tốc (Velocity & Drop Ratios)** | ✅ Giữ lại | ✅ **Giữ lại 100%** |
| **Cờ khuyết thiếu (Missing Indicators)** | ✅ Giữ lại | ✅ **Giữ lại 100%** |
| **Tài khoản & Gói (`is_paid_tier`, `tier`)** | ✅ Giữ lại | ❌ **Loại bỏ (Để tránh Shortcut Bias & Shift)** |
| **Nhân khẩu học (`gender`, `region`, `age`)** | ❌ Loại bỏ qua MI Filter | ❌ **Loại bỏ** |
