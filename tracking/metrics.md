📌 1. Bản Chất Của --dataset pit (Point-in-Time) Và Tính Hợp Lệ Của Việc Shuffle Trong CV
Bạn có một sự liên tưởng hoàn toàn chính xác về mặt bản chất mô hình hóa:

A. Mô Hình Chuỗi (LSTM / GRU Many-to-One):
   Khách hàng i: [ Timestep 1 ──► Timestep 2 ──► ... ──► Timestep T ] ──► [ Nhãn Churn y_i ]
   👉 Trong 1 chuỗi: KHÔNG ĐƯỢC đảo trật tự thời gian (t1 -> t2 -> ... -> tT).
   👉 Giữa các chuỗi (các KH khác nhau): HOÀN TOÀN ĐƯỢC shuffle vì mỗi KH là 1 thực thể độc lập.
B. Mô Hình Bảng Tăng Cường (Point-in-Time Tabular Boosting - XGBoost):
   Thay vì đưa chuỗi thô 3D vào mạng nơ-ron, ta đã trích xuất toàn bộ động thái của chuỗi thành
   các vector đặc trưng động (Dynamic Features):
   * Độ dốc/Xu hướng (Velocity): activity_slope_3m, usage_trend_30d
   * Độ trễ (Lags): num_usage_events_30d_lag1m
   * Cửa sổ trượt (Rolling aggregations): orders_roll3m_sum, avg_session_duration_roll3m_mean
   
   👉 Mỗi khách hàng trở thành 1 vector đặc trưng độc lập x_i với nhãn tương ứng y_i.
   👉 Vì 1 khách hàng chỉ xuất hiện đúng 1 lần (1 item), việc áp dụng Stratified 5-Fold CV 
      có SHUFFLE là 100% hợp lệ về mặt thống kê và KHÔNG HỀ BỊ rò rỉ dữ liệu (Data Leakage)!
⏳ 2. Giải Thích Các Đại Lượng Trong Công Thức Trọng Số Bán Rã (Exponential Decay)
Công thức tính trọng số học cho mẫu $i$ theo thời gian:

$$w_i = 2^{-\frac{T_{\text{max}} - t_i}{\text{half_life_months}}}$$

Chi tiết từng đại lượng:
$t_i$ (Snapshot Month): Mốc thời gian của mẫu dữ liệu $i$ (ví dụ: tháng 2024-05).
$T_{\text{max}}$ (Latest Train Month): Mốc thời gian mới nhất trong tập huấn luyện (ví dụ: tháng 2025-12).
$\Delta t_i = T_{\text{max}} - t_i$ (Age / Khoảng cách thời gian): Độ "cũ" của mẫu dữ liệu tính theo số tháng so với hiện tại. Mẫu càng nằm xa trong quá khứ thì $\Delta t_i$ càng lớn.
$\text{half_life_months}$ (Chu kỳ bán rã $t_{1/2}$): Số tháng mà sau khoảng thời gian đó, mức độ quan trọng (trọng số) của mẫu dữ liệu sẽ bị giảm đi đúng một nửa (50%).
Ví dụ với half_life = 12 tháng:
Mẫu tại $T_{\text{max}}$ ($\Delta t = 0$ tháng): $w = 2^{-0/12} = 2^0 = \mathbf{1.0}$ (100% trọng số)
Mẫu cách đây 12 tháng ($\Delta t = 12$ tháng): $w = 2^{-12/12} = 2^{-1} = \mathbf{0.5}$ (còn 50% trọng số)
Mẫu cách đây 24 tháng ($\Delta t = 24$ tháng): $w = 2^{-24/12} = 2^{-2} = \mathbf{0.25}$ (còn 25% trọng số)
Bước chuẩn hóa ($\tilde{w}_i = \frac{w_i}{\bar{w}}$): Giúp trung bình trọng số luôn bằng $1.0$, tránh làm thay đổi quy mô của hàm mất mát (Loss scale) và Gradient của XGBoost.
🎯 3. Precision@K% Và Recall@K%: K Là Cố Định Hay Tính Kiểu Gì?
$K%$ là tỷ lệ phần trăm phân vị cắt ngưỡng (Top Percentile Cutoff) trên danh sách khách hàng sau khi mô hình đã chấm điểm xác suất rời bỏ ($\hat{p}$) và sắp xếp giảm dần:

$$\text{Số khách hàng được chọn } (n_K) = \text{Tổng số khách hàng } (N) \times K%$$

Ví dụ trên tập Test có $N = 43,485$ khách hàng:
$K = 1%$: Lấy ra $43,485 \times 1% = \mathbf{434}$ khách hàng có điểm rủi ro cao nhất.
$K = 2%$: Lấy ra $43,485 \times 2% = \mathbf{869}$ khách hàng có điểm rủi ro cao nhất.
$K = 5%$: Lấy ra $43,485 \times 5% = \mathbf{2,174}$ khách hàng có điểm rủi ro cao nhất.
$K = 10%$: Lấy ra $43,485 \times 10% = \mathbf{4,348}$ khách hàng có điểm rủi ro cao nhất.
Tại sao dùng tỷ lệ $K%$ thay vì số lượng cố định? Vì quy mô khách hàng mỗi tháng biến động (tháng này 40,000, tháng sau 60,000). Dùng $K%$ giúp doanh nghiệp khống chế chính xác ngân sách chăm sóc khách hàng (ví dụ: "Mỗi tháng đội Marketing chỉ đủ ngân sách gửi quà cho đúng Top 5% khách hàng nguy cơ cao nhất").

📊 4. Giải Thích Bảng Log, Ý Nghĩa Từng Cột & Cách Tính Baseline
Hãy cùng phân tích trực tiếp bảng kết quả thực tế của bạn:

Decile / Segment ($K$)	Targeted Users ($n_K$)	Captured Churns ($c_K / C_{\text{total}}$)	Precision@K ($P_K$)	Recall@K / Coverage ($R_K$)	Cumulative Lift ($\text{Lift}_K$)
Top 1%	434	169 / 570	38.94%	29.65%	29.71x
Top 2%	869	203 / 570	23.36%	35.61%	17.82x
A. Giải thích chi tiết từng cột:
Targeted Users ($n_K = 434$): Số lượng khách hàng thuộc nhóm Top 1% rủi ro cao nhất.
Captured Churns ($169 / 570$):
$c_K = 169$: Số khách hàng thực sự Churn nằm trong nhóm 434 người được chọn này.
$C_{\text{total}} = 570$: Tổng số khách hàng Churn thực tế trên toàn bộ $43,485$ mẫu tập Test.
Precision@K ($P_K = 38.94%$): Tỷ lệ chính xác trong nhóm được chọn: $$P_K = \frac{\text{Số Churners bắt trúng}}{\text{Số người được chọn}} = \frac{169}{434} = \mathbf{38.94%}$$ (Nghĩa là: Cứ nhắm vào 100 người trong Top 1% này thì có tới ~39 người chắc chắn sẽ bỏ dịch vụ!)
Recall@K ($R_K = 29.65%$): Tỷ lệ bao phủ Churn trên toàn hệ thống: $$R_K = \frac{\text{Số Churners bắt trúng}}{\text{Tổng số Churners toàn bộ tập Test}} = \frac{169}{570} = \mathbf{29.65%}$$ (Nghĩa là: Chỉ cần can thiệp đúng 1% lượng khách hàng mà đã giữ chân trước được gần 30% tổng số người có nguy cơ rời bỏ!)
B. Baseline Là Gì Và Được Tính Thế Nào?
Baseline Churn Rate ($p_{\text{base}}$) là tỷ lệ Churn tự nhiên ngẫu nhiên của toàn bộ tập dữ liệu (tương đương việc bốc thăm ngẫu nhiên không cần mô hình):

$$p_{\text{base}} = \frac{\text{Tổng số Churners thực tế}}{\text{Tổng số khách hàng toàn tập}} = \frac{570}{43,485} \approx \mathbf{0.013108 \ (1.3108%)}$$

Nếu không có mô hình: Bạn chọn bừa 434 khách hàng thì theo xác suất ngẫu nhiên chỉ tóm được: $$434 \times 1.3108% \approx \mathbf{5.69 \text{ khách hàng Churn}} \quad (\text{Precision ngẫu nhiên chỉ là } 1.31%)$$
Nhờ có mô hình XGBoost: Bạn tóm trúng được $169$ khách hàng Churn!
C. Công Thức Tính Cumulative Lift:
$$\text{Cumulative Lift}K = \frac{\text{Precision@K}}{p{\text{base}}} = \frac{38.94%}{1.3108%} \approx \mathbf{29.71\text{x}}$$

(Hoặc tính tương đương qua Recall: $\text{Lift}_K = \frac{\text{Recall@K}}{K%} = \frac{29.65%}{1%} \approx 29.65\text{x}$)

💡 Ý nghĩa kinh doanh của con số 29.71x: Mô hình AI giúp việc xác định khách hàng rời bỏ hiệu quả gấp gần 30 LẦN so với việc chọn ngẫu nhiên. Doanh nghiệp tiết kiệm được 30 lần chi phí marketing/telesale so với việc đi tặng voucher đại trà cho toàn bộ khách hàng.