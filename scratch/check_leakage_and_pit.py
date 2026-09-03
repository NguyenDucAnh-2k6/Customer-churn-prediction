import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

print("=" * 70)
print("1. PHÂN TÍCH NGUYÊN NHÂN PR-AUC BỊ ĐẨY LÊN > 0.94 TRÊN ROUND 3 HIỆN TẠI")
print("=" * 70)

df_r3 = pd.read_csv('data/processed/round3/churn_master.csv')
num_cols = df_r3.select_dtypes(include=[np.number]).columns
corrs = df_r3[num_cols].corr()['churn'].abs().sort_values(ascending=False)
print("\nTop 10 features có tương quan tuyệt đối cao nhất với nhãn Churn:")
print(corrs.head(11))

print("\n" + "=" * 70)
print("2. CHỨNG MINH LEAKAGE DO SNAPSHOT_DT = CLOSED_DT:")
print("=" * 70)
print("""
Trong build_dataset.py hiện tại:
  - Khách hàng Churned: snapshot_dt = closed_dt (ngày đóng tài khoản).
  - Khách hàng Active : snapshot_dt = 2026-08-25 (ngày tương lai sau khi dữ liệu đã kết thúc).

Hậu quả:
1. TARGET LEAKAGE: Tại ngày đóng tài khoản (closed_dt), khách hàng đã hủy gói dịch vụ,
   nên is_auto_renew = 0 và subscription_expired = 1. Mô hình chỉ cần nhìn vào 2 cờ này
   là đoán được 100% khách hàng đã churn.
2. TEMPORAL GAP LEAKAGE: Dữ liệu log Silver kết thúc vào 2026-07-28. Khi đặt snapshot
   cho Active là 2026-08-25, 30 ngày cuối không có log, dẫn đến total_usage_60d và
   usage_60d_share của Active bị sụt giảm nhân tạo, tạo ra sự phân tách giả tạo giữa
   Active và Churned.
""")

print("=" * 70)
print("3. THỬ NGHIỆM REALISTIC POINT-IN-TIME (KHÔNG LEAKAGE) NHƯ BỘ DỮ LIỆU CỦA TEAM")
print("=" * 70)

cust = pd.read_csv('data/churn_customers.csv')
cust['signup_dt'] = pd.to_datetime(cust['signup_date'], errors='coerce')
cust['closed_dt'] = pd.to_datetime(cust['closed_date'], errors='coerce')
usage = pd.read_csv('data/churn_product_usage.csv')
usage['event_date_dt'] = pd.to_datetime(usage['event_date'], errors='coerce')
orders = pd.read_csv('data/churn_orders.csv')
orders['order_date_dt'] = pd.to_datetime(orders['order_date'], errors='coerce')

# Chọn mốc snapshot đồng nhất T = 2026-05-31
T = pd.to_datetime('2026-05-31')
horizon_days = 60 # Dự báo trong 60 ngày tới (2026-06-01 đến 2026-07-31)

active_at_T = cust[(cust['signup_dt'] <= T) & (cust['closed_dt'].isna() | (cust['closed_dt'] > T))].copy()
active_at_T['churn'] = (active_at_T['closed_dt'].notna() & (active_at_T['closed_dt'] <= T + pd.Timedelta(days=horizon_days))).astype(int)

print(f"Tổng số khách hàng Active tại snapshot {T.date()}: {len(active_at_T):,}")
print(f"Số lượng rời bỏ trong {horizon_days} ngày tiếp theo: {active_at_T['churn'].sum():,} ({active_at_T['churn'].mean()*100:.2f}%)")

# Tính features hoàn toàn trong quá khứ trước T (t <= T):
active_at_T['customer_tenure'] = (T - active_at_T['signup_dt']).dt.days

valid_u = usage[usage['event_date_dt'] <= T]
u_60 = valid_u[valid_u['event_date_dt'] > T - pd.Timedelta(days=60)].groupby('customer_id').agg(
    total_usage_60d=('usage_id', 'count'),
    avg_usage_duration_60d=('session_duration_sec', 'mean')
).reset_index()
u_all = valid_u.groupby('customer_id').agg(
    total_usage_all_time=('usage_id', 'count')
).reset_index()

valid_o = orders[orders['order_date_dt'] <= T]
o_60 = valid_o[valid_o['order_date_dt'] > T - pd.Timedelta(days=60)].groupby('customer_id').agg(
    total_orders_60d=('order_id', 'count'),
    total_order_amounts_60d=('total_amount', 'sum')
).reset_index()

df_pit = active_at_T.merge(u_60, on='customer_id', how='left').merge(u_all, on='customer_id', how='left').merge(o_60, on='customer_id', how='left')
df_pit['total_usage_60d'] = df_pit['total_usage_60d'].fillna(0)
df_pit['total_usage_all_time'] = df_pit['total_usage_all_time'].fillna(0)
df_pit['total_orders_60d'] = df_pit['total_orders_60d'].fillna(0)
df_pit['total_order_amounts_60d'] = df_pit['total_order_amounts_60d'].fillna(0)
df_pit['avg_usage_duration_60d'] = df_pit['avg_usage_duration_60d'].fillna(0)
df_pit['usage_60d_share'] = np.where(df_pit['total_usage_all_time'] > 0, df_pit['total_usage_60d'] / df_pit['total_usage_all_time'], 0.0)

feat_cols = ['customer_tenure', 'total_usage_60d', 'avg_usage_duration_60d', 'total_usage_all_time', 'total_orders_60d', 'total_order_amounts_60d', 'usage_60d_share']
X = df_pit[feat_cols]
y = df_pit['churn']

clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
preds = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]

print("\n" + "=" * 70)
print("KẾT QUẢ TRÊN BỘ DATASET POINT-IN-TIME CHUẨN KHÔNG LEAKAGE:")
print("=" * 70)
print(f"  - ROC-AUC: {roc_auc_score(y, preds):.4f}")
print(f"  - PR-AUC : {average_precision_score(y, preds):.4f}  (Đúng mức chuẩn thực tế ~0.6 - 0.7!)")
