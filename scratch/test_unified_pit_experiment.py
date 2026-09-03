import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

# Load silver tables
cust = pd.read_csv('data/churn_customers.csv')
cust['signup_dt'] = pd.to_datetime(cust['signup_date'], errors='coerce')
cust['closed_dt'] = pd.to_datetime(cust['closed_date'], errors='coerce')
cust['birth_dt'] = pd.to_datetime(cust['birth_date'], errors='coerce')
orders = pd.read_csv('data/churn_orders.csv')
orders['order_date_dt'] = pd.to_datetime(orders['order_date'], errors='coerce')
payments = pd.read_csv('data/churn_payments.csv')
payments['payment_date_dt'] = pd.to_datetime(payments['payment_date'], errors='coerce')
usage = pd.read_csv('data/churn_product_usage.csv')
usage['event_date_dt'] = pd.to_datetime(usage['event_date'], errors='coerce')
subs = pd.read_csv('data/churn_subscriptions.csv')
subs['created_at_dt'] = pd.to_datetime(subs['start_date'] if 'start_date' in subs.columns else subs['created_at'], errors='coerce')
subs['end_date_dt'] = pd.to_datetime(subs['end_date'], errors='coerce')
mkt = pd.read_csv('data/churn_marketing_interactions.csv')
mkt['sent_at_dt'] = pd.to_datetime(mkt['sent_at'], errors='coerce')
tickets = pd.read_csv('data/churn_support_tickets.csv')
tickets['created_at_dt'] = pd.to_datetime(tickets['created_at'], errors='coerce')

# Single Unified Observation Snapshot Date: T = 2026-05-31
# Target Prediction Horizon: 60 days (up to 2026-07-31)
T = pd.to_datetime('2026-05-31')
horizon = pd.Timedelta(days=60)

# Filter cohort: customers who were active at T
cohort = cust[(cust['signup_dt'] <= T) & (cust['closed_dt'].isna() | (cust['closed_dt'] > T))].copy()
cohort['churn'] = (cohort['closed_dt'].notna() & (cohort['closed_dt'] <= T + horizon)).astype(int)
n_churn = cohort['churn'].sum()
print(f'Cohort size at T ({T.date()}): {len(cohort):,} customers')
print(f'Churn rate in next 60d: {cohort["churn"].mean()*100:.2f}% ({n_churn}/{len(cohort)})')

# Features strictly at or before T:
cohort['customer_age'] = np.where(cohort['birth_dt'].notna(), (T - cohort['birth_dt']).dt.days // 365, 35.0)
cohort['customer_tenure'] = (T - cohort['signup_dt']).dt.days.fillna(0).astype(int)

# Orders
val_ord = orders[orders['order_date_dt'] <= T].copy()
val_ord['days_before'] = (T - val_ord['order_date_dt']).dt.days
ord_60 = val_ord[val_ord['days_before'] <= 60].groupby('customer_id').agg(
    total_order_amounts_60d=('total_amount', 'sum'),
    total_orders_60d=('order_id', 'count'),
    avg_order_amount_60d=('total_amount', 'mean')
).reset_index()

# Payments
val_pay = payments[payments['payment_date_dt'] <= T].copy()
val_pay['days_before'] = (T - val_pay['payment_date_dt']).dt.days
pay_60 = val_pay[val_pay['days_before'] <= 60].groupby('customer_id').agg(
    total_payment_amounts_60d=('amount', 'sum'),
    total_payments_60d=('payment_id', 'count'),
    avg_payment_amount_60d=('amount', 'mean'),
    failed_payments_60d=('status', lambda x: (x == 'Failed').sum())
).reset_index()
pay_60['failed_payment_rate_60d'] = np.where(pay_60['total_payments_60d'] > 0, pay_60['failed_payments_60d'] / pay_60['total_payments_60d'], 0.0)
pay_60 = pay_60.drop(columns=['failed_payments_60d'])

# Subscriptions at T
val_subs = subs[subs['created_at_dt'] <= T].sort_values('created_at_dt').groupby('customer_id').last().reset_index()
val_subs['is_auto_renew'] = val_subs['auto_renew'].fillna(0).astype(float)
val_subs['is_downgrade'] = (val_subs['change_type'] == 'Downgrade').astype(float)
val_subs['subscription_expired'] = ((val_subs['end_date_dt'] < T) | (val_subs['status'].isin(['Expired', 'Cancelled']))).astype(int)
sub_feats = val_subs[['customer_id', 'plan_tier', 'is_auto_renew', 'is_downgrade', 'subscription_expired']]

# Usage
val_u = usage[usage['event_date_dt'] <= T].copy()
val_u['days_before'] = (T - val_u['event_date_dt']).dt.days
u_all = val_u.groupby('customer_id').agg(
    total_usage_all_time=('usage_id', 'count'),
    avg_usage_duration_all_time=('session_duration_sec', 'mean')
).reset_index()
u_60 = val_u[val_u['days_before'] <= 60].groupby('customer_id').agg(
    total_usage_60d=('usage_id', 'count'),
    avg_usage_duration_60d=('session_duration_sec', 'mean')
).reset_index()

# Tickets
val_tix = tickets[tickets['created_at_dt'] <= T].copy()
val_tix['days_before'] = (T - val_tix['created_at_dt']).dt.days
tix_60 = val_tix[val_tix['days_before'] <= 60].groupby('customer_id').agg(
    total_tickets_60d=('ticket_id', 'count'),
    missing_csat=('csat_score', lambda x: x.isna().sum())
).reset_index()
tix_60['missing_csat_rate_60d'] = np.where(tix_60['total_tickets_60d'] > 0, tix_60['missing_csat'] / tix_60['total_tickets_60d'], 0.0)
tix_60 = tix_60.drop(columns=['missing_csat'])

# Marketing
val_mkt = mkt[mkt['sent_at_dt'] <= T].copy()
val_mkt['days_before'] = (T - val_mkt['sent_at_dt']).dt.days
mkt_all = val_mkt.groupby('customer_id').agg(
    total_interactions_all_time=('interaction_id', 'count'),
    opened_all=('opened', 'sum'),
    clicked_all=('clicked', 'sum'),
    converted_all=('converted', 'sum')
).reset_index()
mkt_all['opened_rate_all_time'] = mkt_all['opened_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
mkt_all['clicked_rate_all_time'] = mkt_all['clicked_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
mkt_all['converted_rate_all_time'] = mkt_all['converted_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
mkt_all = mkt_all.drop(columns=['opened_all', 'clicked_all', 'converted_all'])

mkt_60 = val_mkt[val_mkt['days_before'] <= 60].groupby('customer_id').agg(
    total_interactions_60d=('interaction_id', 'count'),
    opened_60=('opened', 'sum'),
    clicked_60=('clicked', 'sum'),
    converted_60=('converted', 'sum')
).reset_index()
mkt_60['opened_rate_60d'] = mkt_60['opened_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
mkt_60['clicked_rate_60d'] = mkt_60['clicked_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
mkt_60['converted_rate_60d'] = mkt_60['converted_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
mkt_60 = mkt_60.drop(columns=['opened_60', 'clicked_60', 'converted_60'])

# Merge all into cohort
df_clean = cohort[['customer_id', 'gender', 'customer_age', 'customer_tenure', 'churn']].copy()
df_clean = df_clean.merge(ord_60, on='customer_id', how='left')
df_clean = df_clean.merge(pay_60, on='customer_id', how='left')
df_clean = df_clean.merge(sub_feats, on='customer_id', how='left')
df_clean = df_clean.merge(tix_60, on='customer_id', how='left')
df_clean = df_clean.merge(mkt_all, on='customer_id', how='left')
df_clean = df_clean.merge(mkt_60, on='customer_id', how='left')
df_clean = df_clean.merge(u_all, on='customer_id', how='left')
df_clean = df_clean.merge(u_60, on='customer_id', how='left')

# Fillna & derived features
df_clean = df_clean.fillna(0)
df_clean['plan_tier'] = df_clean['plan_tier'].replace(0, 'None').astype(str)
df_clean['gender'] = df_clean['gender'].astype(str)

df_clean['opened_rate_change'] = df_clean['opened_rate_60d'] - df_clean['opened_rate_all_time']
df_clean['clicked_rate_change'] = df_clean['clicked_rate_60d'] - df_clean['clicked_rate_all_time']
df_clean['converted_rate_change'] = df_clean['converted_rate_60d'] - df_clean['converted_rate_all_time']
df_clean['interaction_60d_share'] = np.where(df_clean['total_interactions_all_time'] > 0, df_clean['total_interactions_60d'] / df_clean['total_interactions_all_time'], 0.0)
df_clean['usage_60d_share'] = np.where(df_clean['total_usage_all_time'] > 0, df_clean['total_usage_60d'] / df_clean['total_usage_all_time'], 0.0)
df_clean['usage_duration_change'] = df_clean['avg_usage_duration_60d'] - df_clean['avg_usage_duration_all_time']

# Train XGBoost on this unified dataset
feat_cols = [c for c in df_clean.columns if c not in ['customer_id', 'churn', 'gender', 'plan_tier']]
X = df_clean[feat_cols]
y = df_clean['churn']

clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
preds = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
print('\n=== RESULTS ON UNIFIED POINT-IN-TIME (NO ASYMMETRIC BIAS) ===')
print(f'ROC-AUC: {roc_auc_score(y, preds):.4f}')
print(f'PR-AUC : {average_precision_score(y, preds):.4f}')
