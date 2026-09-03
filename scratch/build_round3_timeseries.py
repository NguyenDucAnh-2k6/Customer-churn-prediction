"""
Build Full Lakehouse (lqminh) TimeSeries Panel Dataset.
Generates monthly snapshot observations for all 10,002 customers from 2023-09 to 2026-07.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 70)
print("🚀 BUILDING FULL LAKEHOUSE (LQMINH) TIME-SERIES PANEL DATASET...")
print("=" * 70)

t0 = time.time()

# 1. Load silver tables
print("\n[1/7] Loading Silver tables from data/...")
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
sub_start_col = 'start_date' if 'start_date' in subs.columns else 'created_at'
subs['created_at_dt'] = pd.to_datetime(subs[sub_start_col], errors='coerce')
subs['end_date_dt'] = pd.to_datetime(subs['end_date'], errors='coerce')

mkt = pd.read_csv('data/churn_marketing_interactions.csv')
mkt['sent_at_dt'] = pd.to_datetime(mkt['sent_at'], errors='coerce')

tickets = pd.read_csv('data/churn_support_tickets.csv')
tickets['created_at_dt'] = pd.to_datetime(tickets['created_at'], errors='coerce')

print(f"  Customers: {len(cust):,}")
print(f"  Orders:    {len(orders):,}")
print(f"  Payments:  {len(payments):,}")
print(f"  Usage:     {len(usage):,}")
print(f"  Subs:      {len(subs):,}")
print(f"  Marketing: {len(mkt):,}")
print(f"  Tickets:   {len(tickets):,}")

# 2. Define monthly snapshot dates
snapshot_dates = pd.date_range(start='2023-09-30', end='2026-06-30', freq='M').tolist()
# Add latest available date
snapshot_dates.append(pd.to_datetime('2026-07-28'))
snapshot_dates = sorted(list(set(snapshot_dates)))
print(f"\n[2/7] Generated {len(snapshot_dates)} snapshot dates from {snapshot_dates[0].date()} to {snapshot_dates[-1].date()}")

horizon_days = 60 # Target: will customer churn in next 60 days?

panel_frames = []

print("\n[3/7] Processing monthly snapshots across all tables...")
for idx, T in enumerate(snapshot_dates):
    snap_str = T.strftime('%Y-%m') if T != snapshot_dates[-1] else '2026-07'
    
    # Filter active cohort at snapshot T:
    # 1. Registered before or on T
    # 2. Not closed before T
    cohort = cust[(cust['signup_dt'] <= T) & (cust['closed_dt'].isna() | (cust['closed_dt'] > T))].copy()
    if len(cohort) == 0:
        continue
    
    cohort['snapshot_month'] = snap_str
    cohort['snapshot_date'] = T.strftime('%Y-%m-%d')
    
    # Target: Multi-Component Churn Definition (Aligned with teammate schema)
    # c1: Closed account within horizon
    # c2: Free tier and NO next activity within horizon
    # c3: Downgraded and NO next activity within horizon
    t_end = T + pd.Timedelta(days=horizon_days)
    
    # Calculate next activity after T across orders, payments, usage, tickets
    u_next = usage[usage['event_date_dt'] > T].groupby('customer_id')['event_date_dt'].min()
    ord_next = orders[orders['order_date_dt'] > T].groupby('customer_id')['order_date_dt'].min()
    pay_next = payments[payments['payment_date_dt'] > T].groupby('customer_id')['payment_date_dt'].min()
    
    next_acts = pd.concat([u_next, ord_next, pay_next], axis=1).min(axis=1)
    cohort['next_act_after_t'] = cohort['customer_id'].map(next_acts)
    
    c1 = cohort['closed_dt'].notna() & (cohort['closed_dt'] > T) & (cohort['closed_dt'] <= t_end)
    
    # Subscriptions status at T
    subs_at_t = subs[subs['created_at_dt'] <= T].sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    is_free_map = subs_at_t.set_index('customer_id')['plan_tier'].map(lambda x: 1 if str(x).lower() == 'free' else 0).to_dict()
    cohort['is_free_tier'] = cohort['customer_id'].map(is_free_map).fillna(1).astype(int)
    
    inactive_h = cohort['next_act_after_t'].isna() | (cohort['next_act_after_t'] > t_end)
    c2 = (cohort['is_free_tier'] == 1) & inactive_h
    
    # Downgrade in horizon
    subs_downgrade = subs[(subs['created_at_dt'] > T) & (subs['created_at_dt'] <= t_end) & (subs['change_type'] == 'Downgrade')]['customer_id'].unique()
    c3 = cohort['customer_id'].isin(subs_downgrade) & inactive_h
    
    cohort['label_churn'] = (c1 | c2 | c3).astype(int)
    cohort = cohort.drop(columns=['next_act_after_t', 'is_free_tier'])
    
    # Customer baseline features
    cohort['customer_age'] = np.where(cohort['birth_dt'].notna(), (T - cohort['birth_dt']).dt.days // 365, 35.0)
    cohort['customer_tenure'] = (T - cohort['signup_dt']).dt.days.fillna(0).astype(int)
    
    # 1. Orders
    ord_v = orders[orders['order_date_dt'] <= T].copy()
    if len(ord_v) > 0:
        ord_v['days_before'] = (T - ord_v['order_date_dt']).dt.days
        ord_rec = ord_v.groupby('customer_id')['days_before'].min().rename('days_since_last_order')
        ord_60 = ord_v[ord_v['days_before'] <= 60].groupby('customer_id').agg(
            total_orders_60d=('order_id', 'count'),
            total_order_amounts_60d=('total_amount', 'sum'),
            avg_order_amount_60d=('total_amount', 'mean')
        )
        ord_30 = ord_v[ord_v['days_before'] <= 30].groupby('customer_id').agg(
            total_orders_30d=('order_id', 'count'),
            total_order_amounts_30d=('total_amount', 'sum'),
            avg_order_amount_30d=('total_amount', 'mean')
        )
        cohort = cohort.merge(ord_rec, on='customer_id', how='left').merge(ord_60, on='customer_id', how='left').merge(ord_30, on='customer_id', how='left')
    else:
        for c in ['days_since_last_order', 'total_orders_60d', 'total_order_amounts_60d', 'avg_order_amount_60d', 'total_orders_30d', 'total_order_amounts_30d', 'avg_order_amount_30d']:
            cohort[c] = 0.0

    # 2. Payments
    pay_v = payments[payments['payment_date_dt'] <= T].copy()
    if len(pay_v) > 0:
        pay_v['days_before'] = (T - pay_v['payment_date_dt']).dt.days
        pay_rec = pay_v.groupby('customer_id')['days_before'].min().rename('days_since_last_payment')
        pay_60 = pay_v[pay_v['days_before'] <= 60].groupby('customer_id').agg(
            total_payments_60d=('payment_id', 'count'),
            total_payment_amounts_60d=('amount', 'sum'),
            avg_payment_amount_60d=('amount', 'mean'),
            failed_payments_60d=('status', lambda x: (x == 'Failed').sum())
        )
        pay_60['failed_payment_rate_60d'] = np.where(pay_60['total_payments_60d'] > 0, pay_60['failed_payments_60d'] / pay_60['total_payments_60d'], 0.0)
        pay_60 = pay_60.drop(columns=['failed_payments_60d'])

        pay_30 = pay_v[pay_v['days_before'] <= 30].groupby('customer_id').agg(
            total_payments_30d=('payment_id', 'count'),
            total_payment_amounts_30d=('amount', 'sum'),
            avg_payment_amount_30d=('amount', 'mean'),
            failed_payments_30d=('status', lambda x: (x == 'Failed').sum())
        )
        pay_30['failed_payment_rate_30d'] = np.where(pay_30['total_payments_30d'] > 0, pay_30['failed_payments_30d'] / pay_30['total_payments_30d'], 0.0)
        pay_30 = pay_30.drop(columns=['failed_payments_30d'])

        cohort = cohort.merge(pay_rec, on='customer_id', how='left').merge(pay_60, on='customer_id', how='left').merge(pay_30, on='customer_id', how='left')
    else:
        for c in ['days_since_last_payment', 'total_payments_60d', 'total_payment_amounts_60d', 'avg_payment_amount_60d', 'failed_payment_rate_60d', 'total_payments_30d', 'total_payment_amounts_30d', 'avg_payment_amount_30d', 'failed_payment_rate_30d']:
            cohort[c] = 0.0

    # 3. Subscriptions
    subs_v = subs[subs['created_at_dt'] <= T].sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    if len(subs_v) > 0:
        subs_v['is_auto_renew'] = subs_v['auto_renew'].fillna(0).astype(float)
        subs_v['is_downgrade'] = (subs_v['change_type'] == 'Downgrade').astype(float)
        subs_v['days_until_end_from_snapshot'] = (subs_v['end_date_dt'] - T).dt.days.fillna(-999.0)
        subs_v['subscription_age_days'] = (T - subs_v['created_at_dt']).dt.days.fillna(0.0)
        subs_v['subscription_expired'] = ((subs_v['end_date_dt'] < T) | (subs_v['status'].isin(['Expired', 'Cancelled']))).astype(int)
        sub_cols = ['customer_id', 'plan_tier', 'is_auto_renew', 'is_downgrade', 'days_until_end_from_snapshot', 'subscription_age_days', 'subscription_expired']
        cohort = cohort.merge(subs_v[sub_cols], on='customer_id', how='left')
    else:
        for c in ['plan_tier', 'is_auto_renew', 'is_downgrade', 'days_until_end_from_snapshot', 'subscription_age_days', 'subscription_expired']:
            cohort[c] = 0.0

    # 4. Usage
    u_v = usage[usage['event_date_dt'] <= T].copy()
    if len(u_v) > 0:
        u_v['days_before'] = (T - u_v['event_date_dt']).dt.days
        u_rec = u_v.groupby('customer_id')['days_before'].min().rename('days_since_last_usage')
        u_all = u_v.groupby('customer_id').agg(
            total_usage_all_time=('usage_id', 'count'),
            avg_usage_duration_all_time=('session_duration_sec', 'mean')
        )
        u_60 = u_v[u_v['days_before'] <= 60].groupby('customer_id').agg(
            total_usage_60d=('usage_id', 'count'),
            avg_usage_duration_60d=('session_duration_sec', 'mean')
        )
        u_30 = u_v[u_v['days_before'] <= 30].groupby('customer_id').agg(
            total_usage_30d=('usage_id', 'count'),
            avg_usage_duration_30d=('session_duration_sec', 'mean')
        )
        cohort = cohort.merge(u_rec, on='customer_id', how='left').merge(u_all, on='customer_id', how='left').merge(u_60, on='customer_id', how='left').merge(u_30, on='customer_id', how='left')
    else:
        for c in ['days_since_last_usage', 'total_usage_all_time', 'avg_usage_duration_all_time', 'total_usage_60d', 'avg_usage_duration_60d', 'total_usage_30d', 'avg_usage_duration_30d']:
            cohort[c] = 0.0

    # 5. Tickets
    tix_v = tickets[tickets['created_at_dt'] <= T].copy()
    if len(tix_v) > 0:
        tix_v['days_before'] = (T - tix_v['created_at_dt']).dt.days
        tix_rec = tix_v.groupby('customer_id')['days_before'].min().rename('days_since_last_ticket')
        tix_60 = tix_v[tix_v['days_before'] <= 60].groupby('customer_id').agg(
            total_tickets_60d=('ticket_id', 'count'),
            missing_csat_count_60d=('csat_score', lambda x: x.isna().sum())
        )
        tix_60['missing_csat_rate_60d'] = np.where(tix_60['total_tickets_60d'] > 0, tix_60['missing_csat_count_60d'] / tix_60['total_tickets_60d'], 0.0)
        tix_60 = tix_60.drop(columns=['missing_csat_count_60d'])
        tix_30 = tix_v[tix_v['days_before'] <= 30].groupby('customer_id')['ticket_id'].count().rename('total_tickets_30d')
        cohort = cohort.merge(tix_rec, on='customer_id', how='left').merge(tix_60, on='customer_id', how='left').merge(tix_30, on='customer_id', how='left')
    else:
        for c in ['days_since_last_ticket', 'total_tickets_60d', 'missing_csat_rate_60d', 'total_tickets_30d']:
            cohort[c] = 0.0

    # 6. Marketing
    mkt_v = mkt[mkt['sent_at_dt'] <= T].copy()
    if len(mkt_v) > 0:
        mkt_v['days_before'] = (T - mkt_v['sent_at_dt']).dt.days
        mkt_all = mkt_v.groupby('customer_id').agg(
            total_interactions_all_time=('interaction_id', 'count'),
            opened_all=('opened', 'sum'),
            clicked_all=('clicked', 'sum')
        )
        mkt_all['opened_rate_all_time'] = mkt_all['opened_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
        mkt_all['clicked_rate_all_time'] = mkt_all['clicked_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
        mkt_all = mkt_all.drop(columns=['opened_all', 'clicked_all'])

        mkt_60 = mkt_v[mkt_v['days_before'] <= 60].groupby('customer_id').agg(
            total_interactions_60d=('interaction_id', 'count'),
            opened_60=('opened', 'sum'),
            clicked_60=('clicked', 'sum')
        )
        mkt_60['opened_rate_60d'] = mkt_60['opened_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
        mkt_60['clicked_rate_60d'] = mkt_60['clicked_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
        mkt_60 = mkt_60.drop(columns=['opened_60', 'clicked_60'])

        mkt_30 = mkt_v[mkt_v['days_before'] <= 30].groupby('customer_id').agg(
            total_interactions_30d=('interaction_id', 'count'),
            opened_30=('opened', 'sum'),
            clicked_30=('clicked', 'sum')
        )
        mkt_30['opened_rate_30d'] = mkt_30['opened_30'] / (mkt_30['total_interactions_30d'] + 1e-5)
        mkt_30['clicked_rate_30d'] = mkt_30['clicked_30'] / (mkt_30['total_interactions_30d'] + 1e-5)
        mkt_30 = mkt_30.drop(columns=['opened_30', 'clicked_30'])

        cohort = cohort.merge(mkt_all, on='customer_id', how='left').merge(mkt_60, on='customer_id', how='left').merge(mkt_30, on='customer_id', how='left')
    else:
        for c in ['total_interactions_all_time', 'opened_rate_all_time', 'clicked_rate_all_time', 'total_interactions_60d', 'opened_rate_60d', 'clicked_rate_60d', 'total_interactions_30d', 'opened_rate_30d', 'clicked_rate_30d']:
            cohort[c] = 0.0

    # Fillna default values for this snapshot
    fill_defaults = {
        'days_since_last_order': 999.0,
        'days_since_last_payment': 999.0,
        'days_since_last_usage': 999.0,
        'days_since_last_ticket': 999.0,
        'days_until_end_from_snapshot': -999.0,
        'plan_tier': 'None',
        'gender': 'Unknown',
    }
    for col, val in fill_defaults.items():
        if col in cohort.columns:
            cohort[col] = cohort[col].fillna(val)
    cohort = cohort.fillna(0.0)

    # 7. Velocity & Share Dynamics
    prev_u = np.maximum(0.0, cohort['total_usage_60d'] - cohort['total_usage_30d'])
    cohort['usage_velocity_30d_60d'] = cohort['total_usage_30d'] / (prev_u + 1.0)
    
    prev_ord = np.maximum(0.0, cohort['total_orders_60d'] - cohort['total_orders_30d'])
    cohort['orders_velocity_30d_60d'] = cohort['total_orders_30d'] / (prev_ord + 1.0)

    prev_p = np.maximum(0.0, cohort['total_payments_60d'] - cohort['total_payments_30d'])
    cohort['payments_velocity_30d_60d'] = cohort['total_payments_30d'] / (prev_p + 1.0)

    cohort['contract_churn_risk_score'] = (1.0 - cohort['is_auto_renew']) * 2.0 + cohort['is_downgrade'] * 2.0 + cohort['subscription_expired'] * 3.0
    cohort['is_renewal_imminent_30d'] = ((cohort['days_until_end_from_snapshot'] >= 0) & (cohort['days_until_end_from_snapshot'] <= 30)).astype(int)
    
    # Financial Technical Momentum
    # MACD on Usage: 30d usage vs (60d usage / 2)
    cohort['usage_macd'] = cohort['total_usage_30d'] - (cohort['total_usage_60d'] / 2.0)
    # Relative Peer Beta Z-Score
    mean_u = cohort['total_usage_30d'].mean()
    std_u = cohort['total_usage_30d'].std()
    cohort['peer_usage_zscore'] = (cohort['total_usage_30d'] - mean_u) / (std_u + 1e-5)

    # Keep clean columns
    drop_raw_dt = ['signup_dt', 'closed_dt', 'birth_dt', 'signup_date', 'closed_date', 'birth_date', 'last_login_at', 'account_status']
    cohort_clean = cohort.drop(columns=[c for c in drop_raw_dt if c in cohort.columns])
    panel_frames.append(cohort_clean)
    
    churn_cnt = cohort_clean['label_churn'].sum()
    churn_pct = cohort_clean['label_churn'].mean() * 100
    print(f"  [{idx+1:02d}/{len(snapshot_dates):02d}] {snap_str} ({T.date()}): {len(cohort_clean):,} active customers | Churn in next 60d: {churn_pct:5.2f}% ({churn_cnt}/{len(cohort_clean)})")

# Combine into master panel
df_panel = pd.concat(panel_frames, ignore_index=True)
print(f"\n[4/7] Master Time-Series Panel Built: {len(df_panel):,} rows x {df_panel.shape[1]} columns across {df_panel['customer_id'].nunique():,} unique customers.")
print(f"      Overall Positive Churn Rate: {df_panel['label_churn'].mean()*100:.2f}% ({df_panel['label_churn'].sum():,}/{len(df_panel):,})")

# Save panel dataset
out_dir = 'data/processed/round3_timeseries'
os.makedirs(out_dir, exist_ok=True)
out_csv = f'{out_dir}/churn_timeseries_master.csv'
df_panel.to_csv(out_csv, index=False)
print(f"\n[5/7] Saved Master Panel to '{out_csv}'")

# Split Train & Test with Customer-Stratified Group K-Fold (Zero Customer Leakage)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

print("\n[6/7] Running Customer-Stratified Group Split (5-Fold CV)...")
cust_churn = df_panel.groupby('customer_id')['label_churn'].max()
strat_label = df_panel['customer_id'].map(cust_churn)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

exclude_cols = ['customer_id', 'snapshot_month', 'snapshot_date', 'label_churn', 'gender', 'plan_tier', 'region', 'city', 'churn_reason', 'account_status']
feat_cols = [c for c in df_panel.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

pr_aucs = []
roc_aucs = []

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(df_panel, strat_label, df_panel['customer_id'])):
    X_tr, y_tr = df_panel.iloc[tr_idx][feat_cols], df_panel.iloc[tr_idx]['label_churn'].astype(int)
    X_te, y_te = df_panel.iloc[te_idx][feat_cols], df_panel.iloc[te_idx]['label_churn'].astype(int)
    
    clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_te)[:, 1]
    
    roc = roc_auc_score(y_te, preds)
    pr = average_precision_score(y_te, preds)
    roc_aucs.append(roc)
    pr_aucs.append(pr)
    print(f"  Fold {fold+1}: Train {len(X_tr):,} (churn: {y_tr.mean()*100:.2f}%), Test {len(X_te):,} (churn: {y_te.mean()*100:.2f}%) ──► ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

print("\n" + "=" * 70)
print(f"📊 BENCHMARK RESULTS ON LQMINH TIME-SERIES PANEL (5-FOLD GROUP CV):")
print(f"   Mean ROC-AUC : {np.mean(roc_aucs):.4f}")
print(f"   Mean PR-AUC  : {np.mean(pr_aucs):.4f}")
print(f"   Baseline PR  : {df_panel['label_churn'].mean():.4f}")
print(f"   Lift vs Base : {np.mean(pr_aucs) / df_panel['label_churn'].mean():.2f}x")
print("=" * 70)

clf.fit(df_panel[feat_cols], df_panel['label_churn'].astype(int))
imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nTop 15 Most Important Features in lqminh Time-Series Panel:")
print(imp.head(15))

print(f"\n🎉 Time elapsed: {time.time() - t0:.2f}s")
