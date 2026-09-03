"""
Build Complete Lakehouse (lqminh) Time-Series Panel Dataset with Full Teammate Features & Stock Technical Indicators.

Generates monthly panel observations with:
1. Complete Teammate Feature Set (Recency, Rolling 7d/30d/60d/90d/3m windows, Activity Days, CSAT, Slopes, Trends, Derived Rules).
2. Advanced Stock & Market Quantitative Features (RSI, Stochastic Oscillator, MACD Signal & Hist, Drawdown from Peak, Downside Volatility, Peer Beta & Cohort Strength).
3. Dynamic Contract, Renewal Proximity & Velocity Features.
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

from src.features.financial_indicators import add_all_financial_indicators
from src.features.contract_dynamics import compute_contract_dynamics

print("=" * 80)
print("🚀 BUILDING COMPLETE LQMINH TIME-SERIES PANEL DATASET (FULL TEAMMATE + STOCK FEATURES)...")
print("=" * 80)

t0 = time.time()

# 1. Load silver tables
print("\n[1/7] Loading Silver tables from data/...")
cust = pd.read_csv('data/churn_customers.csv')
cust['signup_dt'] = pd.to_datetime(cust['signup_date'], errors='coerce')
cust['closed_dt'] = pd.to_datetime(cust['closed_date'], errors='coerce')
cust['birth_dt'] = pd.to_datetime(cust['birth_date'], errors='coerce')
cust['last_login_dt'] = pd.to_datetime(cust.get('last_login_at', pd.Series(dtype=object)), errors='coerce')

orders = pd.read_csv('data/churn_orders.csv')
orders['order_date_dt'] = pd.to_datetime(orders['order_date'], errors='coerce')

payments = pd.read_csv('data/churn_payments.csv')
payments['payment_date_dt'] = pd.to_datetime(payments['payment_date'], errors='coerce')

usage = pd.read_csv('data/churn_product_usage.csv')
usage['event_date_dt'] = pd.to_datetime(usage['event_date'], errors='coerce')
usage['activity_day'] = usage['event_date_dt'].dt.normalize()

subs = pd.read_csv('data/churn_subscriptions.csv')
sub_start_col = 'start_date' if 'start_date' in subs.columns else 'created_at'
subs['created_at_dt'] = pd.to_datetime(subs[sub_start_col], errors='coerce')
subs['end_date_dt'] = pd.to_datetime(subs['end_date'], errors='coerce')

mkt = pd.read_csv('data/churn_marketing_interactions.csv')
mkt['sent_at_dt'] = pd.to_datetime(mkt['sent_at'], errors='coerce')

tickets = pd.read_csv('data/churn_support_tickets.csv')
tickets['created_at_dt'] = pd.to_datetime(tickets['created_at'], errors='coerce')
tickets['resolution_hours_clean'] = pd.to_numeric(tickets.get('resolution_hours', pd.Series(dtype=float)), errors='coerce')
tickets['resolved_at_dt'] = tickets['created_at_dt'] + pd.to_timedelta(tickets['resolution_hours_clean'].fillna(24.0), unit='h')

print(f"  Customers: {len(cust):,}")
print(f"  Orders:    {len(orders):,}")
print(f"  Payments:  {len(payments):,}")
print(f"  Usage:     {len(usage):,}")
print(f"  Subs:      {len(subs):,}")
print(f"  Marketing: {len(mkt):,}")
print(f"  Tickets:   {len(tickets):,}")

# 2. Define monthly snapshot dates
snapshot_dates = pd.date_range(start='2023-09-30', end='2026-06-30', freq='ME').tolist()
snapshot_dates.append(pd.to_datetime('2026-07-28'))
snapshot_dates = sorted(list(set(snapshot_dates)))
print(f"\n[2/7] Generated {len(snapshot_dates)} snapshot dates from {snapshot_dates[0].date()} to {snapshot_dates[-1].date()}")

horizon_days = 60 # Multi-component churn horizon

panel_frames = []

print("\n[3/7] Processing monthly snapshots across all tables...")
for idx, T in enumerate(snapshot_dates):
    snap_str = T.strftime('%Y-%m') if T != snapshot_dates[-1] else '2026-07'
    
    # Active cohort at T: registered <= T, closed > T or open
    cohort = cust[(cust['signup_dt'] <= T) & (cust['closed_dt'].isna() | (cust['closed_dt'] > T))].copy()
    if len(cohort) == 0:
        continue
    
    cohort['snapshot_month'] = snap_str
    cohort['snapshot_date'] = T.strftime('%Y-%m-%d')
    cohort['snapshot_month_ord'] = idx + 1
    
    # ── Target: Multi-Component Churn Definition ──
    t_end = T + pd.Timedelta(days=horizon_days)
    u_next = usage[usage['event_date_dt'] > T].groupby('customer_id')['event_date_dt'].min()
    ord_next = orders[orders['order_date_dt'] > T].groupby('customer_id')['order_date_dt'].min()
    pay_next = payments[payments['payment_date_dt'] > T].groupby('customer_id')['payment_date_dt'].min()
    next_acts = pd.concat([u_next, ord_next, pay_next], axis=1).min(axis=1)
    cohort['next_act_after_t'] = cohort['customer_id'].map(next_acts)
    
    c1 = cohort['closed_dt'].notna() & (cohort['closed_dt'] > T) & (cohort['closed_dt'] <= t_end)
    
    subs_at_t = subs[subs['created_at_dt'] <= T].sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    is_free_map = subs_at_t.set_index('customer_id')['plan_tier'].map(lambda x: 1 if str(x).lower() == 'free' else 0).to_dict()
    cohort['is_free_tier'] = cohort['customer_id'].map(is_free_map).fillna(1).astype(int)
    
    inactive_h = cohort['next_act_after_t'].isna() | (cohort['next_act_after_t'] > t_end)
    c2 = (cohort['is_free_tier'] == 1) & inactive_h
    
    subs_downgrade = subs[(subs['created_at_dt'] > T) & (subs['created_at_dt'] <= t_end) & (subs['change_type'] == 'Downgrade')]['customer_id'].unique()
    c3 = cohort['customer_id'].isin(subs_downgrade) & inactive_h
    
    cohort['label_churn'] = (c1 | c2 | c3).astype(int)
    cohort = cohort.drop(columns=['next_act_after_t'])

    # ── Customer Demographics & Tenures ──
    cohort['customer_age'] = np.where(cohort['birth_dt'].notna(), (T - cohort['birth_dt']).dt.days // 365, 35.0)
    cohort['age'] = cohort['customer_age']
    cohort['tenure_days'] = (T - cohort['signup_dt']).dt.days.fillna(0).astype(int)
    cohort['customer_tenure'] = cohort['tenure_days']
    tenure_months = np.maximum(cohort['tenure_days'] / 30.0, 1.0)
    
    # Login Recency
    if 'last_login_dt' in cohort.columns:
        cohort['days_since_last_login'] = np.where(
            (cohort['last_login_dt'].notna()) & (cohort['last_login_dt'] <= T),
            (T - cohort['last_login_dt']).dt.days,
            cohort['tenure_days']
        )
    else:
        cohort['days_since_last_login'] = cohort['tenure_days']

    # ── 1. Orders (30d, 60d, 90d, All-Time, Spend, AOV) ──
    ord_v = orders[orders['order_date_dt'] <= T].copy()
    if len(ord_v) > 0:
        ord_v['days_before'] = (T - ord_v['order_date_dt']).dt.days
        ord_rec = ord_v.groupby('customer_id')['days_before'].min().rename('days_since_last_order')
        
        ord_30 = ord_v[ord_v['days_before'] <= 30].groupby('customer_id').agg(
            total_orders_30d=('order_id', 'count'),
            total_order_amounts_30d=('total_amount', 'sum'),
            avg_order_amount_30d=('total_amount', 'mean')
        )
        ord_60 = ord_v[ord_v['days_before'] <= 60].groupby('customer_id').agg(
            total_orders_60d=('order_id', 'count'),
            total_order_amounts_60d=('total_amount', 'sum'),
            avg_order_amount_60d=('total_amount', 'mean')
        )
        ord_90 = ord_v[ord_v['days_before'] <= 90].groupby('customer_id').agg(
            orders_last_90d=('order_id', 'count'),
            total_order_amounts_90d=('total_amount', 'sum'),
            avg_order_value_90d=('total_amount', 'mean')
        )
        ord_all = ord_v.groupby('customer_id').agg(
            total_spend_to_date=('total_amount', 'sum'),
            has_completed_order=('order_id', lambda x: 1)
        )
        cohort = cohort.merge(ord_rec, on='customer_id', how='left') \
                       .merge(ord_30, on='customer_id', how='left') \
                       .merge(ord_60, on='customer_id', how='left') \
                       .merge(ord_90, on='customer_id', how='left') \
                       .merge(ord_all, on='customer_id', how='left')
    else:
        for c in ['days_since_last_order', 'total_orders_30d', 'total_order_amounts_30d', 'avg_order_amount_30d',
                  'total_orders_60d', 'total_order_amounts_60d', 'avg_order_amount_60d',
                  'orders_last_90d', 'total_order_amounts_90d', 'avg_order_value_90d',
                  'total_spend_to_date', 'has_completed_order']:
            cohort[c] = 0.0
    
    cohort['orders_last_30d'] = cohort['total_orders_30d'].fillna(0.0)
    cohort['avg_spend_to_date_per_month'] = cohort['total_spend_to_date'].fillna(0.0) / tenure_months

    # ── 2. Payments (30d, 60d, 90d, Rates, Recency) ──
    pay_v = payments[payments['payment_date_dt'] <= T].copy()
    if len(pay_v) > 0:
        pay_v['days_before'] = (T - pay_v['payment_date_dt']).dt.days
        pay_rec = pay_v.groupby('customer_id')['days_before'].min().rename('days_since_last_payment')
        
        pay_30 = pay_v[pay_v['days_before'] <= 30].groupby('customer_id').agg(
            total_payments_30d=('payment_id', 'count'),
            total_payment_amounts_30d=('amount', 'sum'),
            avg_payment_amount_30d=('amount', 'mean'),
            failed_payments_30d=('status', lambda x: (x == 'Failed').sum())
        )
        pay_30['failed_payment_rate_30d'] = np.where(pay_30['total_payments_30d'] > 0, pay_30['failed_payments_30d'] / pay_30['total_payments_30d'], 0.0)
        pay_30 = pay_30.drop(columns=['failed_payments_30d'])

        pay_60 = pay_v[pay_v['days_before'] <= 60].groupby('customer_id').agg(
            total_payments_60d=('payment_id', 'count'),
            total_payment_amounts_60d=('amount', 'sum'),
            avg_payment_amount_60d=('amount', 'mean'),
            failed_payments_60d=('status', lambda x: (x == 'Failed').sum())
        )
        pay_60['failed_payment_rate_60d'] = np.where(pay_60['total_payments_60d'] > 0, pay_60['failed_payments_60d'] / pay_60['total_payments_60d'], 0.0)
        pay_60 = pay_60.drop(columns=['failed_payments_60d'])

        pay_90 = pay_v[pay_v['days_before'] <= 90].groupby('customer_id').agg(
            total_payments_90d=('payment_id', 'count'),
            failed_payments_90d=('status', lambda x: (x == 'Failed').sum()),
            success_payments_90d=('status', lambda x: (x == 'Success').sum())
        )
        pay_90['payments_success_rate'] = np.where(pay_90['total_payments_90d'] > 0, pay_90['success_payments_90d'] / pay_90['total_payments_90d'], 1.0)
        pay_90 = pay_90.drop(columns=['success_payments_90d'])

        cohort = cohort.merge(pay_rec, on='customer_id', how='left') \
                       .merge(pay_30, on='customer_id', how='left') \
                       .merge(pay_60, on='customer_id', how='left') \
                       .merge(pay_90, on='customer_id', how='left')
    else:
        for c in ['days_since_last_payment', 'total_payments_30d', 'total_payment_amounts_30d', 'avg_payment_amount_30d', 'failed_payment_rate_30d',
                  'total_payments_60d', 'total_payment_amounts_60d', 'avg_payment_amount_60d', 'failed_payment_rate_60d',
                  'total_payments_90d', 'failed_payments_90d', 'payments_success_rate']:
            cohort[c] = 0.0

    cohort['payments_success_rate_missing'] = cohort['payments_success_rate'].isna().astype(int)
    cohort['payments_success_rate'] = cohort['payments_success_rate'].fillna(1.0)

    # ── 3. Subscriptions (Contract Proximity, Status, Up/Downgrades) ──
    subs_v = subs[subs['created_at_dt'] <= T].sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    if len(subs_v) > 0:
        subs_v['is_auto_renew'] = subs_v['auto_renew'].fillna(0).astype(float)
        subs_v['auto_renew'] = subs_v['is_auto_renew']
        subs_v['is_downgrade'] = (subs_v['change_type'] == 'Downgrade').astype(float)
        subs_v['days_until_end_from_snapshot'] = (subs_v['end_date_dt'] - T).dt.days.fillna(-999.0)
        subs_v['days_until_subscription_end'] = subs_v['days_until_end_from_snapshot']
        subs_v['subscription_age_days'] = (T - subs_v['created_at_dt']).dt.days.fillna(0.0)
        subs_v['days_on_current_tier'] = subs_v['subscription_age_days']
        subs_v['subscription_expired'] = ((subs_v['end_date_dt'] < T) | (subs_v['status'].isin(['Expired', 'Cancelled']))).astype(int)
        subs_v['is_paid_tier'] = subs_v['plan_tier'].map(lambda x: 1 if str(x).lower() in ['basic', 'premium', 'pro', 'enterprise'] else 0)
        subs_v['subscription_tier'] = subs_v['plan_tier']
        
        # 30d & 90d sub changes
        subs_30 = subs[(subs['created_at_dt'] > T - pd.Timedelta(days=30)) & (subs['created_at_dt'] <= T)]
        down30 = (subs_30['change_type'] == 'Downgrade').groupby(subs_30['customer_id']).sum().rename('had_downgrade_30d')
        
        subs_90 = subs[(subs['created_at_dt'] > T - pd.Timedelta(days=90)) & (subs['created_at_dt'] <= T)]
        down90 = (subs_90['change_type'] == 'Downgrade').groupby(subs_90['customer_id']).sum().rename('had_downgrade_90d')
        up90 = (subs_90['change_type'] == 'Upgrade').groupby(subs_90['customer_id']).sum().rename('had_upgrade_90d')
        plan_chg90 = subs_90.groupby('customer_id').size().rename('plan_changes_90d')

        sub_cols = ['customer_id', 'plan_tier', 'subscription_tier', 'is_auto_renew', 'auto_renew', 'is_downgrade',
                    'days_until_end_from_snapshot', 'days_until_subscription_end', 'subscription_age_days',
                    'days_on_current_tier', 'subscription_expired', 'is_paid_tier']
        cohort = cohort.merge(subs_v[sub_cols], on='customer_id', how='left') \
                       .merge(down30, on='customer_id', how='left') \
                       .merge(down90, on='customer_id', how='left') \
                       .merge(up90, on='customer_id', how='left') \
                       .merge(plan_chg90, on='customer_id', how='left')
    else:
        for c in ['plan_tier', 'subscription_tier', 'is_auto_renew', 'auto_renew', 'is_downgrade',
                  'days_until_end_from_snapshot', 'days_until_subscription_end', 'subscription_age_days',
                  'days_on_current_tier', 'subscription_expired', 'is_paid_tier',
                  'had_downgrade_30d', 'had_downgrade_90d', 'had_upgrade_90d', 'plan_changes_90d']:
            cohort[c] = 0.0

    # ── 4. Usage & Multi-Window Activity Days (7d, 14d, 30d, 60d, 90d, Diversity) ──
    u_v = usage[usage['event_date_dt'] <= T].copy()
    if len(u_v) > 0:
        u_v['days_before'] = (T - u_v['event_date_dt']).dt.days
        u_rec = u_v.groupby('customer_id')['days_before'].min().rename('days_since_last_usage')
        
        # Multi-window Active Days
        ad7 = u_v[u_v['days_before'] <= 7].groupby('customer_id')['activity_day'].nunique().rename('total_active_days_7d')
        ad14 = u_v[u_v['days_before'] <= 14].groupby('customer_id')['activity_day'].nunique().rename('total_active_days_14d')
        ad30 = u_v[u_v['days_before'] <= 30].groupby('customer_id')['activity_day'].nunique().rename('total_active_days_30d')
        ad60 = u_v[u_v['days_before'] <= 60].groupby('customer_id')['activity_day'].nunique().rename('total_active_days_60d')
        ad90 = u_v[u_v['days_before'] <= 90].groupby('customer_id')['activity_day'].nunique().rename('total_active_days_90d')

        # Usage events & durations
        u_7 = u_v[u_v['days_before'] <= 7].groupby('customer_id')['usage_id'].count().rename('num_usage_events_7d')
        u_30 = u_v[u_v['days_before'] <= 30].groupby('customer_id').agg(
            num_usage_events_30d=('usage_id', 'count'),
            total_usage_30d=('usage_id', 'count'),
            avg_session_duration_30d=('session_duration_sec', 'mean'),
            avg_usage_duration_30d=('session_duration_sec', 'mean'),
            total_session_time_30d=('session_duration_sec', 'sum'),
            event_type_diversity_30d=('event_type', 'nunique') if 'event_type' in u_v.columns else ('usage_id', lambda x: 1)
        )
        u_60 = u_v[u_v['days_before'] <= 60].groupby('customer_id').agg(
            num_usage_events_60d=('usage_id', 'count'),
            total_usage_60d=('usage_id', 'count'),
            avg_usage_duration_60d=('session_duration_sec', 'mean')
        )
        u_all = u_v.groupby('customer_id').agg(
            total_usage_all_time=('usage_id', 'count'),
            avg_usage_duration_all_time=('session_duration_sec', 'mean')
        )
        
        cohort = cohort.merge(u_rec, on='customer_id', how='left') \
                       .merge(ad7, on='customer_id', how='left') \
                       .merge(ad14, on='customer_id', how='left') \
                       .merge(ad30, on='customer_id', how='left') \
                       .merge(ad60, on='customer_id', how='left') \
                       .merge(ad90, on='customer_id', how='left') \
                       .merge(u_7, on='customer_id', how='left') \
                       .merge(u_30, on='customer_id', how='left') \
                       .merge(u_60, on='customer_id', how='left') \
                       .merge(u_all, on='customer_id', how='left')
    else:
        for c in ['days_since_last_usage', 'total_active_days_7d', 'total_active_days_14d', 'total_active_days_30d',
                  'total_active_days_60d', 'total_active_days_90d', 'num_usage_events_7d',
                  'num_usage_events_30d', 'total_usage_30d', 'avg_session_duration_30d', 'avg_usage_duration_30d',
                  'total_session_time_30d', 'event_type_diversity_30d', 'num_usage_events_60d', 'total_usage_60d',
                  'avg_usage_duration_60d', 'total_usage_all_time', 'avg_usage_duration_all_time']:
            cohort[c] = 0.0

    cohort['days_since_last_usage_event'] = cohort['days_since_last_usage']
    cohort['has_any_activity_7d'] = (cohort['total_active_days_7d'].fillna(0) > 0).astype(int)
    cohort['has_any_activity_14d'] = (cohort['total_active_days_14d'].fillna(0) > 0).astype(int)
    cohort['has_any_activity_30d'] = (cohort['total_active_days_30d'].fillna(0) > 0).astype(int)
    cohort['usage_7d_over_30d'] = np.where(cohort['total_usage_30d'].fillna(0) > 0, cohort['num_usage_events_7d'].fillna(0) / cohort['total_usage_30d'], 0.0)
    cohort['usage_60d_share'] = np.where(cohort['total_usage_60d'].fillna(0) > 0, cohort['total_usage_30d'].fillna(0) / cohort['total_usage_60d'], 0.5)

    # ── 5. Tickets & CSAT (30d, 90d, Unresolved, Missingness) ──
    tix_v = tickets[tickets['created_at_dt'] <= T].copy()
    if len(tix_v) > 0:
        tix_v['days_before'] = (T - tix_v['created_at_dt']).dt.days
        tix_rec = tix_v.groupby('customer_id')['days_before'].min().rename('days_since_last_ticket')
        
        tix_30 = tix_v[tix_v['days_before'] <= 30].groupby('customer_id')['ticket_id'].count().rename('total_tickets_30d')
        tix_60 = tix_v[tix_v['days_before'] <= 60].groupby('customer_id').agg(
            total_tickets_60d=('ticket_id', 'count'),
            missing_csat_count_60d=('csat_score', lambda x: x.isna().sum())
        )
        tix_60['missing_csat_rate_60d'] = np.where(tix_60['total_tickets_60d'] > 0, tix_60['missing_csat_count_60d'] / tix_60['total_tickets_60d'], 0.0)
        tix_60 = tix_60.drop(columns=['missing_csat_count_60d'])

        tix_90 = tix_v[tix_v['days_before'] <= 90].groupby('customer_id').agg(
            num_tickets_90d=('ticket_id', 'count'),
            avg_csat_score=('csat_score', 'mean'),
            tickets_about_cancel_90d=('subject', lambda x: (x.astype(str).str.lower().str.contains('cancel|churn|close|terminate')).sum()) if 'subject' in tix_v.columns else ('ticket_id', lambda x: 0)
        )
        # Unresolved tickets at T
        open_tk = tix_v[(tix_v['created_at_dt'] <= T) & (tix_v['resolved_at_dt'].isna() | (tix_v['resolved_at_dt'] > T))]
        unresolved = open_tk.groupby('customer_id').size().rename('has_unresolved_ticket')

        cohort = cohort.merge(tix_rec, on='customer_id', how='left') \
                       .merge(tix_30, on='customer_id', how='left') \
                       .merge(tix_60, on='customer_id', how='left') \
                       .merge(tix_90, on='customer_id', how='left') \
                       .merge(unresolved, on='customer_id', how='left')
    else:
        for c in ['days_since_last_ticket', 'total_tickets_30d', 'total_tickets_60d', 'missing_csat_rate_60d',
                  'num_tickets_90d', 'avg_csat_score', 'tickets_about_cancel_90d', 'has_unresolved_ticket']:
            cohort[c] = 0.0

    cohort['avg_csat_score_missing'] = cohort['avg_csat_score'].isna().astype(int)
    cohort['avg_csat_score'] = cohort['avg_csat_score'].fillna(3.5)
    cohort['has_unresolved_ticket'] = (cohort['has_unresolved_ticket'].fillna(0) > 0).astype(int)

    # ── 6. Marketing Interactions (30d, 60d, All-Time Rates) ──
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
        mkt_30['open_rate_30d'] = mkt_30['opened_rate_30d']
        mkt_30['clicked_rate_30d'] = mkt_30['clicked_30'] / (mkt_30['total_interactions_30d'] + 1e-5)
        mkt_30['has_marketing_click_30d'] = (mkt_30['clicked_30'] > 0).astype(int)
        mkt_30 = mkt_30.drop(columns=['opened_30', 'clicked_30'])

        cohort = cohort.merge(mkt_all, on='customer_id', how='left') \
                       .merge(mkt_60, on='customer_id', how='left') \
                       .merge(mkt_30, on='customer_id', how='left')
    else:
        for c in ['total_interactions_all_time', 'opened_rate_all_time', 'clicked_rate_all_time',
                  'total_interactions_60d', 'opened_rate_60d', 'clicked_rate_60d',
                  'total_interactions_30d', 'opened_rate_30d', 'open_rate_30d', 'clicked_rate_30d', 'has_marketing_click_30d']:
            cohort[c] = 0.0

    # ── 7. Overall Combined Activity Recency & Gaps ──
    cohort['days_since_last_order'] = cohort['days_since_last_order'].fillna(cohort['tenure_days'])
    cohort['days_since_last_payment'] = cohort['days_since_last_payment'].fillna(cohort['tenure_days'])
    cohort['days_since_last_usage'] = cohort['days_since_last_usage'].fillna(cohort['tenure_days'])
    cohort['days_since_last_ticket'] = cohort['days_since_last_ticket'].fillna(cohort['tenure_days'])
    cohort['days_until_end_from_snapshot'] = cohort['days_until_end_from_snapshot'].fillna(-999.0)
    
    cohort['days_since_last_activity'] = cohort[['days_since_last_login', 'days_since_last_usage', 'days_since_last_order']].min(axis=1)
    cohort['activity_gap_ratio'] = cohort['days_since_last_activity'] / (cohort['tenure_days'] + 1.0)

    # ── 8. Derived Business Rule Indicators ──
    cohort['free_and_inactive_14d'] = ((cohort['is_free_tier'] == 1) & (cohort['days_since_last_activity'] >= 14)).astype(int)
    cohort['free_and_inactive_21d'] = ((cohort['is_free_tier'] == 1) & (cohort['days_since_last_activity'] >= 21)).astype(int)
    cohort['paid_weak_engagement'] = ((cohort['is_paid_tier'] == 1) & (cohort['days_since_last_activity'] >= 14)).astype(int)
    cohort['recent_downgrade_and_quiet'] = ((cohort['had_downgrade_30d'] > 0) & (cohort['days_since_last_activity'] >= 7)).astype(int)
    cohort['auto_renew_off_paid'] = ((cohort['is_paid_tier'] == 1) & (cohort['is_auto_renew'] == 0)).astype(int)
    cohort['contract_churn_risk_score'] = (1.0 - cohort['is_auto_renew']) * 2.0 + cohort['is_downgrade'] * 2.0 + cohort['subscription_expired'] * 3.0
    cohort['is_renewal_imminent_30d'] = ((cohort['days_until_end_from_snapshot'] >= 0) & (cohort['days_until_end_from_snapshot'] <= 30)).astype(int)

    # ── 9. Velocity Ratios (30d vs 60d) ──
    prev_u = np.maximum(0.0, cohort['total_usage_60d'].fillna(0) - cohort['total_usage_30d'].fillna(0))
    cohort['usage_velocity_30d_60d'] = cohort['total_usage_30d'].fillna(0) / (prev_u + 1.0)
    cohort['usage_trend_30d'] = cohort['usage_velocity_30d_60d'] - 1.0
    
    prev_ord = np.maximum(0.0, cohort['total_orders_60d'].fillna(0) - cohort['total_orders_30d'].fillna(0))
    cohort['orders_velocity_30d_60d'] = cohort['total_orders_30d'].fillna(0) / (prev_ord + 1.0)

    prev_p = np.maximum(0.0, cohort['total_payments_60d'].fillna(0) - cohort['total_payments_30d'].fillna(0))
    cohort['payments_velocity_30d_60d'] = cohort['total_payments_30d'].fillna(0) / (prev_p + 1.0)

    # Clean raw datetime columns before adding to panel
    drop_raw_dt = ['signup_dt', 'closed_dt', 'birth_dt', 'signup_date', 'closed_date', 'birth_date', 'last_login_at', 'last_login_dt', 'account_status']
    cohort_clean = cohort.drop(columns=[c for c in drop_raw_dt if c in cohort.columns])
    panel_frames.append(cohort_clean)
    
    churn_cnt = cohort_clean['label_churn'].sum()
    churn_pct = cohort_clean['label_churn'].mean() * 100
    print(f"  [{idx+1:02d}/{len(snapshot_dates):02d}] {snap_str} ({T.date()}): {len(cohort_clean):,} active customers | Churn: {churn_pct:5.2f}% ({churn_cnt:,})")

# ── 10. Combine Full Panel & Calculate Cross-Month Rolling Momentum ──
df_panel = pd.concat(panel_frames, ignore_index=True)
print(f"\n[4/7] Raw Panel Built: {len(df_panel):,} rows across {df_panel['customer_id'].nunique():,} unique customers.")

print("\n[5/7] Computing Panel Rolling Momentum, Lags, Slopes & Stock Technical Indicators...")
df_panel = df_panel.sort_values(['customer_id', 'snapshot_date']).reset_index(drop=True)

# Lags & 3-month Rolling Window Calculations
g = df_panel.groupby('customer_id', sort=False)
df_panel['days_since_last_activity_lag1m'] = g['days_since_last_activity'].shift(1).fillna(df_panel['days_since_last_activity'])
df_panel['days_since_last_activity_diff1'] = df_panel['days_since_last_activity'] - df_panel['days_since_last_activity_lag1m']
df_panel['num_usage_events_30d_lag1m'] = g['total_usage_30d'].shift(1).fillna(0.0)
df_panel['num_usage_events_roll3m_sum'] = g['total_usage_30d'].transform(lambda s: s.rolling(3, min_periods=1).sum())
df_panel['avg_session_duration_roll3m_mean'] = g['avg_usage_duration_30d'].transform(lambda s: s.rolling(3, min_periods=1).mean())
df_panel['orders_roll3m_sum'] = g['total_orders_30d'].transform(lambda s: s.rolling(3, min_periods=1).sum())

# Slopes & Changing Rates
usage_lag2 = g['total_usage_30d'].shift(2).fillna(df_panel['total_usage_30d'])
df_panel['activity_slope_3m'] = (df_panel['total_usage_30d'] - usage_lag2) / 2.0
df_panel['is_declining_engagement'] = ((df_panel['total_usage_30d'] < df_panel['num_usage_events_30d_lag1m']) & (df_panel['num_usage_events_30d_lag1m'] < usage_lag2)).astype(int)
df_panel['reactivation_flag'] = ((df_panel['days_since_last_activity_lag1m'] > 30) & (df_panel['days_since_last_activity'] <= 30)).astype(int)

# Session duration change & marketing rate changes
dur_lag1 = g['avg_usage_duration_30d'].shift(1).fillna(df_panel['avg_usage_duration_30d'])
df_panel['session_duration_trend'] = df_panel['avg_usage_duration_30d'] - dur_lag1
df_panel['session_duration_trend_missing'] = df_panel['session_duration_trend'].isna().astype(int)
df_panel['usage_duration_change'] = df_panel['session_duration_trend'].fillna(0.0)

mkt_open_lag1 = g['opened_rate_30d'].shift(1).fillna(df_panel['opened_rate_30d'])
mkt_click_lag1 = g['clicked_rate_30d'].shift(1).fillna(df_panel['clicked_rate_30d'])
df_panel['opened_rate_change'] = df_panel['opened_rate_30d'] - mkt_open_lag1
df_panel['clicked_rate_change'] = df_panel['clicked_rate_30d'] - mkt_click_lag1

# ── 11. Add Quantitative Stock & Market Technical Features ──
df_panel = add_all_financial_indicators(
    df_panel,
    customer_id_col='customer_id',
    snapshot_month_col='snapshot_month',
    usage_col='total_usage_30d',
    active_days_col='total_active_days_30d'
)

# Fillna for all numeric columns
num_cols = df_panel.select_dtypes(include=[np.number]).columns
df_panel[num_cols] = df_panel[num_cols].fillna(0.0)

# Categorical column string fillna
for cat in ['gender', 'plan_tier', 'subscription_tier', 'region', 'city']:
    if cat in df_panel.columns:
        df_panel[cat] = df_panel[cat].fillna('Unknown').astype(str)

print(f"\n[6/7] Final Enriched Panel Shape: {df_panel.shape[0]:,} rows x {df_panel.shape[1]} columns.")
print(f"      Overall Positive Churn Rate: {df_panel['label_churn'].mean()*100:.2f}% ({df_panel['label_churn'].sum():,}/{len(df_panel):,})")

# ── 12. Save Dataset Outputs & Split Train/Test with Stratified Group K-Fold ──
from sklearn.model_selection import StratifiedGroupKFold

out_dir = 'data/processed/round3_timeseries'
os.makedirs(out_dir, exist_ok=True)
out_master = f'{out_dir}/churn_timeseries_master.csv'
df_panel.to_csv(out_master, index=False)
print(f"\n[Saved] Master TimeSeries -> '{out_master}'")

cust_churn = df_panel.groupby('customer_id')['label_churn'].max()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df_panel, df_panel['customer_id'].map(cust_churn), df_panel['customer_id']))

df_train = df_panel.iloc[train_idx].reset_index(drop=True)
df_test = df_panel.iloc[test_idx].reset_index(drop=True)

df_train.to_csv(f'{out_dir}/churn_timeseries_train.csv', index=False)
df_test.to_csv(f'{out_dir}/churn_timeseries_test.csv', index=False)
print(f"[Saved] Train Set -> '{out_dir}/churn_timeseries_train.csv' ({len(df_train):,} rows)")
print(f"[Saved] Test Set  -> '{out_dir}/churn_timeseries_test.csv' ({len(df_test):,} rows)")

# ── 13. Quick 5-Fold Benchmark Evaluation ──
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

print("\n[7/7] Running 5-Fold Customer-Stratified Group Benchmark (XGBoost)...")
exclude_cols = ['customer_id', 'snapshot_month', 'snapshot_date', 'label_churn', 'gender', 'plan_tier', 'subscription_tier', 'region', 'city', 'Unnamed: 0']
feat_cols = [c for c in df_panel.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

pr_aucs, roc_aucs = [], []
for fold, (tr_idx, te_idx) in enumerate(sgkf.split(df_panel, df_panel['customer_id'].map(cust_churn), df_panel['customer_id'])):
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

print("\n" + "=" * 80)
print(f"📊 ENRICHED LQMINH TIME-SERIES BENCHMARK (5-FOLD GROUP CV):")
print(f"   Mean ROC-AUC : {np.mean(roc_aucs):.4f}")
print(f"   Mean PR-AUC  : {np.mean(pr_aucs):.4f}")
print(f"   Baseline PR  : {df_panel['label_churn'].mean():.4f}")
print(f"   Lift vs Base : {np.mean(pr_aucs) / df_panel['label_churn'].mean():.2f}x")
print("=" * 80)

clf.fit(df_panel[feat_cols], df_panel['label_churn'].astype(int))
imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nTop 20 Most Important Features in Enriched Time-Series Panel:")
print(imp.head(20).to_string())

print(f"\n🎉 Total pipeline completed in: {time.time() - t0:.2f}s")
