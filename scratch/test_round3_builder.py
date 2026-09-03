import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'
ROUND3_DIR = os.path.join(DATA_DIR, 'processed', 'round3')

def load_and_dedup(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    return df.drop_duplicates()

def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce')

def build_round3_dataset():
    ref_date = pd.to_datetime('2026-08-25')

    # 1. Customers
    cust = load_and_dedup('churn_customers.csv')
    cust['signup_dt'] = safe_to_datetime(cust['signup_date'])
    cust['closed_dt'] = safe_to_datetime(cust['closed_date'])
    cust['birth_dt'] = safe_to_datetime(cust['birth_date'])
    cust['last_login_dt'] = safe_to_datetime(cust['last_login_at'])

    # Customer snapshot date: ref_date if active, or closed_dt if closed
    cust['snapshot_dt'] = cust['closed_dt'].fillna(ref_date)
    cust['churn'] = (cust['account_status'] == 'Closed').astype(int)
    
    # customer_age & tenure
    cust['customer_age'] = np.where(cust['birth_dt'].notna(), (cust['snapshot_dt'] - cust['birth_dt']).dt.days // 365, np.nan)
    cust['customer_tenure'] = (cust['snapshot_dt'] - cust['signup_dt']).dt.days.fillna(0).astype(int)
    
    master = cust[['customer_id', 'gender', 'customer_age', 'customer_tenure', 'churn', 'snapshot_dt']].copy()

    # 2. Orders
    orders = load_and_dedup('churn_orders.csv')
    orders['order_date_dt'] = safe_to_datetime(orders['order_date'])
    orders = orders.merge(master[['customer_id', 'snapshot_dt']], on='customer_id', how='left')
    orders['days_before_snapshot'] = (orders['snapshot_dt'] - orders['order_date_dt']).dt.days

    # Valid orders up to snapshot
    valid_orders = orders[orders['days_before_snapshot'] >= 0].copy()
    
    # All-time orders stats
    ord_all = valid_orders.groupby('customer_id').agg(
        avg_order_amount=('total_amount', 'mean')
    ).reset_index()

    # 60d orders stats
    ord_60 = valid_orders[(valid_orders['days_before_snapshot'] <= 60)].groupby('customer_id').agg(
        total_order_amounts_60d=('total_amount', 'sum'),
        total_orders_60d=('order_id', 'count'),
        avg_order_amount_60d=('total_amount', 'mean')
    ).reset_index()

    # Last completed order
    comp_orders = valid_orders[valid_orders['status'] == 'Completed']
    last_comp = comp_orders.groupby('customer_id')['days_before_snapshot'].min().reset_index().rename(
        columns={'days_before_snapshot': 'days_since_last_completed_order'}
    )

    master = master.merge(ord_all, on='customer_id', how='left')
    master = master.merge(ord_60, on='customer_id', how='left')
    master = master.merge(last_comp, on='customer_id', how='left')

    # 3. Payments (60d)
    payments = load_and_dedup('churn_payments.csv')
    payments = payments.merge(orders[['order_id', 'days_before_snapshot']], on='order_id', how='left')
    
    valid_pay = payments[payments['days_before_snapshot'] >= 0].copy()
    pay_60 = valid_pay[valid_pay['days_before_snapshot'] <= 60].groupby('customer_id').agg(
        total_payment_amounts_60d=('amount', 'sum'),
        total_payments_60d=('payment_id', 'count'),
        avg_payment_amount_60d=('amount', 'mean'),
        failed_payments_60d=('status', lambda x: (x == 'Failed').sum())
    ).reset_index()
    pay_60['failed_payment_rate_60d'] = np.where(
        pay_60['total_payments_60d'] > 0,
        pay_60['failed_payments_60d'] / pay_60['total_payments_60d'],
        0.0
    )
    pay_60 = pay_60.drop(columns=['failed_payments_60d'])
    master = master.merge(pay_60, on='customer_id', how='left')

    # 4. Subscriptions
    subs = load_and_dedup('churn_subscriptions.csv')
    subs['created_at_dt'] = safe_to_datetime(subs['created_at'])
    subs['end_date_dt'] = safe_to_datetime(subs['end_date'])
    subs = subs.merge(master[['customer_id', 'snapshot_dt']], on='customer_id', how='left')
    
    # Sort and take latest subscription per customer up to snapshot
    subs_valid = subs[subs['created_at_dt'] <= subs['snapshot_dt']].copy()
    sub_latest = subs_valid.sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    
    sub_feats = pd.DataFrame()
    sub_feats['customer_id'] = sub_latest['customer_id']
    sub_feats['plan_tier'] = sub_latest['plan_tier'].fillna('None')
    sub_feats['is_auto_renew'] = sub_latest['auto_renew'].fillna(0).astype(float)
    sub_feats['is_downgrade'] = (sub_latest['change_type'] == 'Downgrade').astype(float)
    sub_feats['subscription_age_days'] = (sub_latest['snapshot_dt'] - sub_latest['created_at_dt']).dt.days.astype(float)
    sub_feats['days_until_end_from_snapshot'] = (sub_latest['end_date_dt'] - sub_latest['snapshot_dt']).dt.days.astype(float)
    sub_feats['subscription_expired'] = (
        (sub_latest['end_date_dt'] < sub_latest['snapshot_dt']) | 
        (sub_latest['status'].isin(['Expired', 'Cancelled']))
    ).astype(int)
    master = master.merge(sub_feats, on='customer_id', how='left')

    # 5. Support Tickets
    tickets = load_and_dedup('churn_support_tickets.csv')
    tickets['created_at_dt'] = safe_to_datetime(tickets['created_at'])
    tickets = tickets.merge(master[['customer_id', 'snapshot_dt']], on='customer_id', how='left')
    tickets['days_before_snapshot'] = (tickets['snapshot_dt'] - tickets['created_at_dt']).dt.days
    valid_tix = tickets[tickets['days_before_snapshot'] >= 0].copy()

    tix_all = valid_tix.groupby('customer_id').agg(
        avg_csat_score_all_time=('csat_score', 'mean')
    ).reset_index()

    tix_60 = valid_tix[valid_tix['days_before_snapshot'] <= 60].groupby('customer_id').agg(
        total_tickets_60d=('ticket_id', 'count'),
        avg_csat_score_60d=('csat_score', 'mean'),
        avg_resolution_hours_60d=('resolution_hours', 'mean'),
        missing_csat_count_60d=('csat_score', lambda x: x.isna().sum())
    ).reset_index()
    tix_60['missing_csat_rate_60d'] = np.where(
        tix_60['total_tickets_60d'] > 0,
        tix_60['missing_csat_count_60d'] / tix_60['total_tickets_60d'],
        0.0
    )
    tix_60 = tix_60.drop(columns=['missing_csat_count_60d'])

    master = master.merge(tix_all, on='customer_id', how='left')
    master = master.merge(tix_60, on='customer_id', how='left')

    # 6. Marketing Interactions
    mkt = load_and_dedup('churn_marketing_interactions.csv')
    mkt['sent_at_dt'] = safe_to_datetime(mkt['sent_at'])
    mkt = mkt.merge(master[['customer_id', 'snapshot_dt']], on='customer_id', how='left')
    mkt['days_before_snapshot'] = (mkt['snapshot_dt'] - mkt['sent_at_dt']).dt.days
    valid_mkt = mkt[mkt['days_before_snapshot'] >= 0].copy()

    mkt_all = valid_mkt.groupby('customer_id').agg(
        total_interactions_all_time=('interaction_id', 'count'),
        opened_all=('opened', 'sum'),
        clicked_all=('clicked', 'sum'),
        converted_all=('converted', 'sum')
    ).reset_index()
    mkt_all['opened_rate_all_time'] = mkt_all['opened_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
    mkt_all['clicked_rate_all_time'] = mkt_all['clicked_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
    mkt_all['converted_rate_all_time'] = mkt_all['converted_all'] / (mkt_all['total_interactions_all_time'] + 1e-5)
    mkt_all = mkt_all.drop(columns=['opened_all', 'clicked_all', 'converted_all'])

    mkt_60 = valid_mkt[valid_mkt['days_before_snapshot'] <= 60].groupby('customer_id').agg(
        total_interactions_60d=('interaction_id', 'count'),
        opened_60=('opened', 'sum'),
        clicked_60=('clicked', 'sum'),
        converted_60=('converted', 'sum')
    ).reset_index()
    mkt_60['opened_rate_60d'] = mkt_60['opened_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
    mkt_60['clicked_rate_60d'] = mkt_60['clicked_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
    mkt_60['converted_rate_60d'] = mkt_60['converted_60'] / (mkt_60['total_interactions_60d'] + 1e-5)
    mkt_60 = mkt_60.drop(columns=['opened_60', 'clicked_60', 'converted_60'])

    master = master.merge(mkt_all, on='customer_id', how='left')
    master = master.merge(mkt_60, on='customer_id', how='left')

    # Rate changes & share
    master['opened_rate_change'] = master['opened_rate_60d'].fillna(0) - master['opened_rate_all_time'].fillna(0)
    master['clicked_rate_change'] = master['clicked_rate_60d'].fillna(0) - master['clicked_rate_all_time'].fillna(0)
    master['converted_rate_change'] = master['converted_rate_60d'].fillna(0) - master['converted_rate_all_time'].fillna(0)
    master['interaction_60d_share'] = np.where(
        master['total_interactions_all_time'] > 0,
        master['total_interactions_60d'].fillna(0) / master['total_interactions_all_time'],
        0.0
    )

    # 7. Product Usage
    usage = load_and_dedup('churn_product_usage.csv')
    usage['event_date_dt'] = safe_to_datetime(usage['event_date'])
    usage = usage.merge(master[['customer_id', 'snapshot_dt']], on='customer_id', how='left')
    usage['days_before_snapshot'] = (usage['snapshot_dt'] - usage['event_date_dt']).dt.days
    valid_usage = usage[usage['days_before_snapshot'] >= 0].copy()

    usage_all = valid_usage.groupby('customer_id').agg(
        total_usage_all_time=('usage_id', 'count'),
        avg_usage_duration_all_time=('session_duration_sec', 'mean'),
        days_since_last_activity=('days_before_snapshot', 'min')
    ).reset_index()

    usage_60 = valid_usage[valid_usage['days_before_snapshot'] <= 60].groupby('customer_id').agg(
        total_usage_60d=('usage_id', 'count'),
        avg_usage_duration_60d=('session_duration_sec', 'mean')
    ).reset_index()

    master = master.merge(usage_all, on='customer_id', how='left')
    master = master.merge(usage_60, on='customer_id', how='left')

    master['usage_60d_share'] = np.where(
        master['total_usage_all_time'] > 0,
        master['total_usage_60d'].fillna(0) / master['total_usage_all_time'],
        0.0
    )
    master['usage_duration_change'] = master['avg_usage_duration_60d'].fillna(0) - master['avg_usage_duration_all_time'].fillna(0)

    # Clean strings / missing categorical values
    master['gender'] = master['gender'].fillna('Unknown').astype(str)
    master['plan_tier'] = master['plan_tier'].fillna('None').astype(str)

    # Fill 0 for non-occurring counts
    zero_fill_cols = [
        'total_order_amounts_60d', 'total_orders_60d', 'avg_order_amount_60d',
        'total_payment_amounts_60d', 'total_payments_60d', 'avg_payment_amount_60d', 'failed_payment_rate_60d',
        'is_auto_renew', 'is_downgrade', 'subscription_expired',
        'total_tickets_60d', 'missing_csat_rate_60d',
        'total_interactions_all_time', 'opened_rate_all_time', 'clicked_rate_all_time', 'converted_rate_all_time',
        'total_interactions_60d', 'opened_rate_60d', 'clicked_rate_60d', 'converted_rate_60d',
        'opened_rate_change', 'clicked_rate_change', 'converted_rate_change', 'interaction_60d_share',
        'total_usage_all_time', 'total_usage_60d', 'usage_60d_share', 'usage_duration_change'
    ]
    for col in zero_fill_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    # Ensure EXACT column list requested by user
    target_columns = [
        'customer_id',
        'gender',
        'customer_age',
        'customer_tenure',
        'avg_order_amount',
        'total_order_amounts_60d',
        'total_orders_60d',
        'total_payment_amounts_60d',
        'total_payments_60d',
        'failed_payment_rate_60d',
        'days_until_end_from_snapshot',
        'is_auto_renew',
        'is_downgrade',
        'plan_tier',
        'subscription_age_days',
        'avg_csat_score_all_time',
        'avg_csat_score_60d',
        'avg_resolution_hours_60d',
        'total_tickets_60d',
        'missing_csat_rate_60d',
        'total_interactions_all_time',
        'opened_rate_all_time',
        'clicked_rate_all_time',
        'converted_rate_all_time',
        'total_interactions_60d',
        'opened_rate_60d',
        'clicked_rate_60d',
        'converted_rate_60d',
        'total_usage_all_time',
        'avg_usage_duration_all_time',
        'total_usage_60d',
        'avg_usage_duration_60d',
        'days_since_last_activity',
        'days_since_last_completed_order',
        'avg_order_amount_60d',
        'avg_payment_amount_60d',
        'subscription_expired',
        'opened_rate_change',
        'clicked_rate_change',
        'converted_rate_change',
        'interaction_60d_share',
        'usage_60d_share',
        'usage_duration_change',
        'churn'
    ]

    final_df = master[target_columns].copy()
    print("Final DataFrame Shape:", final_df.shape)
    print("Columns match exactly:", final_df.columns.tolist() == target_columns)
    print("Dtypes:\n", final_df.dtypes)
    return final_df

if __name__ == '__main__':
    df = build_round3_dataset()
