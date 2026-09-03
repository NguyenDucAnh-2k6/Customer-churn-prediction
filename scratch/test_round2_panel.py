import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'
ROUND2_DIR = os.path.join(DATA_DIR, 'processed', 'round2')

def load_and_dedup(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    return df.drop_duplicates()

def safe_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')

def build_round2_monthly_panel():
    ref_date = pd.to_datetime('2026-08-25')

    # 1. Customers
    cust = load_and_dedup('churn_customers.csv')
    cust['signup_dt'] = safe_to_datetime(cust['signup_date'])
    cust['closed_dt'] = safe_to_datetime(cust['closed_date'])
    cust['birth_dt'] = safe_to_datetime(cust['birth_date'])
    cust['last_login_dt'] = safe_to_datetime(cust['last_login_at'])

    # 2. C360
    c360 = load_and_dedup('customer_360.csv')
    c360_churn = c360[c360['churn_original_id'].notna()].copy()
    c360_churn['customer_id'] = c360_churn['churn_original_id'].astype(int)
    c360_churn = c360_churn.drop_duplicates(subset=['customer_id'])
    c360_feats = c360_churn[['customer_id', 'province', 'crm_channel', 'national_id', 'phone']].copy()
    c360_feats['has_national_id'] = c360_feats['national_id'].notna().astype(int)
    c360_feats['has_phone'] = c360_feats['phone'].notna().astype(int)
    c360_feats = c360_feats.drop(columns=['national_id', 'phone'])

    cust_full = cust.merge(c360_feats, on='customer_id', how='left')

    # Generate customer monthly spine
    spine_records = []
    for _, row in cust_full.iterrows():
        cid = int(row['customer_id'])
        start = row['signup_dt'] if pd.notna(row['signup_dt']) else pd.to_datetime('2025-01-01')
        end = row['closed_dt'] if pd.notna(row['closed_dt']) else ref_date
        
        # Monthly periods
        months = pd.date_range(start=start.strftime('%Y-%m-01'), end=end.strftime('%Y-%m-01'), freq='MS')
        for m in months:
            m_str = m.strftime('%Y-%m')
            # Snapshot date is the end of the month or cutoff
            m_end = (m + pd.offsets.MonthEnd(1))
            snap_date = min(m_end, ref_date)
            
            # Tenure in months at this snapshot
            tenure_m = (m.year - start.year) * 12 + (m.month - start.month)
            age_at_snap = (m.year - row['birth_dt'].year) if pd.notna(row['birth_dt']) else np.nan
            
            # Target label_churn: 1 if account closed and this snapshot is the final churn month, else 0
            is_closed = (row['account_status'] == 'Closed')
            is_churn_this_month = int(is_closed and (m_str == end.strftime('%Y-%m')))
            cust_churn_label = int(is_closed)

            spine_records.append({
                'customer_id': cid,
                'snapshot_month': m_str,
                'snapshot_date': snap_date.strftime('%Y-%m-%d'),
                'tenure_months': max(0, tenure_m),
                'age': age_at_snap,
                'gender': row['gender'],
                'city': row['city'],
                'region': row['region'],
                'province': row['province'],
                'crm_channel': row['crm_channel'],
                'has_national_id': row['has_national_id'],
                'has_phone': row['has_phone'],
                'acquisition_channel': row['acquisition_channel'],
                'account_status': row['account_status'],
                'label_churn': is_churn_this_month,
                'customer_churn_status': cust_churn_label
            })
            
    panel = pd.DataFrame(spine_records)

    # 3. Monthly Orders & Items
    orders = load_and_dedup('churn_orders.csv')
    orders['order_date_dt'] = safe_to_datetime(orders['order_date'])
    orders['order_month'] = orders['order_date_dt'].dt.strftime('%Y-%m')
    
    order_items = load_and_dedup('churn_order_items.csv')
    products = load_and_dedup('churn_products.csv')
    items_prod = order_items.merge(products, on='product_id', how='left')
    items_orders = items_prod.merge(orders[['order_id', 'customer_id', 'order_month']], on='order_id', how='left')

    m_orders = orders.groupby(['customer_id', 'order_month']).agg(
        monthly_orders=('order_id', 'count'),
        monthly_completed_orders=('status', lambda x: (x == 'Completed').sum()),
        monthly_cancelled_orders=('status', lambda x: (x == 'Cancelled').sum()),
        monthly_returned_orders=('status', lambda x: (x == 'Returned').sum()),
        monthly_spent=('total_amount', 'sum'),
        monthly_avg_order_value=('total_amount', 'mean')
    ).reset_index().rename(columns={'order_month': 'snapshot_month'})

    m_items = items_orders.groupby(['customer_id', 'order_month']).agg(
        monthly_items_purchased=('quantity', 'sum'),
        monthly_distinct_products=('product_id', 'nunique'),
        monthly_distinct_categories=('category', 'nunique'),
        monthly_top_category=('category', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index().rename(columns={'order_month': 'snapshot_month'})

    # 4. Monthly Payments
    payments = load_and_dedup('churn_payments.csv')
    # Join with orders to get order_month
    pay_orders = payments.merge(orders[['order_id', 'order_month']], on='order_id', how='left')
    m_payments = pay_orders.groupby(['customer_id', 'order_month']).agg(
        monthly_payments=('payment_id', 'count'),
        monthly_successful_payments=('status', lambda x: (x == 'Success').sum()),
        monthly_failed_payments=('status', lambda x: (x == 'Failed').sum()),
        monthly_payment_amount=('amount', 'sum'),
        monthly_primary_payment_method=('method', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index().rename(columns={'order_month': 'snapshot_month'})

    # 5. Monthly Subscriptions
    sub = load_and_dedup('churn_subscriptions.csv')
    sub['created_at_dt'] = safe_to_datetime(sub['created_at'])
    sub['sub_month'] = sub['created_at_dt'].dt.strftime('%Y-%m')
    sub_sorted = sub.sort_values('created_at_dt').groupby(['customer_id', 'sub_month']).last().reset_index()
    m_subs = sub_sorted[['customer_id', 'sub_month', 'plan_tier', 'status', 'auto_renew', 'change_type']].rename(columns={
        'sub_month': 'snapshot_month',
        'plan_tier': 'monthly_sub_plan_tier',
        'status': 'monthly_sub_status',
        'auto_renew': 'monthly_sub_auto_renew',
        'change_type': 'monthly_sub_change_type'
    })
    m_subs['has_subscription'] = 1

    # 6. Monthly Support Tickets
    tickets = load_and_dedup('churn_support_tickets.csv')
    tickets['created_at_dt'] = safe_to_datetime(tickets['created_at'])
    tickets['ticket_month'] = tickets['created_at_dt'].dt.strftime('%Y-%m')
    m_tickets = tickets.groupby(['customer_id', 'ticket_month']).agg(
        monthly_support_tickets=('ticket_id', 'count'),
        monthly_urgent_tickets=('priority', lambda x: (x == 'Urgent').sum()),
        monthly_account_tickets=('category', lambda x: (x == 'Account').sum()),
        monthly_avg_csat_score=('csat_score', 'mean'),
        monthly_avg_resolution_hours=('resolution_hours', 'mean')
    ).reset_index().rename(columns={'ticket_month': 'snapshot_month'})

    # 7. Monthly Product Usage
    usage = load_and_dedup('churn_product_usage.csv')
    usage['event_date_dt'] = safe_to_datetime(usage['event_date'])
    usage['usage_month'] = usage['event_date_dt'].dt.strftime('%Y-%m')
    usage['usage_day'] = usage['event_date_dt'].dt.strftime('%Y-%m-%d')
    m_usage = usage.groupby(['customer_id', 'usage_month']).agg(
        monthly_usage_sessions=('usage_id', 'count'),
        monthly_usage_seconds=('session_duration_sec', 'sum'),
        monthly_avg_session_seconds=('session_duration_sec', 'mean'),
        monthly_active_days=('usage_day', 'nunique'),
        monthly_primary_device=('device', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index().rename(columns={'usage_month': 'snapshot_month'})

    # 8. Monthly Marketing
    mkt = load_and_dedup('churn_marketing_interactions.csv')
    mkt['sent_at_dt'] = safe_to_datetime(mkt['sent_at'])
    mkt['mkt_month'] = mkt['sent_at_dt'].dt.strftime('%Y-%m')
    m_mkt = mkt.groupby(['customer_id', 'mkt_month']).agg(
        monthly_mkt_interactions=('interaction_id', 'count'),
        monthly_mkt_opened=('opened', 'sum'),
        monthly_mkt_clicked=('clicked', 'sum'),
        monthly_mkt_converted=('converted', 'sum')
    ).reset_index().rename(columns={'mkt_month': 'snapshot_month'})

    # 9. Merge all into Panel
    panel = panel.merge(m_orders, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_items, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_payments, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_subs, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_tickets, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_usage, on=['customer_id', 'snapshot_month'], how='left')
    panel = panel.merge(m_mkt, on=['customer_id', 'snapshot_month'], how='left')

    # Fill default zeros for numerical monthly activity columns
    fill_zero_cols = [
        'monthly_orders', 'monthly_completed_orders', 'monthly_cancelled_orders', 'monthly_returned_orders',
        'monthly_spent', 'monthly_items_purchased', 'monthly_distinct_products', 'monthly_distinct_categories',
        'monthly_payments', 'monthly_successful_payments', 'monthly_failed_payments', 'monthly_payment_amount',
        'has_subscription', 'monthly_support_tickets', 'monthly_urgent_tickets', 'monthly_account_tickets',
        'monthly_usage_sessions', 'monthly_usage_seconds', 'monthly_active_days',
        'monthly_mkt_interactions', 'monthly_mkt_opened', 'monthly_mkt_clicked', 'monthly_mkt_converted',
        'has_national_id', 'has_phone'
    ]
    for c in fill_zero_cols:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0)

    # Fill default 'None' for categorical columns
    fill_none_cols = [
        'province', 'crm_channel', 'monthly_top_category', 'monthly_primary_payment_method',
        'monthly_sub_plan_tier', 'monthly_sub_status', 'monthly_sub_change_type', 'monthly_primary_device'
    ]
    for c in fill_none_cols:
        if c in panel.columns:
            panel[c] = panel[c].fillna('None')

    print("Panel shape:", panel.shape)
    print("Columns:", panel.columns.tolist())
    print("Head:\n", panel.head(3))
    return panel

if __name__ == '__main__':
    build_round2_monthly_panel()
