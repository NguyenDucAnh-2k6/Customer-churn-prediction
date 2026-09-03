import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'

def load_and_dedup(filename):
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    return df

def safe_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')

def build_test():
    ref_date = pd.to_datetime('2026-08-25')

    # 1. Customers
    cust = load_and_dedup('churn_customers.csv')
    cust['is_churn'] = (cust['account_status'] == 'Closed').astype(int)
    cust['signup_date_dt'] = safe_to_datetime(cust['signup_date'])
    cust['closed_date_dt'] = safe_to_datetime(cust['closed_date'])
    cust['last_login_at_dt'] = safe_to_datetime(cust['last_login_at'])
    cust['birth_date_dt'] = safe_to_datetime(cust['birth_date'])
    
    cust['age'] = np.where(cust['birth_date_dt'].notna(), (ref_date - cust['birth_date_dt']).dt.days // 365, np.nan)
    end_tenure_date = cust['closed_date_dt'].fillna(ref_date)
    cust['tenure_days'] = (end_tenure_date - cust['signup_date_dt']).dt.days
    cust['days_since_last_login'] = (ref_date - cust['last_login_at_dt']).dt.days

    master = cust[['customer_id', 'is_churn', 'account_status', 'acquisition_channel', 
                   'gender', 'city', 'region', 'age', 'tenure_days', 'days_since_last_login']].copy()

    # 2. C360
    c360 = load_and_dedup('customer_360.csv')
    c360_churn = c360[c360['churn_original_id'].notna()].copy()
    c360_churn['customer_id'] = c360_churn['churn_original_id'].astype(int)
    c360_churn = c360_churn.drop_duplicates(subset=['customer_id'])
    c360_feats = c360_churn[['customer_id', 'province', 'crm_channel', 'national_id', 'phone']].copy()
    c360_feats['has_national_id'] = c360_feats['national_id'].notna().astype(int)
    c360_feats['has_phone'] = c360_feats['phone'].notna().astype(int)
    c360_feats = c360_feats.drop(columns=['national_id', 'phone'])
    master = master.merge(c360_feats, on='customer_id', how='left')

    # 3. Orders (Lifetime + 30d, 60d, 90d)
    orders = load_and_dedup('churn_orders.csv')
    orders['order_date_dt'] = safe_to_datetime(orders['order_date'])
    orders['days_ago'] = (ref_date - orders['order_date_dt']).dt.days

    # Lifetime agg
    order_agg = orders.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        completed_orders=('status', lambda x: (x == 'Completed').sum()),
        returned_orders=('status', lambda x: (x == 'Returned').sum()),
        cancelled_orders=('status', lambda x: (x == 'Cancelled').sum()),
        total_spent=('total_amount', 'sum'),
        avg_order_value=('total_amount', 'mean'),
        max_order_value=('total_amount', 'max'),
        days_since_last_order=('days_ago', 'min')
    ).reset_index()

    # 30d, 60d, 90d
    for days in [30, 60, 90]:
        sub = orders[(orders['days_ago'] >= 0) & (orders['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'orders_{days}d': ('order_id', 'count'),
                f'completed_orders_{days}d': ('status', lambda x: (x == 'Completed').sum()),
                f'cancelled_orders_{days}d': ('status', lambda x: (x == 'Cancelled').sum()),
                f'returned_orders_{days}d': ('status', lambda x: (x == 'Returned').sum()),
                f'total_spent_{days}d': ('total_amount', 'sum'),
                f'avg_order_value_{days}d': ('total_amount', 'mean'),
            }
        ).reset_index()
        order_agg = order_agg.merge(sub_agg, on='customer_id', how='left')

    # Order Velocity
    order_agg['order_velocity_30d_vs_90d'] = order_agg['orders_30d'].fillna(0) / (order_agg['orders_90d'].fillna(0) / 3.0 + 1e-5)
    order_agg['spend_velocity_30d_vs_90d'] = order_agg['total_spent_30d'].fillna(0) / (order_agg['total_spent_90d'].fillna(0) / 3.0 + 1e-5)
    master = master.merge(order_agg, on='customer_id', how='left')

    # 4. Items & Products
    order_items = load_and_dedup('churn_order_items.csv')
    products = load_and_dedup('churn_products.csv')
    items_prod = order_items.merge(products, on='product_id', how='left')
    items_orders = items_prod.merge(orders[['order_id', 'customer_id', 'days_ago']], on='order_id', how='left')
    
    item_agg = items_orders.groupby('customer_id').agg(
        total_items_purchased=('quantity', 'sum'),
        distinct_products_bought=('product_id', 'nunique'),
        distinct_categories_bought=('category', 'nunique'),
        top_category=('category', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()

    for days in [30, 60, 90]:
        sub = items_orders[(items_orders['days_ago'] >= 0) & (items_orders['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'items_purchased_{days}d': ('quantity', 'sum'),
                f'distinct_products_{days}d': ('product_id', 'nunique')
            }
        ).reset_index()
        item_agg = item_agg.merge(sub_agg, on='customer_id', how='left')
    master = master.merge(item_agg, on='customer_id', how='left')

    # 5. Payments (Lifetime + 30d, 60d, 90d)
    payments = load_and_dedup('churn_payments.csv')
    payments_order = payments.merge(orders[['order_id', 'days_ago']], on='order_id', how='left')
    
    pay_agg = payments_order.groupby('customer_id').agg(
        total_payments=('payment_id', 'count'),
        successful_payments=('status', lambda x: (x == 'Success').sum()),
        failed_payments=('status', lambda x: (x == 'Failed').sum()),
        total_payment_amount=('amount', 'sum'),
        primary_payment_method=('method', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()

    for days in [30, 60, 90]:
        sub = payments_order[(payments_order['days_ago'] >= 0) & (payments_order['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'payments_{days}d': ('payment_id', 'count'),
                f'successful_payments_{days}d': ('status', lambda x: (x == 'Success').sum()),
                f'failed_payments_{days}d': ('status', lambda x: (x == 'Failed').sum()),
                f'payment_amount_{days}d': ('amount', 'sum'),
            }
        ).reset_index()
        sub_agg[f'failed_payment_rate_{days}d'] = sub_agg[f'failed_payments_{days}d'] / (sub_agg[f'payments_{days}d'] + 1e-5)
        pay_agg = pay_agg.merge(sub_agg, on='customer_id', how='left')
    master = master.merge(pay_agg, on='customer_id', how='left')

    # 6. Subscriptions
    sub_df = load_and_dedup('churn_subscriptions.csv')
    sub_df['created_at_dt'] = safe_to_datetime(sub_df['created_at'])
    sub_df['days_ago'] = (ref_date - sub_df['created_at_dt']).dt.days
    sub_latest = sub_df.sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    sub_feats = sub_latest[['customer_id', 'plan_tier', 'status', 'auto_renew', 'change_type', 'days_ago']].copy()
    sub_feats = sub_feats.rename(columns={
        'status': 'sub_status',
        'plan_tier': 'sub_plan_tier',
        'auto_renew': 'sub_auto_renew',
        'change_type': 'sub_change_type',
        'days_ago': 'days_since_sub_created'
    })
    sub_feats['has_subscription'] = 1
    master = master.merge(sub_feats, on='customer_id', how='left')

    # 7. Support Tickets (Lifetime + 30d, 60d, 90d)
    tickets = load_and_dedup('churn_support_tickets.csv')
    tickets['created_at_dt'] = safe_to_datetime(tickets['created_at'])
    tickets['days_ago'] = (ref_date - tickets['created_at_dt']).dt.days
    
    tix_agg = tickets.groupby('customer_id').agg(
        total_support_tickets=('ticket_id', 'count'),
        avg_csat_score=('csat_score', 'mean'),
        avg_ticket_resolution_hours=('resolution_hours', 'mean'),
        urgent_tickets=('priority', lambda x: (x == 'Urgent').sum()),
        account_tickets=('category', lambda x: (x == 'Account').sum()),
        days_since_last_ticket=('days_ago', 'min')
    ).reset_index()

    for days in [30, 60, 90]:
        sub = tickets[(tickets['days_ago'] >= 0) & (tickets['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'tickets_{days}d': ('ticket_id', 'count'),
                f'urgent_tickets_{days}d': ('priority', lambda x: (x == 'Urgent').sum()),
                f'account_tickets_{days}d': ('category', lambda x: (x == 'Account').sum()),
                f'avg_csat_score_{days}d': ('csat_score', 'mean'),
                f'avg_ticket_resolution_hours_{days}d': ('resolution_hours', 'mean'),
            }
        ).reset_index()
        tix_agg = tix_agg.merge(sub_agg, on='customer_id', how='left')
    master = master.merge(tix_agg, on='customer_id', how='left')

    # 8. Product Usage (Lifetime + 30d, 60d, 90d)
    usage = load_and_dedup('churn_product_usage.csv')
    usage['event_date_dt'] = safe_to_datetime(usage['event_date'])
    usage['days_ago'] = (ref_date - usage['event_date_dt']).dt.days

    usage_agg = usage.groupby('customer_id').agg(
        total_usage_sessions=('usage_id', 'count'),
        total_usage_seconds=('session_duration_sec', 'sum'),
        avg_session_seconds=('session_duration_sec', 'mean'),
        primary_usage_device=('device', lambda x: x.mode()[0] if not x.empty else 'Unknown'),
        days_since_last_usage=('days_ago', 'min')
    ).reset_index()

    for days in [30, 60, 90]:
        sub = usage[(usage['days_ago'] >= 0) & (usage['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'usage_sessions_{days}d': ('usage_id', 'count'),
                f'usage_seconds_{days}d': ('session_duration_sec', 'sum'),
                f'avg_session_seconds_{days}d': ('session_duration_sec', 'mean'),
            }
        ).reset_index()
        usage_agg = usage_agg.merge(sub_agg, on='customer_id', how='left')

    usage_agg['usage_session_velocity_30d_vs_90d'] = usage_agg['usage_sessions_30d'].fillna(0) / (usage_agg['usage_sessions_90d'].fillna(0) / 3.0 + 1e-5)
    usage_agg['usage_seconds_velocity_30d_vs_90d'] = usage_agg['usage_seconds_30d'].fillna(0) / (usage_agg['usage_seconds_90d'].fillna(0) / 3.0 + 1e-5)
    master = master.merge(usage_agg, on='customer_id', how='left')

    # 9. Marketing Interactions (Lifetime + 30d, 60d, 90d)
    mkt = load_and_dedup('churn_marketing_interactions.csv')
    mkt['sent_at_dt'] = safe_to_datetime(mkt['sent_at'])
    mkt['days_ago'] = (ref_date - mkt['sent_at_dt']).dt.days

    mkt_agg = mkt.groupby('customer_id').agg(
        mkt_total_interactions=('interaction_id', 'count'),
        mkt_opened_count=('opened', 'sum'),
        mkt_clicked_count=('clicked', 'sum'),
        mkt_converted_count=('converted', 'sum'),
        days_since_last_mkt=('days_ago', 'min')
    ).reset_index()
    mkt_agg['mkt_open_rate'] = mkt_agg['mkt_opened_count'] / mkt_agg['mkt_total_interactions']
    mkt_agg['mkt_click_rate'] = mkt_agg['mkt_clicked_count'] / mkt_agg['mkt_total_interactions']
    mkt_agg['mkt_conversion_rate'] = mkt_agg['mkt_converted_count'] / mkt_agg['mkt_total_interactions']

    for days in [30, 60, 90]:
        sub = mkt[(mkt['days_ago'] >= 0) & (mkt['days_ago'] <= days)]
        sub_agg = sub.groupby('customer_id').agg(
            **{
                f'mkt_interactions_{days}d': ('interaction_id', 'count'),
                f'mkt_opened_{days}d': ('opened', 'sum'),
                f'mkt_clicked_{days}d': ('clicked', 'sum'),
                f'mkt_converted_{days}d': ('converted', 'sum'),
            }
        ).reset_index()
        sub_agg[f'mkt_open_rate_{days}d'] = sub_agg[f'mkt_opened_{days}d'] / sub_agg[f'mkt_interactions_{days}d']
        sub_agg[f'mkt_click_rate_{days}d'] = sub_agg[f'mkt_clicked_{days}d'] / sub_agg[f'mkt_interactions_{days}d']
        sub_agg[f'mkt_conversion_rate_{days}d'] = sub_agg[f'mkt_converted_{days}d'] / sub_agg[f'mkt_interactions_{days}d']
        mkt_agg = mkt_agg.merge(sub_agg, on='customer_id', how='left')

    mkt_agg['mkt_interaction_velocity_30d_vs_90d'] = mkt_agg['mkt_interactions_30d'].fillna(0) / (mkt_agg['mkt_interactions_90d'].fillna(0) / 3.0 + 1e-5)
    master = master.merge(mkt_agg, on='customer_id', how='left')

    # Fill default zeros for count/sum metrics
    zero_prefixes = ['total_', 'completed_', 'returned_', 'cancelled_', 'distinct_', 'successful_', 'failed_', 'urgent_', 'account_', 'items_purchased_', 'orders_', 'payments_', 'usage_', 'tickets_', 'mkt_interactions_', 'mkt_opened_', 'mkt_clicked_', 'mkt_converted_', 'has_']
    for col in master.columns:
        if any(col.startswith(p) for p in zero_prefixes):
            master[col] = master[col].fillna(0)

    print("Master shape:", master.shape)
    print("Master columns count:", len(master.columns))
    print("Sample columns:", master.columns.tolist()[:25])
    return master

if __name__ == '__main__':
    build_test()
