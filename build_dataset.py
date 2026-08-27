"""
build_dataset.py
----------------
Script tự động làm sạch, thực hiện SQL-like JOINs và Feature Aggregation từ 10 file CSV trong data/
để tạo ra dataset hoàn chỉnh (data/churn_ml_dataset.csv) cho mô hình Customer Churn Prediction.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = 'data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'churn_ml_dataset.csv')

def load_and_dedup(filename):
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    orig_len = len(df)
    df = df.drop_duplicates()
    new_len = len(df)
    if orig_len != new_len:
        print(f"  [Deduplication] {filename}: {orig_len} -> {new_len} rows (Removed {orig_len - new_len} duplicates)")
    else:
        print(f"  [Loaded] {filename}: {new_len} rows")
    return df

def safe_to_datetime(series):
    # Safe parsing of dates, coercing out-of-range dates like year 20266 to NaT
    return pd.to_datetime(series, errors='coerce')

def main():
    print("=" * 60)
    print(">>> Bat dau qua trinh xay dung Customer Churn ML Dataset...")
    print("=" * 60)

    # 1. Khách hàng chính (churn_customers.csv)
    print("\n--- 1. Processing Master Customer Table ---")
    cust = load_and_dedup('churn_customers.csv')
    
    # Ground Truth target is_churn
    cust['is_churn'] = (cust['account_status'] == 'Closed').astype(int)
    
    # Safe date parsing
    cust['signup_date_dt'] = safe_to_datetime(cust['signup_date'])
    cust['closed_date_dt'] = safe_to_datetime(cust['closed_date'])
    cust['last_login_at_dt'] = safe_to_datetime(cust['last_login_at'])
    cust['birth_date_dt'] = safe_to_datetime(cust['birth_date'])
    
    # Compute age
    ref_date = pd.to_datetime('2026-08-25')
    cust['age'] = np.where(
        cust['birth_date_dt'].notna(),
        (ref_date - cust['birth_date_dt']).dt.days // 365,
        np.nan
    )
    
    # Compute customer tenure days
    # If closed, tenure = closed_date - signup_date; else ref_date - signup_date
    end_tenure_date = cust['closed_date_dt'].fillna(ref_date)
    cust['tenure_days'] = (end_tenure_date - cust['signup_date_dt']).dt.days
    
    # Days since last login
    cust['days_since_last_login'] = (ref_date - cust['last_login_at_dt']).dt.days

    # 2. Customer 360 (customer_360.csv)
    print("\n--- 2. Processing Customer 360 Metadata ---")
    c360 = load_and_dedup('customer_360.csv')
    c360_churn = c360[c360['churn_original_id'].notna()].copy()
    c360_churn['customer_id'] = c360_churn['churn_original_id'].astype(int)
    c360_churn = c360_churn.drop_duplicates(subset=['customer_id'])
    
    c360_feats = c360_churn[['customer_id', 'province', 'crm_channel', 'national_id', 'phone']].copy()
    c360_feats['has_national_id'] = c360_feats['national_id'].notna().astype(int)
    c360_feats['has_phone'] = c360_feats['phone'].notna().astype(int)
    c360_feats = c360_feats.drop(columns=['national_id', 'phone'])

    # 3. Orders, Order Items & Products (churn_orders, churn_order_items, churn_products)
    print("\n--- 3. Processing Orders & Product Purchases ---")
    orders = load_and_dedup('churn_orders.csv')
    order_items = load_and_dedup('churn_order_items.csv')
    products = load_and_dedup('churn_products.csv')

    # Aggregate orders per customer
    orders['order_date_dt'] = safe_to_datetime(orders['order_date'])
    
    order_agg = orders.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        completed_orders=('status', lambda x: (x == 'Completed').sum()),
        returned_orders=('status', lambda x: (x == 'Returned').sum()),
        cancelled_orders=('status', lambda x: (x == 'Cancelled').sum()),
        total_spent=('total_amount', 'sum'),
        avg_order_value=('total_amount', 'mean'),
        max_order_value=('total_amount', 'max'),
        last_order_date=('order_date_dt', 'max')
    ).reset_index()
    order_agg['days_since_last_order'] = (ref_date - order_agg['last_order_date']).dt.days
    order_agg = order_agg.drop(columns=['last_order_date'])

    # Items & Products aggregation
    items_prod = order_items.merge(products, on='product_id', how='left')
    items_orders = items_prod.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
    
    item_agg = items_orders.groupby('customer_id').agg(
        total_items_purchased=('quantity', 'sum'),
        distinct_products_bought=('product_id', 'nunique'),
        distinct_categories_bought=('category', 'nunique'),
        top_category=('category', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()

    # 4. Payments (churn_payments.csv)
    print("\n--- 4. Processing Payments ---")
    payments = load_and_dedup('churn_payments.csv')
    pay_agg = payments.groupby('customer_id').agg(
        total_payments=('payment_id', 'count'),
        successful_payments=('status', lambda x: (x == 'Success').sum()),
        failed_payments=('status', lambda x: (x == 'Failed').sum()),
        total_payment_amount=('amount', 'sum'),
        primary_payment_method=('method', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()

    # 5. Subscriptions (churn_subscriptions.csv)
    print("\n--- 5. Processing Subscriptions ---")
    sub = load_and_dedup('churn_subscriptions.csv')
    # Keep latest subscription per customer if multiple
    sub['created_at_dt'] = safe_to_datetime(sub['created_at'])
    sub_latest = sub.sort_values('created_at_dt').groupby('customer_id').last().reset_index()
    sub_feats = sub_latest[['customer_id', 'plan_tier', 'status', 'auto_renew', 'change_type']].copy()
    sub_feats = sub_feats.rename(columns={
        'status': 'sub_status',
        'plan_tier': 'sub_plan_tier',
        'auto_renew': 'sub_auto_renew',
        'change_type': 'sub_change_type'
    })
    sub_feats['has_subscription'] = 1

    # 6. Support Tickets (churn_support_tickets.csv)
    print("\n--- 6. Processing Support Tickets ---")
    tickets = load_and_dedup('churn_support_tickets.csv')
    tix_agg = tickets.groupby('customer_id').agg(
        total_support_tickets=('ticket_id', 'count'),
        avg_csat_score=('csat_score', 'mean'),
        avg_ticket_resolution_hours=('resolution_hours', 'mean'),
        urgent_tickets=('priority', lambda x: (x == 'Urgent').sum()),
        account_tickets=('category', lambda x: (x == 'Account').sum())
    ).reset_index()

    # 7. Product Usage (churn_product_usage.csv)
    print("\n--- 7. Processing Product Usage Telemetry ---")
    usage = load_and_dedup('churn_product_usage.csv')
    usage_agg = usage.groupby('customer_id').agg(
        total_usage_sessions=('usage_id', 'count'),
        total_usage_seconds=('session_duration_sec', 'sum'),
        avg_session_seconds=('session_duration_sec', 'mean'),
        primary_usage_device=('device', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()

    # 8. Marketing Interactions (churn_marketing_interactions.csv)
    print("\n--- 8. Processing Marketing Interactions ---")
    mkt = load_and_dedup('churn_marketing_interactions.csv')
    mkt_agg = mkt.groupby('customer_id').agg(
        mkt_total_interactions=('interaction_id', 'count'),
        mkt_opened_count=('opened', 'sum'),
        mkt_clicked_count=('clicked', 'sum'),
        mkt_converted_count=('converted', 'sum')
    ).reset_index()
    
    mkt_agg['mkt_open_rate'] = mkt_agg['mkt_opened_count'] / mkt_agg['mkt_total_interactions']
    mkt_agg['mkt_click_rate'] = mkt_agg['mkt_clicked_count'] / mkt_agg['mkt_total_interactions']
    mkt_agg['mkt_conversion_rate'] = mkt_agg['mkt_converted_count'] / mkt_agg['mkt_total_interactions']

    # 9. MASTER LEFT JOIN
    print("\n--- 9. Executing Master SQL LEFT JOINs ---")
    master = cust[['customer_id', 'is_churn', 'account_status', 'acquisition_channel', 
                   'gender', 'city', 'region', 'age', 'tenure_days', 'days_since_last_login']].copy()
    
    master = master.merge(c360_feats, on='customer_id', how='left')
    master = master.merge(order_agg, on='customer_id', how='left')
    master = master.merge(item_agg, on='customer_id', how='left')
    master = master.merge(pay_agg, on='customer_id', how='left')
    master = master.merge(sub_feats, on='customer_id', how='left')
    master = master.merge(tix_agg, on='customer_id', how='left')
    master = master.merge(usage_agg, on='customer_id', how='left')
    master = master.merge(mkt_agg, on='customer_id', how='left')

    # Impute default values for customers without transactions/activities
    fill_zero_cols = [
        'total_orders', 'completed_orders', 'returned_orders', 'cancelled_orders',
        'total_spent', 'total_items_purchased', 'distinct_products_bought', 'distinct_categories_bought',
        'total_payments', 'successful_payments', 'failed_payments', 'total_payment_amount',
        'has_subscription', 'total_support_tickets', 'urgent_tickets', 'account_tickets',
        'total_usage_sessions', 'total_usage_seconds', 'mkt_total_interactions',
        'mkt_opened_count', 'mkt_clicked_count', 'mkt_converted_count',
        'mkt_open_rate', 'mkt_click_rate', 'mkt_conversion_rate', 'has_national_id', 'has_phone'
    ]
    for col in fill_zero_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)
            
    fill_none_cols = ['sub_status', 'sub_plan_tier', 'sub_change_type', 'top_category', 'primary_payment_method', 'primary_usage_device', 'province', 'crm_channel']
    for col in fill_none_cols:
        if col in master.columns:
            master[col] = master[col].fillna('None')

    print(f"\n[Success] Master dataset built successfully!")
    print(f"[Info] Final Dataset Shape: {master.shape} (Rows: {master.shape[0]}, Columns: {master.shape[1]})")
    print(f"[Info] Target 'is_churn' value counts:\n{master['is_churn'].value_counts()}")
    print(f"[Info] Target 'is_churn' percentage:\n{master['is_churn'].value_counts(normalize=True) * 100}")

    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Saved] Saved complete ML dataset to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
