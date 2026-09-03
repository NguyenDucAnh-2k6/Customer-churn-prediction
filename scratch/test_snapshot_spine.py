import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'

def load_and_dedup(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    return df.drop_duplicates()

def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce')

def test_monthly_snapshots():
    cust = load_and_dedup('churn_customers.csv')
    cust['signup_dt'] = safe_to_datetime(cust['signup_date'])
    cust['closed_dt'] = safe_to_datetime(cust['closed_date'])
    ref_date = pd.to_datetime('2026-08-25')
    
    print(f"Loaded {len(cust)} unique customers.")
    print("Signup min:", cust['signup_dt'].min(), "max:", cust['signup_dt'].max())
    print("Closed min:", cust['closed_dt'].min(), "max:", cust['closed_dt'].max())

    # Build customer monthly spine
    records = []
    for _, row in cust.iterrows():
        cid = row['customer_id']
        start = row['signup_dt'] if pd.notna(row['signup_dt']) else pd.to_datetime('2025-01-01')
        end = row['closed_dt'] if pd.notna(row['closed_dt']) else ref_date
        
        # Monthly periods
        months = pd.date_range(start=start.strftime('%Y-%m-01'), end=end.strftime('%Y-%m-01'), freq='MS')
        for m in months:
            m_str = m.strftime('%Y-%m')
            is_final_churn = int(row['account_status'] == 'Closed' and m.strftime('%Y-%m') == end.strftime('%Y-%m'))
            records.append({
                'customer_id': cid,
                'snapshot_month': m_str,
                'is_churn_snapshot': is_final_churn,
                'customer_is_churn': int(row['account_status'] == 'Closed')
            })
            
    df_spine = pd.DataFrame(records)
    print(f"Generated spine with {len(df_spine)} rows, {df_spine['customer_id'].nunique()} customers.")
    print("Snapshot months:", sorted(df_spine['snapshot_month'].unique()))

test_monthly_snapshots()
