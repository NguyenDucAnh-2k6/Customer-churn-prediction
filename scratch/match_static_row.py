import sys
import pandas as pd
import numpy as np

df_static = pd.read_csv('data/processed/dataset02_fixed.csv')
cust = pd.read_csv('data/churn_customers.csv')
cust['signup_dt'] = pd.to_datetime(cust['signup_date'], errors='coerce')
cust['closed_dt'] = pd.to_datetime(cust['closed_date'], errors='coerce')
mkt = pd.read_csv('data/churn_marketing_interactions.csv')
mkt_cnt = mkt.groupby('customer_id')['interaction_id'].count().to_dict()

# Row 2 in static:
# customer_tenure = 16.0, churn = 1.0, total_interactions_all_time = 1.0
candidates = cust[cust['account_status'] == 'Closed']
for cid in candidates['customer_id']:
    if mkt_cnt.get(cid, 0) == 1:
        c_row = cust[cust['customer_id'] == cid].iloc[0]
        diff_closed = (c_row['closed_dt'] - c_row['signup_dt']).days
        print(f"Customer {cid}: signup={c_row['signup_date']}, closed={c_row['closed_date']}, closed-signup diff={diff_closed} days")
