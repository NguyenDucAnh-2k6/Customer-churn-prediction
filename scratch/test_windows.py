import pandas as pd
import numpy as np

def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce')

ref_date = pd.to_datetime('2026-08-25')

# Test Orders 30d, 60d, 90d
orders = pd.read_csv('data/churn_orders.csv').drop_duplicates()
orders['order_date_dt'] = safe_to_datetime(orders['order_date'])
orders['days_ago'] = (ref_date - orders['order_date_dt']).dt.days

orders_30 = orders[(orders['days_ago'] >= 0) & (orders['days_ago'] <= 30)]
orders_60 = orders[(orders['days_ago'] >= 0) & (orders['days_ago'] <= 60)]
orders_90 = orders[(orders['days_ago'] >= 0) & (orders['days_ago'] <= 90)]

print(f"Total orders: {len(orders)}, in 30d: {len(orders_30)}, in 60d: {len(orders_60)}, in 90d: {len(orders_90)}")

# Test Product Usage 30d, 60d, 90d
usage = pd.read_csv('data/churn_product_usage.csv').drop_duplicates()
usage['event_date_dt'] = safe_to_datetime(usage['event_date'])
usage['days_ago'] = (ref_date - usage['event_date_dt']).dt.days

usage_30 = usage[(usage['days_ago'] >= 0) & (usage['days_ago'] <= 30)]
usage_60 = usage[(usage['days_ago'] >= 0) & (usage['days_ago'] <= 60)]
usage_90 = usage[(usage['days_ago'] >= 0) & (usage['days_ago'] <= 90)]

print(f"Total usage: {len(usage)}, in 30d: {len(usage_30)}, in 60d: {len(usage_60)}, in 90d: {len(usage_90)}")

# Test Marketing 30d, 60d, 90d
mkt = pd.read_csv('data/churn_marketing_interactions.csv').drop_duplicates()
mkt['sent_at_dt'] = safe_to_datetime(mkt['sent_at'])
mkt['days_ago'] = (ref_date - mkt['sent_at_dt']).dt.days

mkt_30 = mkt[(mkt['days_ago'] >= 0) & (mkt['days_ago'] <= 30)]
mkt_60 = mkt[(mkt['days_ago'] >= 0) & (mkt['days_ago'] <= 60)]
mkt_90 = mkt[(mkt['days_ago'] >= 0) & (mkt['days_ago'] <= 90)]

print(f"Total mkt: {len(mkt)}, in 30d: {len(mkt_30)}, in 60d: {len(mkt_60)}, in 90d: {len(mkt_90)}")

# Test Tickets 30d, 60d, 90d
tickets = pd.read_csv('data/churn_support_tickets.csv').drop_duplicates()
tickets['created_at_dt'] = safe_to_datetime(tickets['created_at'])
tickets['days_ago'] = (ref_date - tickets['created_at_dt']).dt.days

tix_30 = tickets[(tickets['days_ago'] >= 0) & (tickets['days_ago'] <= 30)]
tix_60 = tickets[(tickets['days_ago'] >= 0) & (tickets['days_ago'] <= 60)]
tix_90 = tickets[(tickets['days_ago'] >= 0) & (tickets['days_ago'] <= 90)]

print(f"Total tickets: {len(tickets)}, in 30d: {len(tix_30)}, in 60d: {len(tix_60)}, in 90d: {len(tix_90)}")
