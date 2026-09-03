import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv('data/processed/churn_feature_dataset_processed.csv')
print(f"Total Rows: {len(df):,}, Unique Customers: {df['customer_id'].nunique():,}")

cust_churn = df.groupby('customer_id')['label_churn'].max()
strat_labels = df['customer_id'].map(cust_churn)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, te_idx = next(sgkf.split(df, strat_labels, groups=df['customer_id']))

train_df = df.iloc[tr_idx]
test_df = df.iloc[te_idx]

train_custs = set(train_df['customer_id'])
test_custs = set(test_df['customer_id'])
overlap = train_custs.intersection(test_custs)

print(f"Train Rows: {len(train_df):,} | Test Rows: {len(test_df):,}")
print(f"Train Customers: {len(train_custs):,} | Test Customers: {len(test_custs):,}")
print(f"Customer Overlap: {len(overlap)} (Zero Leakage Check: {'PASSED' if len(overlap) == 0 else 'FAILED'})")
print(f"Train Churn Rate (rows): {train_df['label_churn'].mean()*100:.2f}% | Test Churn Rate (rows): {test_df['label_churn'].mean()*100:.2f}%")
print(f"Train Customer Churn Rate: {train_df.groupby('customer_id')['label_churn'].max().mean()*100:.2f}% | Test Customer Churn Rate: {test_df.groupby('customer_id')['label_churn'].max().mean()*100:.2f}%")
