import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 1. Inspect TimeSeries dataset
ts_path = 'data/processed/churn_feature_dataset_processed.csv'
print(f"Loading {ts_path}...")
df_ts = pd.read_csv(ts_path)

print(f"Shape: {df_ts.shape}")
print(f"Unique customers: {df_ts['customer_id'].nunique():,}")
print(f"Snapshot months: {df_ts['snapshot_month'].unique().tolist() if 'snapshot_month' in df_ts else 'No snapshot_month'}")
print(f"Target column distribution (label_churn):")
if 'label_churn' in df_ts:
    print(df_ts['label_churn'].value_counts(normalize=True))
    print(df_ts['label_churn'].value_counts())

# Check how churn is distributed by snapshot_month
if 'snapshot_month' in df_ts and 'label_churn' in df_ts:
    print("\nChurn rate by snapshot_month:")
    print(df_ts.groupby('snapshot_month')['label_churn'].agg(['count', 'sum', 'mean']))

# Run XGBoost on timeseries dataset with time-based split (as configured in pipeline)
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier
from src.features.preprocessor import ChurnFeaturePreprocessor

preprocessor = ChurnFeaturePreprocessor(merge_static_master=False, include_stock_features=True)
df_trans = preprocessor.transform(df_ts)

exclude = ['customer_id', 'snapshot_month', 'snapshot_date', 'account_status', 'label_churn']
feat_cols = [c for c in df_trans.columns if c not in exclude]

# Default time split: train <= 2025-09, val 2025-10 to 2025-12, test >= 2026-01
train_mask = df_trans['snapshot_month'] <= '2025-09'
test_mask = df_trans['snapshot_month'] >= '2026-01'

X_train, y_train = df_trans.loc[train_mask, feat_cols], df_trans.loc[train_mask, 'label_churn'].astype(int)
X_test, y_test = df_trans.loc[test_mask, feat_cols], df_trans.loc[test_mask, 'label_churn'].astype(int)

print(f"\nTrain shape: {X_train.shape}, Churn rate: {y_train.mean():.4f} ({y_train.sum()}/{len(y_train)})")
print(f"Test shape:  {X_test.shape}, Churn rate: {y_test.mean():.4f} ({y_test.sum()}/{len(y_test)})")

clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict_proba(X_test)[:, 1]
print(f"\n=== TIMESERIES EVALUATION ON TEST SET (>= 2026-01) ===")
print(f"ROC-AUC: {roc_auc_score(y_test, preds):.4f}")
print(f"PR-AUC : {average_precision_score(y_test, preds):.4f}")

imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nTop 15 Most Important Features in TimeSeries:")
print(imp.head(15))
