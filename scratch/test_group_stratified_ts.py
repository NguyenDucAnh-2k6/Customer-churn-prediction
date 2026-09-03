import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from src.features.preprocessor import ChurnFeaturePreprocessor

df_ts = pd.read_csv('data/processed/churn_feature_dataset_processed.csv')
preprocessor = ChurnFeaturePreprocessor(merge_static_master=False, include_stock_features=True)
df_trans = preprocessor.transform(df_ts)

exclude = ['customer_id', 'snapshot_month', 'snapshot_date', 'account_status', 'label_churn']
feat_cols = [c for c in df_trans.columns if c not in exclude]

# Customer Group Stratification (Zero Customer Leakage)
cust_churn = df_trans.groupby('customer_id')['label_churn'].max()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

pr_aucs = []
roc_aucs = []

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(df_trans, df_trans['customer_id'].map(cust_churn), df_trans['customer_id'])):
    X_tr, y_tr = df_trans.iloc[tr_idx][feat_cols], df_trans.iloc[tr_idx]['label_churn'].astype(int)
    X_te, y_te = df_trans.iloc[te_idx][feat_cols], df_trans.iloc[te_idx]['label_churn'].astype(int)
    
    clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42)
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_te)[:, 1]
    
    roc = roc_auc_score(y_te, preds)
    pr = average_precision_score(y_te, preds)
    roc_aucs.append(roc)
    pr_aucs.append(pr)
    print(f"Fold {fold+1}: Train rows {len(X_tr):,} (churn {y_tr.mean():.4f}), Test rows {len(X_te):,} (churn {y_te.mean():.4f}) -> ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")

print(f"\n=== MEAN 5-FOLD CUSTOMER GROUP STRATIFIED TIME-SERIES ===")
print(f"Mean ROC-AUC: {np.mean(roc_aucs):.4f}")
print(f"Mean PR-AUC : {np.mean(pr_aucs):.4f}")
