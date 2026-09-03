import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier
from src.features.preprocessor import ChurnFeaturePreprocessor

df_tr = pd.read_csv('data/processed/latest/churn_train.csv')
df_va = pd.read_csv('data/processed/latest/churn_val.csv')
df_te = pd.read_csv('data/processed/latest/churn_test.csv')

print(f"Latest Train: {len(df_tr):,}, Val: {len(df_va):,}, Test: {len(df_te):,}")
print("Targets in Train:", [c for c in df_tr.columns if 'churn' in c])
print("churn_30d rate - Train:", df_tr['churn_30d'].mean(), "Val:", df_va['churn_30d'].mean(), "Test:", df_te['churn_30d'].mean())

preprocessor = ChurnFeaturePreprocessor(merge_static_master=True, include_stock_features=True)
df_tr_p = preprocessor.transform(df_tr)
df_va_p = preprocessor.transform(df_va)
df_te_p = preprocessor.transform(df_te)

all_churn_targets = ["churn_30d", "churn_60d", "churn_case1_30d", "churn_case2_30d", "churn_case2a_30d", "churn_case2b_30d"]
non_feature_cols = ["customer_id", "snapshot_month", "snapshot_date", "account_status"] + all_churn_targets
cat_cols = ["gender", "region", "city"]

feature_cols = [c for c in df_tr_p.columns if c not in non_feature_cols]

for col in cat_cols:
    if col in feature_cols:
        cats = {val: idx for idx, val in enumerate(df_tr_p[col].unique())}
        df_tr_p[col] = df_tr_p[col].map(cats).fillna(-1).astype(int)
        df_va_p[col] = df_va_p[col].map(cats).fillna(-1).astype(int)
        df_te_p[col] = df_te_p[col].map(cats).fillna(-1).astype(int)

X_train = df_tr_p[feature_cols]
y_train = df_tr_p['churn_30d'].astype(int)

X_val = df_va_p[feature_cols]
y_val = df_va_p['churn_30d'].astype(int)

X_test = df_te_p[feature_cols]
y_test = df_te_p['churn_30d'].astype(int)

clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42)
clf.fit(X_train, y_train)

preds_val = clf.predict_proba(X_val)[:, 1]
preds_test = clf.predict_proba(X_test)[:, 1]

print("\n=== EVALUATION ON LATEST DATASET ===")
print(f"Val Set  -> ROC-AUC: {roc_auc_score(y_val, preds_val):.4f}, PR-AUC: {average_precision_score(y_val, preds_val):.4f}")
print(f"Test Set -> ROC-AUC: {roc_auc_score(y_test, preds_test):.4f}, PR-AUC: {average_precision_score(y_test, preds_test):.4f}")

imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 Most Important Features in Latest:")
print(imp.head(15))
