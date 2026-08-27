import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.datasets import TimeSeriesDataset
from src.models.lstm import LSTMClassifier
from src.models.evaluate import calculate_top_k_metrics, format_top_k_table

ds = TimeSeriesDataset()
res = ds.load_and_split(
    use_cv=True,
    decay_half_life=12,
    customer_weight_power=0.5,
    use_usage_weight=True,
    behavioral_only=True,
)

X_train, y_train = res.X_train, res.y_train
X_test, y_test = res.X_test, res.y_test
train_weights = res.train_weights

# 1. XGBoost
print("[1/2] Training XGBoost...")
clf_xgb = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=2.5,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)
clf_xgb.fit(X_train, y_train, sample_weight=train_weights)
p_xgb = clf_xgb.predict_proba(X_test)[:, 1]

# 2. LSTM
print("[2/2] Training LSTM...")
clf_lstm = LSTMClassifier(
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.002,
    batch_size=1024,
    epochs=20,
    scale_pos_weight=2.5,
    patience=5,
    random_state=42,
)
clf_lstm.fit(X_train, y_train, sample_weight=train_weights, eval_set=[(X_test, y_test)])
p_lstm = clf_lstm.predict_proba(X_test)[:, 1]

# 3. Blending
print("\n" + "=" * 80)
print(f"{'Ensemble Weight':<25} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Top 1% Lift':<11} | {'Top 2% Lift':<11} | {'Top 5% Lift':<11} | {'Top 10% Rec':<11}")
print("-" * 80)

for w_xgb in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    w_lstm = 1.0 - w_xgb
    p_blend = w_xgb * p_xgb + w_lstm * p_lstm
    auc = roc_auc_score(y_test, p_blend)
    prauc = average_precision_score(y_test, p_blend)
    top_k = calculate_top_k_metrics(y_test.values, p_blend, k_percents=[1, 2, 5, 10, 20])
    
    t1_lift = top_k.loc[top_k["k_percent"] == 1, "lift_at_k"].values[0]
    t2_lift = top_k.loc[top_k["k_percent"] == 2, "lift_at_k"].values[0]
    t5_lift = top_k.loc[top_k["k_percent"] == 5, "lift_at_k"].values[0]
    t10_rec = top_k.loc[top_k["k_percent"] == 10, "recall_at_k"].values[0]
    
    label = f"XGB {w_xgb*100:.0f}% + LSTM {w_lstm*100:.0f}%"
    print(f"{label:<25} | {auc:<8.4f} | {prauc:<8.4f} | {t1_lift:<10.2f}x | {t2_lift:<10.2f}x | {t5_lift:<10.2f}x | {t10_rec*100:<10.2f}%")
print("=" * 80)
