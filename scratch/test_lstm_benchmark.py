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

from src.data.datasets import TimeSeriesDataset
from src.models.lstm import LSTMClassifier, LSTMModelWrapper
from src.models.evaluate import calculate_top_k_metrics, format_top_k_table

print("=" * 60)
print("🧠 BENCHMARKING PYTORCH LSTM ON TIMESERIES DATASET")
print("=" * 60)

# Load TimeSeries Dataset (Behavioral Only / Strategy 4)
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

print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
print(f"Features ({len(res.feature_names)}): {res.feature_names[:10]}...")

# 1. Train Baseline LSTM Model
start_t = time.time()
print("\n[TRAIN] Training PyTorch LSTM Classifier (Hidden=64, Layers=2, Batch=512, Epochs=25)...")

lstm = LSTMClassifier(
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    bidirectional=False,
    use_attention=True,
    learning_rate=0.002,
    weight_decay=1e-4,
    batch_size=512,
    epochs=25,
    scale_pos_weight=2.5,
    patience=5,
    random_state=42,
)

lstm.fit(
    X_train,
    y_train,
    sample_weight=train_weights,
    eval_set=[(X_test, y_test)],
)

elapsed = time.time() - start_t
print(f"[TRAIN] Finished LSTM training in {elapsed:.2f}s")

# Evaluate on Test Set
p_test = lstm.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, p_test)
test_prauc = average_precision_score(y_test, p_test)
top_k = calculate_top_k_metrics(y_test.values, p_test)

print(f"\n📊 LSTM TEST ROC-AUC: {test_auc:.4f} | PR-AUC: {test_prauc:.4f}")
print(format_top_k_table(top_k))
