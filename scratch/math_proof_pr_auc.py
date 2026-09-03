import sys
import numpy as np
import pandas as pd

# Mathematical simulation showing PR-AUC as a function of Prior Positive Rate P
# given the EXACT SAME underlying discrimination power (ROC curve / score distribution)

from sklearn.metrics import roc_auc_score, average_precision_score

np.random.seed(42)
n_samples = 100000

# True positive score distribution vs True negative score distribution
scores_pos = np.random.beta(5, 2, size=n_samples) # Mean ~ 0.71
scores_neg = np.random.beta(2, 5, size=n_samples) # Mean ~ 0.29

print("=== MATHEMATICAL DEMONSTRATION OF PR-AUC vs CLASS IMBALANCE PRIOR ===")
for p_rate in [0.0131, 0.0692, 0.1887, 0.5000]:
    n_pos = int(n_samples * p_rate)
    n_neg = int(n_samples * (1 - p_rate))
    
    y = np.array([1]*n_pos + [0]*n_neg)
    y_score = np.concatenate([scores_pos[:n_pos], scores_neg[:n_neg]])
    
    roc = roc_auc_score(y, y_score)
    pr = average_precision_score(y, y_score)
    lift = pr / p_rate
    print(f"Positive Rate P = {p_rate*100:5.2f}% | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Lift: {lift:.2f}x")
