### XGBOOST - 5-Fold Customer-Stratified Group CV (Out-Of-Fold) (Optimized Threshold = 0.4258)        

| Metric | Model Value | Random Baseline (Prior Rate = 19.03%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.4258` | `0.5000` | `-` |
| **ROC-AUC** | `0.9263` | `0.5000` | `+0.4263` |
| **PR-AUC (Average Precision)** | `0.6709` | `0.1903` (Base Rate) | `3.52x (+0.4806)` |
| **Precision@Top 5%** | `73.30%` (Lift: `3.85x`) | `19.03%` (Lift: `1.00x`) | `+54.27%` |
| **Recall@Top 10%** | `36.07%` | `10.00%` | `3.61x (+26.07%)` |
| **Accuracy** | `0.8529` | `0.6918` (Prior Match) | `+0.1611` |
| **Precision (Churn=1)** | `0.5710` | `0.1903` | `3.00x (+0.3807)` |
| **Recall (Churn=1)** | `0.9124` | `0.1903` (Prior Match) | `+0.7221` |
| **F1-Score (Churn=1)** | `0.7024` | `0.1903` | `3.69x (+0.5121)` |
| **Macro F1-Score** | `0.8024` | `0.5000` | `+0.3024` |
| **Brier Score** | `0.0852` | `0.1541` (Lower is better) | `-0.0689 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `90,012` | False Positive (FP): `17,286`
- False Negative (FN): `2,210` | True Positive (TP): `23,011`
- Total Samples: `132,519` (Positive: `25,221`, Negative: `107,298`)

### Top-K Ranking & Lift Breakdown (XGBOOST - 5-Fold Customer-Stratified Group CV (Out-Of-Fold) (Optimized Threshold = 0.4258))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `1,325` | `1,146 / 25,221` | `86.49%` | `4.54%` | `4.54x` |
| **Top 2%** | `2,650` | `2,100 / 25,221` | `79.25%` | `8.33%` | `4.16x` |
| **Top 5%** | `6,625` | `4,856 / 25,221` | `73.30%` | `19.25%` | `3.85x` |
| **Top 10%** | `13,251` | `9,098 / 25,221` | `68.66%` | `36.07%` | `3.61x` |
| **Top 20%** | `26,503` | `16,601 / 25,221` | `62.64%` | `65.82%` | `3.29x` |