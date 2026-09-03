### XGBOOST - Test Set (Optimized Threshold = 0.3367)

| Metric | Model Value | Random Baseline (Prior Rate = 16.57%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.3367` | `0.5000` | `-` |
| **ROC-AUC** | `0.9572` | `0.5000` | `+0.4572` |
| **PR-AUC (Average Precision)** | `0.8368` | `0.1657` (Base Rate) | `5.05x (+0.6711)` |
| **Precision@Top 5%** | `100.00%` (Lift: `6.04x`) | `16.57%` (Lift: `1.00x`) | `+83.43%` |
| **Recall@Top 10%** | `53.06%` | `10.00%` | `5.31x (+43.06%)` |
| **Accuracy** | `0.8944` | `0.7235` (Prior Match) | `+0.1709` |
| **Precision (Churn=1)** | `0.6488` | `0.1657` | `3.92x (+0.4831)` |
| **Recall (Churn=1)** | `0.7904` | `0.1657` (Prior Match) | `+0.6247` |       
| **F1-Score (Churn=1)** | `0.7126` | `0.1657` | `4.30x (+0.5469)` |
| **Macro F1-Score** | `0.8240` | `0.5000` | `+0.3240` |
| **Brier Score** | `0.0590` | `0.1382` (Lower is better) | `-0.0792 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `27,935` | False Positive (FP): `2,594`
- False Negative (FN): `1,271` | True Positive (TP): `4,792`
- Total Samples: `36,592` (Positive: `6,063`, Negative: `30,529`)

### Top-K Ranking & Lift Breakdown (XGBOOST - Test Set (Optimized Threshold = 0.3367))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `365` | `365 / 6,063` | `100.00%` | `6.02%` | `6.04x` |
| **Top 2%** | `731` | `731 / 6,063` | `100.00%` | `12.06%` | `6.04x` |        
| **Top 5%** | `1,829` | `1,829 / 6,063` | `100.00%` | `30.17%` | `6.04x` |    
| **Top 10%** | `3,659` | `3,217 / 6,063` | `87.92%` | `53.06%` | `5.31x` |    
| **Top 20%** | `7,318` | `4,768 / 6,063` | `65.15%` | `78.64%` | `3.93x` |