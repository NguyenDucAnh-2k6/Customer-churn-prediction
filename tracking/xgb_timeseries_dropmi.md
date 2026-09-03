### XGBOOST - Test Set (Optimized Threshold = 0.3763)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.3763` | `0.5000` | `-` |
| **ROC-AUC** | `0.9285` | `0.5000` | `+0.4285` |
| **PR-AUC (Average Precision)** | `0.6688` | `0.1823` (Base Rate) | `3.67x (+0.4865)` |
| **Precision@Top 5%** | `73.24%` (Lift: `4.02x`) | `18.23%` (Lift: `1.00x`) | `+55.01%` |
| **Recall@Top 10%** | `37.72%` | `10.00%` | `3.77x (+27.72%)` |
| **Accuracy** | `0.8555` | `0.7019` (Prior Match) | `+0.1536` |
| **Precision (Churn=1)** | `0.5641` | `0.1823` | `3.09x (+0.3818)` |
| **Recall (Churn=1)** | `0.9127` | `0.1823` (Prior Match) | `+0.7304` |
| **F1-Score (Churn=1)** | `0.6973` | `0.1823` | `3.82x (+0.5150)` |
| **Macro F1-Score** | `0.8012` | `0.5000` | `+0.3012` |
| **Brier Score** | `0.0821` | `0.1491` (Lower is better) | `-0.0670 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `23,130` | False Positive (FP): `4,316`
- False Negative (FN): `534` | True Positive (TP): `5,585`
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (XGBOOST - Test Set (Optimized Threshold = 0.3763))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |    
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `284 / 6,119` | `84.78%` | `4.64%` | `4.65x` |
| **Top 2%** | `671` | `522 / 6,119` | `77.79%` | `8.53%` | `4.27x` |
| **Top 5%** | `1,678` | `1,229 / 6,119` | `73.24%` | `20.08%` | `4.02x` |
| **Top 10%** | `3,356` | `2,308 / 6,119` | `68.77%` | `37.72%` | `3.77x` |
| **Top 20%** | `6,713` | `4,177 / 6,119` | `62.22%` | `68.26%` | `3.41x` |