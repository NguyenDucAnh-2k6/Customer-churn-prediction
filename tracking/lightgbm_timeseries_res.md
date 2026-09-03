### LIGHTGBM - Test Set (Optimized Threshold = 0.4753)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.4753` | `0.5000` | `-` |
| **ROC-AUC** | `0.9285` | `0.5000` | `+0.4285` |
| **PR-AUC (Average Precision)** | `0.6688` | `0.1823` (Base Rate) | `3.67x (+0.4865)` |
| **Precision@Top 5%** | `73.30%` (Lift: `4.02x`) | `18.23%` (Lift: `1.00x`) | `+55.07%` |
| **Recall@Top 10%** | `37.70%` | `10.00%` | `3.77x (+27.70%)` |
| **Accuracy** | `0.8575` | `0.7019` (Prior Match) | `+0.1556` |
| **Precision (Churn=1)** | `0.5681` | `0.1823` | `3.12x (+0.3858)` |
| **Recall (Churn=1)** | `0.9101` | `0.1823` (Prior Match) | `+0.7278` |
| **F1-Score (Churn=1)** | `0.6995` | `0.1823` | `3.84x (+0.5172)` |
| **Macro F1-Score** | `0.8031` | `0.5000` | `+0.3031` |
| **Brier Score** | `0.0859` | `0.1491` (Lower is better) | `-0.0632 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `23,212` | False Positive (FP): `4,234`
- False Negative (FN): `550` | True Positive (TP): `5,569`
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (LIGHTGBM - Test Set (Optimized Threshold = 0.4753))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |    
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `286 / 6,119` | `85.37%` | `4.67%` | `4.68x` |
| **Top 2%** | `671` | `521 / 6,119` | `77.65%` | `8.51%` | `4.26x` |
| **Top 5%** | `1,678` | `1,230 / 6,119` | `73.30%` | `20.10%` | `4.02x` |
| **Top 10%** | `3,356` | `2,307 / 6,119` | `68.74%` | `37.70%` | `3.77x` |
| **Top 20%** | `6,713` | `4,178 / 6,119` | `62.24%` | `68.28%` | `3.41x` |