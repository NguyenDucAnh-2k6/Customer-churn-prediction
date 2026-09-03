### CATBOOST - Test Set (Default Threshold = 0.5000)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.5000` | `0.5000` | `-` |
| **ROC-AUC** | `0.9284` | `0.5000` | `+0.4284` |
| **PR-AUC (Average Precision)** | `0.6680` | `0.1823` (Base Rate) | `3.66x (+0.4857)` |
| **Precision@Top 5%** | `73.00%` (Lift: `4.00x`) | `18.23%` (Lift: `1.00x`) | `+54.77%` |
| **Recall@Top 10%** | `37.64%` | `10.00%` | `3.76x (+27.64%)` |
| **Accuracy** | `0.8642` | `0.7019` (Prior Match) | `+0.1623` |
| **Precision (Churn=1)** | `0.5921` | `0.1823` | `3.25x (+0.4098)` |
| **Recall (Churn=1)** | `0.8193` | `0.1823` (Prior Match) | `+0.6369` |
| **F1-Score (Churn=1)** | `0.6874` | `0.1823` | `3.77x (+0.5051)` |
| **Macro F1-Score** | `0.8003` | `0.5000` | `+0.3003` |
| **Brier Score** | `0.0828` | `0.1491` (Lower is better) | `-0.0663 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `23,993` | False Positive (FP): `3,453`
- False Negative (FN): `1,106` | True Positive (TP): `5,013`
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (CATBOOST - Test Set (Default Threshold = 0.5000))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `284 / 6,119` | `84.78%` | `4.64%` | `4.65x` |
| **Top 2%** | `671` | `530 / 6,119` | `78.99%` | `8.66%` | `4.33x` |
| **Top 5%** | `1,678` | `1,225 / 6,119` | `73.00%` | `20.02%` | `4.00x` |
| **Top 10%** | `3,356` | `2,303 / 6,119` | `68.62%` | `37.64%` | `3.76x` |
| **Top 20%** | `6,713` | `4,165 / 6,119` | `62.04%` | `68.07%` | `3.40x` |