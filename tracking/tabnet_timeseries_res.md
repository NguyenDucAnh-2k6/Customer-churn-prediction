### TABNET - Test Set (Optimized Threshold = 0.3466)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.3466` | `0.5000` | `-` |
| **ROC-AUC** | `0.9258` | `0.5000` | `+0.4258` |
| **PR-AUC (Average Precision)** | `0.6596` | `0.1823` (Base Rate) | `3.62x (+0.4773)` |
| **Precision@Top 5%** | `71.63%` (Lift: `3.93x`) | `18.23%` (Lift: `1.00x`) | `+53.40%` |
| **Recall@Top 10%** | `37.13%` | `10.00%` | `3.71x (+27.13%)` |
| **Accuracy** | `0.8481` | `0.7019` (Prior Match) | `+0.1463` |
| **Precision (Churn=1)** | `0.5496` | `0.1823` | `3.01x (+0.3673)` |
| **Recall (Churn=1)** | `0.9252` | `0.1823` (Prior Match) | `+0.7428` |
| **F1-Score (Churn=1)** | `0.6896` | `0.1823` | `3.78x (+0.5073)` |
| **Macro F1-Score** | `0.7945` | `0.5000` | `+0.2945` |
| **Brier Score** | `0.0837` | `0.1491` (Lower is better) | `-0.0654 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `22,807` | False Positive (FP): `4,639`
- False Negative (FN): `458` | True Positive (TP): `5,661`
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (TABNET - Test Set (Optimized Threshold = 0.3466))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `284 / 6,119` | `84.78%` | `4.64%` | `4.65x` |
| **Top 2%** | `671` | `518 / 6,119` | `77.20%` | `8.47%` | `4.23x` |
| **Top 5%** | `1,678` | `1,202 / 6,119` | `71.63%` | `19.64%` | `3.93x` |
| **Top 10%** | `3,356` | `2,272 / 6,119` | `67.70%` | `37.13%` | `3.71x` |
| **Top 20%** | `6,713` | `4,138 / 6,119` | `61.64%` | `67.63%` | `3.38x` |