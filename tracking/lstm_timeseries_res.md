### LSTM - Test Set (Optimized Threshold = 0.4753)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.4753` | `0.5000` | `-` |
| **ROC-AUC** | `0.9272` | `0.5000` | `+0.4272` |
| **PR-AUC (Average Precision)** | `0.6627` | `0.1823` (Base Rate) | `3.63x (+0.4804)` |
| **Precision@Top 5%** | `71.75%` (Lift: `3.94x`) | `18.23%` (Lift: `1.00x`) | `+53.52%` |
| **Recall@Top 10%** | `37.42%` | `10.00%` | `3.74x (+27.42%)` |
| **Accuracy** | `0.8547` | `0.7019` (Prior Match) | `+0.1528` |
| **Precision (Churn=1)** | `0.5631` | `0.1823` | `3.09x (+0.3808)` |
| **Recall (Churn=1)** | `0.9049` | `0.1823` (Prior Match) | `+0.7226` |
| **F1-Score (Churn=1)** | `0.6942` | `0.1823` | `3.81x (+0.5119)` |
| **Macro F1-Score** | `0.7994` | `0.5000` | `+0.2994` |
| **Brier Score** | `0.0871` | `0.1491` (Lower is better) | `-0.0620 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `23,150` | False Positive (FP): `4,296`
- False Negative (FN): `582` | True Positive (TP): `5,537`
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (LSTM - Test Set (Optimized Threshold = 0.4753))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `279 / 6,119` | `83.28%` | `4.56%` | `4.57x` |
| **Top 2%** | `671` | `519 / 6,119` | `77.35%` | `8.48%` | `4.24x` |
| **Top 5%** | `1,678` | `1,204 / 6,119` | `71.75%` | `19.68%` | `3.94x` |
| **Top 10%** | `3,356` | `2,290 / 6,119` | `68.24%` | `37.42%` | `3.74x` |
| **Top 20%** | `6,713` | `4,153 / 6,119` | `61.87%` | `67.87%` | `3.39x` |