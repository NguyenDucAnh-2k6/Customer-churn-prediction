### XGBOOST - Test Set (Optimized Threshold = 0.4159)

| Metric | Model Value | Random Baseline (Prior Rate = 18.23%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.4159` | `0.5000` | `-` |
| **ROC-AUC** | `0.9284` | `0.5000` | `+0.4284` |
| **PR-AUC (Average Precision)** | `0.6684` | `0.1823` (Base Rate) | `3.67x (+0.4861)` |
| **Precision@Top 5%** | `73.24%` (Lift: `4.02x`) | `18.23%` (Lift: `1.00x`) | `+55.01%` |
| **Recall@Top 10%** | `37.46%` | `10.00%` | `3.75x (+27.46%)` |
| **Accuracy** | `0.8548` | `0.7019` (Prior Match) | `+0.1530` |
| **Precision (Churn=1)** | `0.5625` | `0.1823` | `3.09x (+0.3802)` |
| **Recall (Churn=1)** | `0.9160` | `0.1823` (Prior Match) | `+0.7337` |
| **F1-Score (Churn=1)** | `0.6970` | `0.1823` | `3.82x (+0.5147)` |
| **Macro F1-Score** | `0.8008` | `0.5000` | `+0.3008` |       
| **Brier Score** | `0.0836` | `0.1491` (Lower is better) | `-0.0654 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `23,087` | False Positive (FP): `4,359`  
- False Negative (FN): `514` | True Positive (TP): `5,605`     
- Total Samples: `33,565` (Positive: `6,119`, Negative: `27,446`)

### Top-K Ranking & Lift Breakdown (XGBOOST - Test Set (Optimized Threshold = 0.4159))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `335` | `288 / 6,119` | `85.97%` | `4.71%` | `4.72x` |
| **Top 2%** | `671` | `528 / 6,119` | `78.69%` | `8.63%` | `4.32x` |
| **Top 5%** | `1,678` | `1,229 / 6,119` | `73.24%` | `20.08%` | `4.02x` |
| **Top 10%** | `3,356` | `2,292 / 6,119` | `68.30%` | `37.46%` | `3.75x` |
| **Top 20%** | `6,713` | `4,160 / 6,119` | `61.97%` | `67.98%` | `3.40x` |