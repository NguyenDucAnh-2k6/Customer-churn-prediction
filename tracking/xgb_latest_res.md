### XGBOOST - Test Set (Optimized Threshold = 0.4060)

| Metric | Model Value | Random Baseline (Prior Rate = 18.06%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.4060` | `0.5000` | `-` |
| **ROC-AUC** | `0.9187` | `0.5000` | `+0.4187` |
| **PR-AUC (Average Precision)** | `0.6307` | `0.1806` (Base Rate) | `3.49x (+0.4501)` |
| **Precision@Top 5%** | `69.91%` (Lift: `3.87x`) | `18.06%` (Lift: `1.00x`) | `+51.85%` |
| **Recall@Top 10%** | `36.33%` | `10.00%` | `3.63x (+26.33%)` |
| **Accuracy** | `0.8481` | `0.7040` (Prior Match) | `+0.1441` |
| **Precision (Churn=1)** | `0.5490` | `0.1806` | `3.04x (+0.3684)` |
| **Recall (Churn=1)** | `0.8910` | `0.1806` (Prior Match) | `+0.7104` |
| **F1-Score (Churn=1)** | `0.6794` | `0.1806` | `3.76x (+0.4988)` |
| **Macro F1-Score** | `0.7900` | `0.5000` | `+0.2900` |
| **Brier Score** | `0.0880` | `0.1480` (Lower is better) | `-0.0600 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `21,022` | False Positive (FP): `4,044`
- False Negative (FN): `602` | True Positive (TP): `4,923`
- Total Samples: `30,591` (Positive: `5,525`, Negative: `25,066`)

### Top-K Ranking & Lift Breakdown (XGBOOST - Test Set (Optimized Threshold = 0.4060))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `305` | `233 / 5,525` | `76.39%` | `4.22%` | `4.23x` |
| **Top 2%** | `611` | `442 / 5,525` | `72.34%` | `8.00%` | `4.01x` |
| **Top 5%** | `1,529` | `1,069 / 5,525` | `69.91%` | `19.35%` | `3.87x` |
| **Top 10%** | `3,059` | `2,007 / 5,525` | `65.61%` | `36.33%` | `3.63x` |
| **Top 20%** | `6,118` | `3,668 / 5,525` | `59.95%` | `66.39%` | `3.32x` |