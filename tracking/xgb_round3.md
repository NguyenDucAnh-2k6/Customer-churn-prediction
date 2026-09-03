### XGBOOST - Test Set (Default Threshold = 0.5000)

| Metric | Model Value | Random Baseline (Prior Rate = 7.70%) | Delta / Lift vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.5000` | `0.5000` | `-` |
| **ROC-AUC** | `0.9896` | `0.5000` | `+0.4896` |
| **PR-AUC (Average Precision)** | `0.9243` | `0.0770` (Base Rate) | `12.00x (+0.8473)` |
| **Precision@Top 5%** | `96.00%` (Lift: `12.47x`) | `7.70%` (Lift: `1.00x`) | `+88.30%` |
| **Recall@Top 10%** | `91.56%` | `10.00%` | `9.16x (+81.56%)` |
| **Accuracy** | `0.9805` | `0.8579` (Prior Match) | `+0.1226` |
| **Precision (Churn=1)** | `0.9021` | `0.0770` | `11.72x (+0.8251)` |
| **Recall (Churn=1)** | `0.8377` | `0.0770` (Prior Match) | `+0.7607` |
| **F1-Score (Churn=1)** | `0.8687` | `0.0770` | `11.28x (+0.7917)` |
| **Macro F1-Score** | `0.9291` | `0.5000` | `+0.4291` |
| **Brier Score** | `0.0165` | `0.0711` (Lower is better) | `-0.0545 (Better)` |

**Confusion Matrix:**
- True Negative (TN): `1,832` | False Positive (FP): `14`
- False Negative (FN): `25` | True Positive (TP): `129`
- Total Samples: `2,000` (Positive: `154`, Negative: `1,846`)

### Top-K Ranking & Lift Breakdown (XGBOOST - Test Set (Default Threshold = 0.5000))

| Decile / Segment | Targeted Users | Captured Churns | Precision@K | Recall@K (Coverage) | Cumulative Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 1%** | `20` | `20 / 154` | `100.00%` | `12.99%` | `12.99x` |
| **Top 2%** | `40` | `40 / 154` | `100.00%` | `25.97%` | `12.99x` |
| **Top 5%** | `100` | `96 / 154` | `96.00%` | `62.34%` | `12.47x` |
| **Top 10%** | `200` | `141 / 154` | `70.50%` | `91.56%` | `9.16x` |
| **Top 20%** | `400` | `153 / 154` | `38.25%` | `99.35%` | `4.97x` |