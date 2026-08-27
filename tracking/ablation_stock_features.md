# 📊 Ablation Study: Impact of Stock / Financial Behavioral Indicators

| Dataset | Configuration | Feature Count | Test ROC-AUC | Test PR-AUC | Precision@Top 5% | Top 5% Lift | Test F1-Score | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TIMESERIES** | `WITH Stock Indicators (Full)` | `52` | `0.9285` | `0.6690` | `72.88%` | `4.00x` | `0.6977` | `0.0824` |
| **TIMESERIES** | `WITHOUT Stock Indicators (Ablated)` | `48` | `0.9278` | `0.6610` | `72.29%` | `3.97x` | `0.6962` | `0.0829` |
| **LATEST** | `WITH Stock Indicators (Full)` | `80` | `0.9187` | `0.6316` | `70.31%` | `3.89x` | `0.6804` | `0.0869` |
| **LATEST** | `WITHOUT Stock Indicators (Ablated)` | `76` | `0.9087` | `0.5955` | `66.19%` | `3.66x` | `0.6635` | `0.0913` |
