# Walkthrough - Hybrid Preprocessing Module (Static + Time-Series Integration)

We have built a dedicated preprocessing and feature engineering module [`src/features/preprocessor.py`](file:///d:/ML_intern/src/features/preprocessor.py) that seamlessly enriches time-series snapshots with static lifetime metrics, dynamic velocity ratios, and categorical cleaning.

---

## 🛠️ Changes Implemented

### 1. `ChurnFeaturePreprocessor` ([`src/features/preprocessor.py`](file:///d:/ML_intern/src/features/preprocessor.py))
- **Static Master Enrichment:** Left joins customer lifetime statistics from [`data/churn_ml_dataset.csv`](file:///d:/ML_intern/data/churn_ml_dataset.csv) (`total_spent`, `completed_orders`, `total_support_tickets`, `total_usage_sessions`, `mkt_open_rate`, etc.).
- **Dynamic Velocity & Share Ratios (Static $\times$ TimeSeries):**
  - `usage_drop_ratio_3m`: $30\text{d events} / (\text{roll3m events} / 3.0 + 1.0)$
  - `session_duration_drop_ratio_3m`: $30\text{d duration} / (\text{roll3m mean} + 1.0)$
  - `active_days_share_90d`: $30\text{d active days} / (90\text{d active days} + 1.0)$
  - `orders_share_90d`: $30\text{d orders} / (\text{roll3m orders} + 1.0)$
  - `usage_duration_change`: $30\text{d duration} - \text{roll3m mean duration}$
  - `activity_acceleration`: $\text{activity\_slope\_3m} \times \text{usage\_trend\_30d}$
  - `usage_30d_share_lifetime`: $30\text{d events} / (\text{total\_usage\_sessions} + 1.0)$
  - `spent_30d_share_lifetime`: $\text{avg\_spend\_to\_date\_per\_month} / (\text{total\_spent} + 1.0)$
- **Categorical & Boolean Standardization:**
  - `is_declining_engagement`, `reactivation_flag`, `has_marketing_click_30d`, `has_unresolved_ticket`, `is_paid_tier` $\to$ integer flags $\{0, 1\}$.
  - `auto_renew`: Cleaned to separate Free users ($-1$) from Paid auto-renew ($1$) and Paid manual ($0$).
  - `subscription_tier`: Ordinal mapping ($\text{Free}: 0, \text{Plus}: 1, \text{Premium}: 2$).
- **Missing Value Indicator Generation:**
  - Generated `payments_success_rate_missing`, `session_duration_trend_missing`, `avg_csat_score_missing`.

### 2. Integration into Dataset Pipelines ([`src/data/datasets.py`](file:///d:/ML_intern/src/data/datasets.py))
- Integrated `ChurnFeaturePreprocessor` automatically into both `PreSplitLatestDataset` and `TimeSeriesDataset`.

---

## 📊 Verification & Experimental Results

Ran 5-fold Walk-Forward Time-Series Cross Validation on both datasets:

| Dataset | Split / CV Strategy | OOF ROC-AUC | OOF PR-AUC | Test Top 1% Lift | Test Top 10% Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`timeseries` (Hybrid Enriched)** | 5-Fold Walk-Forward | **`0.8882`** | **`0.6374`** | **`25.66x`** | **`58.60%`** |
| **`latest` (Hybrid Enriched)** | 5-Fold Walk-Forward | **`0.8778`** | **`0.6163`** | `2.81x` | `28.95%` |

### Top 10 Important Hybrid Features:
1. `auto_renew` ($28.33\%$)
2. `subscription_tier` ($27.37\%$)
3. `is_paid_tier` ($25.45\%$)
4. `num_usage_events_roll3m_sum` ($2.21\%$)
5. **`usage_drop_ratio_3m`** ($1.92\%$) — *New Hybrid Velocity Ratio*
6. `total_active_days_90d` ($0.90\%$)
7. `activity_slope_3m` ($0.51\%$)
8. `num_usage_events_30d_lag1m` ($0.47\%$)
9. `avg_session_duration_roll3m_mean` ($0.43\%$)
10. **`session_duration_drop_ratio_3m`** ($0.37\%$) — *New Hybrid Velocity Ratio*
