"""
Univariate Statistical Feature Filtering Module based on Discriminative Power.

Implements three mathematically grounded filtering strategies:
1. KDE Distribution Divergence (2-Sample Kolmogorov-Smirnov test & Wasserstein distance).
2. Categorical Distribution Homogeneity (Chi-squared test, Cramér's V, & Information Value).
3. Boxplot Separation (Cohen's d standardized mean difference & IQR overlap ratio).
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_ks_divergence(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    numeric_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute 2-Sample Kolmogorov-Smirnov (KS) statistic and p-value for numerical features
    comparing class 0 (retained) vs class 1 (churned).

    Parameters:
        X_train: Training feature matrix
        y_train: Binary target series (0: Retained, 1: Churned)
        numeric_features: Optional list of numeric feature names

    Returns:
        pd.DataFrame sorted by KS statistic descending.
    """
    y_vec = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    mask_0 = (y_vec == 0)
    mask_1 = (y_vec == 1)

    if numeric_features is None:
        numeric_features = [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if X_train[c].nunique() > 2
        ]

    results = []
    for col in numeric_features:
        vals_0 = X_train.loc[mask_0, col].dropna().values
        vals_1 = X_train.loc[mask_1, col].dropna().values

        if len(vals_0) < 5 or len(vals_1) < 5:
            results.append({"feature": col, "ks_stat": 0.0, "p_value": 1.0, "wasserstein_dist": 0.0})
            continue

        ks_res = stats.ks_2samp(vals_0, vals_1)
        # Compute normalized Wasserstein Distance
        std_pooled = np.sqrt((np.var(vals_0) + np.var(vals_1)) / 2.0)
        if std_pooled > 1e-9:
            w_dist = stats.wasserstein_distance(vals_0, vals_1) / std_pooled
        else:
            w_dist = 0.0

        results.append({
            "feature": col,
            "ks_stat": float(ks_res.statistic),
            "p_value": float(ks_res.pvalue),
            "wasserstein_dist": float(w_dist),
        })

    df_ks = pd.DataFrame(results).sort_values(by="ks_stat", ascending=False).reset_index(drop=True)
    return df_ks


def filter_features_by_kde(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    ks_threshold: float = 0.05,
    verbose: bool = True,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Filter out continuous features where KDE distributions of class 0 vs 1 are virtually identical
    (KS statistic < ks_threshold).

    Parameters:
        X_train: Training feature matrix
        y_train: Target
        ks_threshold: Minimum KS statistic required to keep feature (default: 0.05)
        verbose: Whether to log results

    Returns:
        Tuple of (retained_features, dropped_features, summary_df)
    """
    df_ks = compute_ks_divergence(X_train, y_train)
    
    dropped = df_ks[df_ks["ks_stat"] < ks_threshold]["feature"].tolist()
    retained = [c for c in X_train.columns if c not in dropped]

    if verbose:
        print(f"\n[KDE STATISTICAL FILTER] Evaluated {len(df_ks)} continuous features (Threshold D_KS >= {ks_threshold}):")
        print(f"  - Retained: {len(retained)} features")
        print(f"  - Dropped: {len(dropped)} low-divergence features with nearly identical KDE curves")
        if dropped:
            print(f"  - Dropped List: {', '.join(dropped)}")

    return retained, dropped, df_ks


def compute_categorical_metrics(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    cat_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Chi-squared statistic, Cramér's V, and Information Value (IV) for categorical
    and discrete low-cardinality features.

    Parameters:
        X_train: Training feature matrix
        y_train: Target
        cat_features: Optional list of categorical column names

    Returns:
        pd.DataFrame sorted by Cramér's V descending.
    """
    y_vec = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)

    if cat_features is None:
        cat_features = [
            c for c in X_train.columns
            if X_train[c].dtype == object
            or str(X_train[c].dtype) == "category"
            or X_train[c].nunique() <= 20
        ]

    total_pos = np.sum(y_vec == 1)
    total_neg = np.sum(y_vec == 0)

    results = []
    for col in cat_features:
        series_col = X_train[col].fillna("Missing").astype(str)
        contingency = pd.crosstab(series_col, y_vec)

        if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            results.append({
                "feature": col,
                "cramers_v": 0.0,
                "chi2_stat": 0.0,
                "p_value": 1.0,
                "information_value": 0.0,
            })
            continue

        chi2, p, _, _ = stats.chi2_contingency(contingency)
        n = contingency.values.sum()
        r, k = contingency.shape
        cramers_v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 and n > 0 else 0.0

        # Calculate Information Value (IV) & WoE
        iv = 0.0
        for category in contingency.index:
            n_neg = contingency.loc[category, 0] if 0 in contingency.columns else 0
            n_pos = contingency.loc[category, 1] if 1 in contingency.columns else 0

            # Laplace smoothing for zero counts
            pct_neg = (n_neg + 0.5) / (total_neg + 1.0)
            pct_pos = (n_pos + 0.5) / (total_pos + 1.0)

            woe = np.log(pct_neg / pct_pos)
            iv += (pct_neg - pct_pos) * woe

        results.append({
            "feature": col,
            "cramers_v": float(cramers_v),
            "chi2_stat": float(chi2),
            "p_value": float(p),
            "information_value": float(iv),
        })

    df_cat = pd.DataFrame(results).sort_values(by="cramers_v", ascending=False).reset_index(drop=True)
    return df_cat


def filter_features_by_categorical(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    cramers_v_threshold: float = 0.03,
    iv_threshold: float = 0.02,
    verbose: bool = True,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Filter out categorical features where proportions/churn rates across categories are nearly identical
    (Cramér's V < cramers_v_threshold or IV < iv_threshold).

    Parameters:
        X_train: Training feature matrix
        y_train: Target
        cramers_v_threshold: Minimum Cramér's V (default: 0.03)
        iv_threshold: Minimum Information Value (default: 0.02 - "Useless predictor threshold")
        verbose: Whether to log results

    Returns:
        Tuple of (retained_features, dropped_features, summary_df)
    """
    df_cat = compute_categorical_metrics(X_train, y_train)

    # Filter where both Cramér's V and IV are below threshold
    dropped = df_cat[(df_cat["cramers_v"] < cramers_v_threshold) & (df_cat["information_value"] < iv_threshold)]["feature"].tolist()
    retained = [c for c in X_train.columns if c not in dropped]

    if verbose:
        print(f"\n[CATEGORICAL STATISTICAL FILTER] Evaluated {len(df_cat)} categorical/discrete features (Cramer's V >= {cramers_v_threshold} & IV >= {iv_threshold}):")
        print(f"  - Retained: {len(retained)} features")
        print(f"  - Dropped: {len(dropped)} homogeneous categorical features (equal bar heights)")
        if dropped:
            print(f"  - Dropped List: {', '.join(dropped)}")

    return retained, dropped, df_cat


def compute_cohens_d_and_iqr_overlap(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    numeric_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Cohen's d (standardized mean difference) and IQR Overlap Ratio
    comparing class 0 (retained) vs class 1 (churned).

    Parameters:
        X_train: Training feature matrix
        y_train: Target
        numeric_features: Optional list of numeric feature names

    Returns:
        pd.DataFrame sorted by Cohen's d descending.
    """
    y_vec = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    mask_0 = (y_vec == 0)
    mask_1 = (y_vec == 1)

    if numeric_features is None:
        numeric_features = [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if X_train[c].nunique() > 2
        ]

    results = []
    for col in numeric_features:
        vals_0 = X_train.loc[mask_0, col].dropna().values
        vals_1 = X_train.loc[mask_1, col].dropna().values

        if len(vals_0) < 5 or len(vals_1) < 5:
            results.append({
                "feature": col,
                "cohens_d": 0.0,
                "iqr_overlap_ratio": 1.0,
                "median_diff_norm": 0.0,
            })
            continue

        m0, m1 = np.mean(vals_0), np.mean(vals_1)
        v0, v1 = np.var(vals_0, ddof=1), np.var(vals_1, ddof=1)
        n0, n1 = len(vals_0), len(vals_1)

        # Pooled standard deviation
        pooled_var = ((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2) if (n0 + n1 - 2) > 0 else 0.0
        pooled_std = np.sqrt(pooled_var)

        cohens_d = abs(m0 - m1) / pooled_std if pooled_std > 1e-9 else 0.0

        # IQR Overlap calculation
        q1_0, q3_0 = np.percentile(vals_0, 25), np.percentile(vals_0, 75)
        q1_1, q3_1 = np.percentile(vals_1, 25), np.percentile(vals_1, 75)

        iqr_0 = q3_0 - q1_0
        iqr_1 = q3_1 - q1_1
        min_iqr = min(iqr_0, iqr_1)

        overlap_start = max(q1_0, q1_1)
        overlap_end = min(q3_0, q3_1)
        overlap_len = max(0.0, overlap_end - overlap_start)

        iqr_overlap_ratio = (overlap_len / min_iqr) if min_iqr > 1e-9 else (1.0 if abs(q1_0 - q1_1) < 1e-9 else 0.0)

        # Median Difference normalized by pooled std
        med_0, med_1 = np.median(vals_0), np.median(vals_1)
        med_diff_norm = abs(med_0 - med_1) / pooled_std if pooled_std > 1e-9 else 0.0

        results.append({
            "feature": col,
            "cohens_d": float(cohens_d),
            "iqr_overlap_ratio": float(iqr_overlap_ratio),
            "median_diff_norm": float(med_diff_norm),
        })

    df_box = pd.DataFrame(results).sort_values(by="cohens_d", ascending=False).reset_index(drop=True)
    return df_box


def filter_features_by_boxplot(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    cohens_d_threshold: float = 0.08,
    iqr_overlap_threshold: float = 0.90,
    verbose: bool = True,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Filter out continuous features where Mean and IQR of class 0 vs 1 are virtually identical
    (Cohen's d < cohens_d_threshold and IQR Overlap > iqr_overlap_threshold).

    Parameters:
        X_train: Training feature matrix
        y_train: Target
        cohens_d_threshold: Minimum Cohen's d (default: 0.08)
        iqr_overlap_threshold: Maximum IQR overlap ratio (default: 0.90)
        verbose: Whether to log results

    Returns:
        Tuple of (retained_features, dropped_features, summary_df)
    """
    df_box = compute_cohens_d_and_iqr_overlap(X_train, y_train)

    # Filter features where mean difference is tiny AND IQR overlap is huge
    dropped = df_box[(df_box["cohens_d"] < cohens_d_threshold) & (df_box["iqr_overlap_ratio"] > iqr_overlap_threshold)]["feature"].tolist()
    retained = [c for c in X_train.columns if c not in dropped]

    if verbose:
        print(f"\n[BOXPLOT STATISTICAL FILTER] Evaluated {len(df_box)} continuous features (Cohen's d >= {cohens_d_threshold} | IQR Overlap <= {iqr_overlap_threshold}):")
        print(f"  - Retained: {len(retained)} features")
        print(f"  - Dropped: {len(dropped)} overlapping boxplot features (identical mean/IQR)")
        if dropped:
            print(f"  - Dropped List: {', '.join(dropped)}")

    return retained, dropped, df_box


def generate_univariate_screening_report(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    ks_threshold: float = 0.05,
    cramers_v_threshold: float = 0.03,
    iv_threshold: float = 0.02,
    cohens_d_threshold: float = 0.08,
    iqr_overlap_threshold: float = 0.90,
) -> pd.DataFrame:
    """Generate a comprehensive diagnostic DataFrame summarizing all 3 statistical filtering
    criteria across all features in X_train."""
    df_ks = compute_ks_divergence(X_train, y_train).set_index("feature")
    df_cat = compute_categorical_metrics(X_train, y_train).set_index("feature")
    df_box = compute_cohens_d_and_iqr_overlap(X_train, y_train).set_index("feature")

    all_features = list(X_train.columns)
    records = []

    for f in all_features:
        dtype_str = str(X_train[f].dtype)
        is_numeric = f in df_ks.index

        ks_stat = df_ks.loc[f, "ks_stat"] if is_numeric else np.nan
        p_val_ks = df_ks.loc[f, "p_value"] if is_numeric else np.nan
        cohens_d = df_box.loc[f, "cohens_d"] if is_numeric else np.nan
        iqr_overlap = df_box.loc[f, "iqr_overlap_ratio"] if is_numeric else np.nan

        is_cat = f in df_cat.index
        cramers_v = df_cat.loc[f, "cramers_v"] if is_cat else np.nan
        iv = df_cat.loc[f, "information_value"] if is_cat else np.nan

        # Evaluation verdicts
        kde_pass = (ks_stat >= ks_threshold) if pd.notna(ks_stat) else True
        cat_pass = ((cramers_v >= cramers_v_threshold) or (iv >= iv_threshold)) if pd.notna(cramers_v) else True
        box_pass = ((cohens_d >= cohens_d_threshold) or (iqr_overlap <= iqr_overlap_threshold)) if pd.notna(cohens_d) else True

        records.append({
            "feature": f,
            "dtype": dtype_str,
            "ks_stat": ks_stat,
            "p_val_ks": p_val_ks,
            "cohens_d": cohens_d,
            "iqr_overlap_ratio": iqr_overlap,
            "cramers_v": cramers_v,
            "information_value": iv,
            "kde_verdict": "PASS" if kde_pass else "DROP",
            "cat_verdict": "PASS" if cat_pass else "DROP",
            "boxplot_verdict": "PASS" if box_pass else "DROP",
            "overall_signal": "STRONG" if (kde_pass and cat_pass and box_pass) else "WEAK",
        })

    report_df = pd.DataFrame(records)
    return report_df
