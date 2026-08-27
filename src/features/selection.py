"""
Feature selection module based on Mutual Information (MI) and Multicollinearity.
Implements dynamic MI calculation similar to Untitled.ipynb (make_mi_scores).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def make_mi_scores(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    discrete_features: Optional[Union[List[bool], np.ndarray]] = None,
    method: str = "regression",
    random_state: int = 42,
    max_samples: int = 50000,
) -> pd.Series:
    """Calculate Mutual Information scores for features with respect to target y.
    
    Replicates the exact `make_mi_scores` logic from Untitled.ipynb with support for
    both `mutual_info_regression` and `mutual_info_classif`.

    Parameters:
        X: Feature matrix DataFrame
        y: Target series or array
        discrete_features: Boolean mask or list indicating discrete/integer features
        method: 'regression' (used in Untitled.ipynb) or 'classif'
        random_state: Random seed for reproducibility
        max_samples: Subsample limit for fast computation on large datasets (>50k rows)

    Returns:
        pd.Series of Mutual Information scores sorted in descending order.
    """
    X_mat = X.copy()

    # Subsample if dataset is very large to speed up nearest neighbors MI estimation
    if len(X_mat) > max_samples:
        idx_sample = np.random.RandomState(random_state).choice(len(X_mat), size=max_samples, replace=False)
        X_mat = X_mat.iloc[idx_sample].copy()
        if isinstance(y, pd.Series):
            y_vec = y.iloc[idx_sample].values
        else:
            y_vec = y[idx_sample]
    else:
        y_vec = y.values if isinstance(y, pd.Series) else y

    # Encode categorical or boolean columns to numeric
    for col in X_mat.columns:
        if X_mat[col].dtype == object or str(X_mat[col].dtype) == "category" or X_mat[col].dtype == bool:
            X_mat[col] = pd.factorize(X_mat[col])[0]

    # Fill NaNs with 0 for MI computation
    X_mat = X_mat.fillna(0)

    # Detect discrete features if not explicitly provided
    if discrete_features is None:
        discrete_features = [pd.api.types.is_integer_dtype(X_mat[col].dtype) for col in X_mat.columns]

    if method == "regression":
        mi_scores = mutual_info_regression(
            X_mat, y_vec, discrete_features=discrete_features, random_state=random_state
        )
    else:
        mi_scores = mutual_info_classif(
            X_mat, y_vec, discrete_features=discrete_features, random_state=random_state
        )

    mi_series = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)
    return mi_series


def filter_features_by_mi(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    mi_threshold: float = 0.001,
    top_k: Optional[int] = None,
    method: str = "classif",
    verbose: bool = True,
) -> Tuple[List[str], pd.Series]:
    """Filter features by Mutual Information scores computed dynamically on training data.

    Parameters:
        X_train: Training feature DataFrame
        y_train: Training target
        mi_threshold: Minimum MI score required to retain a feature
        top_k: Optional number of top MI features to retain
        method: 'regression' or 'classif'
        verbose: If True, prints retained and dropped features table

    Returns:
        Tuple of (retained_feature_names, mi_scores_series)
    """
    mi_scores = make_mi_scores(X_train, y_train, method=method)

    if top_k is not None:
        retained = mi_scores.head(top_k).index.tolist()
        dropped = mi_scores.iloc[top_k:].index.tolist()
    else:
        retained = mi_scores[mi_scores >= mi_threshold].index.tolist()
        dropped = mi_scores[mi_scores < mi_threshold].index.tolist()

    if verbose:
        print(f"\n[MUTUAL INFORMATION] Calculated MI on {len(X_train):,} rows across {len(X_train.columns)} features (Method: {method})")
        print(f"[MUTUAL INFORMATION] Retained: {len(retained)} features | Dropped: {len(dropped)} low-MI features (Threshold >= {mi_threshold})")
        if dropped:
            print(f"[MUTUAL INFORMATION] Dropped Features: {', '.join(dropped)}")

    return retained, mi_scores


def filter_multicollinear_features(
    X_train: pd.DataFrame,
    threshold: float = 0.90,
    verbose: bool = True,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """Filter highly correlated duplicate features (|r| >= threshold)."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_matrix = X_train[numeric_cols].corr().abs()

    dropped = set()
    pairs = []

    for i in range(len(numeric_cols)):
        col_i = numeric_cols[i]
        if col_i in dropped:
            continue
        for j in range(i + 1, len(numeric_cols)):
            col_j = numeric_cols[j]
            if col_j in dropped:
                continue
            r = corr_matrix.loc[col_i, col_j]
            if r >= threshold:
                dropped.add(col_j)
                pairs.append((col_i, col_j, float(r)))

    retained = [col for col in X_train.columns if col not in dropped]

    if verbose and pairs:
        print(f"\n[MULTICOLLINEARITY] Dropped {len(dropped)} collinear features with |r| >= {threshold}:")
        for c1, c2, r in pairs:
            print(f"  - Dropped '{c2}' (correlated with '{c1}', r={r:.4f})")

    return retained, pairs
