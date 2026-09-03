"""
Features package containing modular preprocessors, financial indicators, velocity ratios, and feature selection tools.
"""

from src.features.cleaning import DataCleaningTransformer
from src.features.velocity import VelocityFeatureGenerator
from src.features.financial_indicators import FinancialFeatureGenerator
from src.features.preprocessor import ChurnFeaturePreprocessor
from src.features.selection import (
    make_mi_scores,
    filter_features_by_mi,
    filter_multicollinear_features,
)
from src.features.statistical_filtering import (
    compute_ks_divergence,
    filter_features_by_kde,
    compute_categorical_metrics,
    filter_features_by_categorical,
    compute_cohens_d_and_iqr_overlap,
    filter_features_by_boxplot,
    generate_univariate_screening_report,
)

__all__ = [
    "DataCleaningTransformer",
    "VelocityFeatureGenerator",
    "FinancialFeatureGenerator",
    "ChurnFeaturePreprocessor",
    "make_mi_scores",
    "filter_features_by_mi",
    "filter_multicollinear_features",
    "compute_ks_divergence",
    "filter_features_by_kde",
    "compute_categorical_metrics",
    "filter_features_by_categorical",
    "compute_cohens_d_and_iqr_overlap",
    "filter_features_by_boxplot",
    "generate_univariate_screening_report",
]

