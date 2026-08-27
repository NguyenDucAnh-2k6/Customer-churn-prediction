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

__all__ = [
    "DataCleaningTransformer",
    "VelocityFeatureGenerator",
    "FinancialFeatureGenerator",
    "ChurnFeaturePreprocessor",
    "make_mi_scores",
    "filter_features_by_mi",
    "filter_multicollinear_features",
]
