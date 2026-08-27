"""
EDA module for dataset exploration, correlation analysis, and visualization.
"""

from src.eda.correlations import (
    detect_multicollinearity,
    plot_correlation_matrix,
    plot_target_correlations,
)
from src.eda.distributions import (
    plot_categorical_churn_rates,
    plot_feature_distributions_by_target,
    plot_target_distribution,
)
from src.eda.report import generate_eda_suite
from src.eda.timeseries_eda import (
    plot_activity_trends_over_time,
    plot_churn_trend_over_time,
)

__all__ = [
    "plot_correlation_matrix",
    "plot_target_correlations",
    "detect_multicollinearity",
    "plot_target_distribution",
    "plot_feature_distributions_by_target",
    "plot_categorical_churn_rates",
    "plot_churn_trend_over_time",
    "plot_activity_trends_over_time",
    "generate_eda_suite",
]
