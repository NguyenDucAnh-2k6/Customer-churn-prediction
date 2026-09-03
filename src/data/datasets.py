"""
Dataset Orchestrator and Re-export Interface.
This module serves as the central hub for dataset access and maintains backwards compatibility.
"""

from src.data.base import BaseDataset, SplitResult
from src.data.weights import (
    compute_dynamic_sample_weights,
    compute_exponential_decay_weights,
    apply_feature_filters,
    LOW_MI_FEATURES,
    COLLINEAR_PAIRS_DROP,
    STATIC_TIER_FEATURES,
)
from src.data.timeseries import TimeSeriesDataset, GroupStratifiedTimeSeriesDataset
from src.data.static import StaticDataset
from src.data.latest import PreSplitLatestDataset, GroupStratifiedLatestDataset
from src.data.pit import PointInTimeTimeSeriesDataset
from src.data.round3 import Round3Dataset
from src.data.round3_timeseries import Round3TimeSeriesDataset
from src.data.registry import DatasetRegistry

__all__ = [
    "BaseDataset",
    "SplitResult",
    "DatasetRegistry",
    "TimeSeriesDataset",
    "GroupStratifiedTimeSeriesDataset",
    "StaticDataset",
    "PreSplitLatestDataset",
    "GroupStratifiedLatestDataset",
    "PointInTimeTimeSeriesDataset",
    "Round3Dataset",
    "Round3TimeSeriesDataset",
    "compute_dynamic_sample_weights",
    "compute_exponential_decay_weights",
    "apply_feature_filters",
    "LOW_MI_FEATURES",
    "COLLINEAR_PAIRS_DROP",
    "STATIC_TIER_FEATURES",
]
