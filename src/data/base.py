"""
Base interfaces and data structures for Dataset loading and splitting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Container for split data and associated metadata."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: List[str]
    train_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def train_pos_ratio(self) -> float:
        return float(self.y_train.mean())

    @property
    def val_pos_ratio(self) -> float:
        return float(self.y_val.mean())

    @property
    def test_pos_ratio(self) -> float:
        return float(self.y_test.mean())

    @property
    def scale_pos_weight_estimate(self) -> float:
        neg_count = float((self.y_train == 0).sum())
        pos_count = float((self.y_train == 1).sum())
        return neg_count / pos_count if pos_count > 0 else 1.0

    def print_summary(self) -> None:
        """Print clean summary of the split datasets."""
        print("\n[SPLIT SUMMARY]")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Train Set: {len(self.X_train):,} samples | Positive Rate: {self.train_pos_ratio:.2%} ({int(self.y_train.sum()):,} / {len(self.y_train):,})")
        print(f"  - Val Set:   {len(self.X_val):,} samples | Positive Rate: {self.val_pos_ratio:.2%} ({int(self.y_val.sum()):,} / {len(self.y_val):,})")
        print(f"  - Test Set:  {len(self.X_test):,} samples | Positive Rate: {self.test_pos_ratio:.2%} ({int(self.y_test.sum()):,} / {len(self.y_test):,})")
        print(f"  - Imbalance Scale Pos Weight Estimate: {self.scale_pos_weight_estimate:.2f}")


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def load_and_split(self, **kwargs: Any) -> SplitResult:
        """Load the dataset from disk and split into Train / Val / Test."""
        pass
