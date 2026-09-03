"""
Static Dataset implementation for cross-sectional customer dataset (dataset02_fixed.csv).
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.base import BaseDataset, SplitResult


class StaticDataset(BaseDataset):
    """Static cross-sectional dataset (dataset02_fixed.csv) with 1 row per customer.
    
    Split is aligned with train_model_sklearn.ipynb:
    - Test set: 20% holdout (test_size=0.2), random_state=42, stratify=None.
    - Val set: split from the remaining 80% train set for Optuna HPO & threshold tuning.
    """

    def __init__(
        self,
        data_path: str = "data/processed/dataset02_fixed.csv",
        test_size: float = 0.2,
        val_size: float = 0.15,
        stratify: bool = False,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.seed = seed

    def load_and_split(self, **kwargs: Any) -> SplitResult:
        data_path = kwargs.get("data_path", self.data_path)
        test_size = kwargs.get("test_size", self.test_size)
        val_size = kwargs.get("val_size", self.val_size)
        stratify_flag = kwargs.get("stratify", self.stratify)
        seed = kwargs.get("seed", self.seed)

        print(f"[DATASET] Loading static dataset from '{data_path}'...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        print(f"[DATASET] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Exclude index column and target
        exclude_cols = ["Unnamed: 0", "churn"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df["churn"].astype(int)

        strat_y = y if stratify_flag else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat_y
        )

        if val_size > 0:
            strat_val = y_train_full if stratify_flag else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=strat_val
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = X_test, y_test

        result = SplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            metadata={
                "dataset_name": "static",
                "data_path": data_path,
                "split_type": "aligned_with_notebook",
                "test_size": test_size,
                "val_size": val_size,
                "stratify": stratify_flag,
                "seed": seed,
            },
        )
        result.print_summary()
        return result
