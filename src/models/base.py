"""
Base interfaces for Model Wrappers and Model Registry.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import optuna
import pandas as pd

from src.data.base import SplitResult


class BaseModelWrapper(ABC):
    """Abstract interface for all registered model types."""

    @abstractmethod
    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Define Optuna hyperparameter search space."""
        pass

    @abstractmethod
    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> Any:
        """Instantiate estimator from hyperparameter dictionary."""
        pass

    def fit(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit model with sample_weight and early stopping on validation set if supported."""
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kwargs)
                return model
            except (TypeError, ValueError):
                pass
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-decision))
        else:
            return model.predict(X).astype(float)

    @abstractmethod
    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Extract and rank feature importances."""
        pass

    @abstractmethod
    def save_model(self, model: Any, filepath: str) -> None:
        """Save model artifact to disk."""
        pass
