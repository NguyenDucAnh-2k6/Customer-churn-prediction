"""
Random Forest Model Wrapper implementation.
"""

from typing import Any, Dict, List, Optional
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances


class RandomForestModelWrapper(BaseModelWrapper):
    """Wrapper for Scikit-Learn RandomForestClassifier baseline."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": class_weight,
            "random_state": seed,
            "n_jobs": -1,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> RandomForestClassifier:
        p = params.copy()
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        return RandomForestClassifier(**p)

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        return extract_feature_importances(model, feature_names)

    def save_model(self, model: Any, filepath: str) -> None:
        joblib.dump(model, filepath)
