"""
Logistic Regression Model Wrapper implementation.
"""

from typing import Any, Dict, List, Optional
import joblib
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances


class LogisticRegressionModelWrapper(BaseModelWrapper):
    """Wrapper for Scikit-Learn LogisticRegression baseline."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "max_iter": 1000,
            "random_state": seed,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> LogisticRegression:
        p = params.copy()
        p.setdefault("random_state", seed)
        p.setdefault("max_iter", 1000)
        return LogisticRegression(**p)

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        return extract_feature_importances(model, feature_names)

    def save_model(self, model: Any, filepath: str) -> None:
        joblib.dump(model, filepath)
