"""
XGBoost Model Wrapper implementation.
"""

from typing import Any, Dict, List, Optional
import joblib
import optuna
import pandas as pd
from xgboost import XGBClassifier

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances


class XGBoostModelWrapper(BaseModelWrapper):
    """Wrapper for XGBoost classifier with adaptive scale_pos_weight search."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        # Estimate upper bound for scale_pos_weight based on dataset imbalance
        max_scale = max(5.0, min(20.0, split_result.scale_pos_weight_estimate * 1.2))

        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_scale),
            "tree_method": "hist",
            "random_state": seed,
            "eval_metric": "logloss",
            "early_stopping_rounds": 30,
            "n_jobs": -1,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> XGBClassifier:
        p = params.copy()
        p.setdefault("tree_method", "hist")
        p.setdefault("random_state", seed)
        p.setdefault("eval_metric", "logloss")
        p.setdefault("early_stopping_rounds", 30)
        p.setdefault("n_jobs", -1)
        return XGBClassifier(**p)

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        return extract_feature_importances(model, feature_names)

    def save_model(self, model: Any, filepath: str) -> None:
        if filepath.endswith(".json"):
            model.save_model(filepath)
        else:
            joblib.dump(model, filepath)
