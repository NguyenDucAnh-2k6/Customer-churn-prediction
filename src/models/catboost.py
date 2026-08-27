"""
CatBoost Model Wrapper with Optuna Hyperparameter Optimization.
"""

from typing import Any, Dict, List, Optional
import joblib
from catboost import CatBoostClassifier
import numpy as np
import optuna
import pandas as pd

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances


class CatBoostModelWrapper(BaseModelWrapper):
    """Wrapper for CatBoost Classifier with adaptive scale_pos_weight and HPO."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        max_scale = max(5.0, min(20.0, split_result.scale_pos_weight_estimate * 1.2))

        return {
            "iterations": trial.suggest_int("iterations", 100, 800, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_scale),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": seed,
            "verbose": False,
            "thread_count": -1,
            "early_stopping_rounds": 30,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> CatBoostClassifier:
        p = params.copy()
        p.setdefault("random_seed", seed)
        p.setdefault("verbose", False)
        p.setdefault("thread_count", -1)
        return CatBoostClassifier(**p)

    def fit(
        self,
        model: CatBoostClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CatBoostClassifier:
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
                **fit_kwargs,
            )
            return model

        model.fit(X_train, y_train, verbose=False, **fit_kwargs)
        return model

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        try:
            importances = model.get_feature_importance(type="PredictionValuesChange")
            total = importances.sum()
            df = pd.DataFrame({
                "feature": feature_names,
                "gain": importances,
                "gain_ratio": importances / total if total > 0 else 0.0,
                "weight": np.zeros(len(feature_names)),
                "cover": np.zeros(len(feature_names)),
            }).sort_values(by="gain", ascending=False).reset_index(drop=True)
            return df
        except Exception:
            return extract_feature_importances(model, feature_names)

    def save_model(self, model: Any, filepath: str) -> None:
        if filepath.endswith(".cbm"):
            model.save_model(filepath)
        else:
            joblib.dump(model, filepath)
