"""
LightGBM Model Wrapper with Optuna Hyperparameter Optimization.
"""

from typing import Any, Dict, List, Optional
import joblib
from lightgbm import LGBMClassifier
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances
import warnings
warnings.filterwarnings("ignore")

class LightGBMModelWrapper(BaseModelWrapper):
    """Wrapper for LightGBM Classifier with adaptive scale_pos_weight and HPO."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        max_scale = max(5.0, min(20.0, split_result.scale_pos_weight_estimate * 1.2))

        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_scale),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> LGBMClassifier:
        p = params.copy()
        p.setdefault("random_state", seed)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbose", -1)
        return LGBMClassifier(**p)

    def fit(
        self,
        model: LGBMClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> LGBMClassifier:
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
                **fit_kwargs,
            )
            return model

        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        try:
            importances = model.booster_.feature_importance(importance_type="gain")
            total = importances.sum()
            df = pd.DataFrame({
                "feature": feature_names,
                "gain": importances,
                "gain_ratio": importances / total if total > 0 else 0.0,
                "weight": model.booster_.feature_importance(importance_type="split"),
                "cover": np.zeros(len(feature_names)),
            }).sort_values(by="gain", ascending=False).reset_index(drop=True)
            return df
        except Exception:
            return extract_feature_importances(model, feature_names)

    def save_model(self, model: Any, filepath: str) -> None:
        if filepath.endswith(".txt") and hasattr(model, "booster_"):
            model.booster_.save_model(filepath)
        else:
            joblib.dump(model, filepath)
