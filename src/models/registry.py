"""
Model implementations and Model Registry.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Type
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
        importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
        df = pd.DataFrame({
            "feature": feature_names,
            "gain": importances,
            "gain_ratio": importances / importances.sum() if importances.sum() > 0 else 0.0,
            "weight": np.zeros(len(feature_names)),
            "cover": np.zeros(len(feature_names)),
        }).sort_values(by="gain", ascending=False).reset_index(drop=True)
        return df

    def save_model(self, model: Any, filepath: str) -> None:
        joblib.dump(model, filepath)


class LogisticRegressionModelWrapper(BaseModelWrapper):
    """Wrapper for Scikit-Learn LogisticRegression baseline."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", None]),
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
        p.setdefault("max_iter", 1000)
        p.setdefault("random_state", seed)
        return LogisticRegression(**p)

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        coefs = np.abs(model.coef_[0]) if hasattr(model, "coef_") else np.zeros(len(feature_names))
        df = pd.DataFrame({
            "feature": feature_names,
            "gain": coefs,
            "gain_ratio": coefs / coefs.sum() if coefs.sum() > 0 else 0.0,
            "weight": np.zeros(len(feature_names)),
            "cover": np.zeros(len(feature_names)),
        }).sort_values(by="gain", ascending=False).reset_index(drop=True)
        return df

    def save_model(self, model: Any, filepath: str) -> None:
        joblib.dump(model, filepath)


class ModelRegistry:
    """Registry pattern for discovering and instantiating model wrappers."""
    _registry: Dict[str, Type[BaseModelWrapper]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseModelWrapper]], Type[BaseModelWrapper]]:
        """Decorator to register a new model wrapper class."""
        def decorator(subclass: Type[BaseModelWrapper]) -> Type[BaseModelWrapper]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseModelWrapper:
        """Instantiate a registered model wrapper by name."""
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Model '{name}' is not registered. Available models: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of all registered model names."""
        return sorted(list(cls._registry.keys()))


from src.models.lstm import LSTMModelWrapper
from src.models.lightgbm import LightGBMModelWrapper
from src.models.catboost import CatBoostModelWrapper
from src.models.tabnet import TabNetModelWrapper

# Register built-in models
ModelRegistry.register("xgboost")(XGBoostModelWrapper)
ModelRegistry.register("xgb")(XGBoostModelWrapper)
ModelRegistry.register("lightgbm")(LightGBMModelWrapper)
ModelRegistry.register("lgb")(LightGBMModelWrapper)
ModelRegistry.register("catboost")(CatBoostModelWrapper)
ModelRegistry.register("cb")(CatBoostModelWrapper)
ModelRegistry.register("tabnet")(TabNetModelWrapper)
ModelRegistry.register("tab_net")(TabNetModelWrapper)
ModelRegistry.register("random_forest")(RandomForestModelWrapper)
ModelRegistry.register("rf")(RandomForestModelWrapper)
ModelRegistry.register("logistic_regression")(LogisticRegressionModelWrapper)
ModelRegistry.register("lr")(LogisticRegressionModelWrapper)
ModelRegistry.register("lstm")(LSTMModelWrapper)
ModelRegistry.register("rnn")(LSTMModelWrapper)
