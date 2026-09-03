"""
Model Registry for discovering and instantiating machine learning models.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Type

from src.models.base import BaseModelWrapper
from src.models.xgboost_model import XGBoostModelWrapper
from src.models.random_forest import RandomForestModelWrapper
from src.models.logistic_regression import LogisticRegressionModelWrapper
from src.models.lightgbm import LightGBMModelWrapper
from src.models.catboost import CatBoostModelWrapper
from src.models.tabnet import TabNetModelWrapper
from src.models.lstm import LSTMModelWrapper


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


# Register built-in models
ModelRegistry.register("xgboost")(XGBoostModelWrapper)
ModelRegistry.register("xgb")(XGBoostModelWrapper)
ModelRegistry.register("random_forest")(RandomForestModelWrapper)
ModelRegistry.register("rf")(RandomForestModelWrapper)
ModelRegistry.register("logistic_regression")(LogisticRegressionModelWrapper)
ModelRegistry.register("lr")(LogisticRegressionModelWrapper)
ModelRegistry.register("lightgbm")(LightGBMModelWrapper)
ModelRegistry.register("lgbm")(LightGBMModelWrapper)
ModelRegistry.register("catboost")(CatBoostModelWrapper)
ModelRegistry.register("cb")(CatBoostModelWrapper)
ModelRegistry.register("tabnet")(TabNetModelWrapper)
ModelRegistry.register("lstm")(LSTMModelWrapper)
