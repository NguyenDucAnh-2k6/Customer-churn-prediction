"""
TabNet Deep Learning Model Wrapper with Optuna Hyperparameter Optimization.
"""

from typing import Any, Dict, List, Optional
import os
import joblib
import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import extract_feature_importances


class TabNetModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch TabNet deep learning tabular architecture."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        n_d = trial.suggest_int("n_d", 8, 48, step=8)
        return {
            "n_d": n_d,
            "n_a": n_d,
            "n_steps": trial.suggest_int("n_steps", 3, 6),
            "gamma": trial.suggest_float("gamma", 1.0, 1.8),
            "n_independent": trial.suggest_int("n_independent", 1, 2),
            "n_shared": trial.suggest_int("n_shared", 1, 2),
            "momentum": trial.suggest_float("momentum", 0.02, 0.3),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
            "virtual_batch_size": 256,
            "max_epochs": 25,
            "patience": 7,
            "seed": seed,
            "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> TabNetClassifier:
        p = params.copy()
        n_d = p.get("n_d", 16)
        n_a = p.get("n_a", n_d)
        n_steps = p.get("n_steps", 4)
        gamma = p.get("gamma", 1.3)
        n_independent = p.get("n_independent", 2)
        n_shared = p.get("n_shared", 2)
        momentum = p.get("momentum", 0.02)
        lambda_sparse = p.get("lambda_sparse", 1e-3)
        device_name = p.get("device_name", "cuda" if torch.cuda.is_available() else "cpu")

        # Save train-loop specific params as model attributes
        model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            lambda_sparse=lambda_sparse,
            seed=seed,
            verbose=0,
            device_name=device_name,
        )
        model._fit_learning_rate = p.get("learning_rate", 0.02)
        model._fit_batch_size = p.get("batch_size", 2048)
        model._fit_virtual_batch_size = p.get("virtual_batch_size", 256)
        model._fit_max_epochs = p.get("max_epochs", 20)
        model._fit_patience = p.get("patience", 5)
        return model

    def _prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values.astype(np.float32)
        # Replace inf and NaN values with column means/zeros
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr

    def fit(
        self,
        model: TabNetClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> TabNetClassifier:
        X_tr = self._prepare_data(X_train)
        y_tr = y_train.values.astype(np.int64)

        lr = getattr(model, "_fit_learning_rate", 0.02)
        batch_size = getattr(model, "_fit_batch_size", 1024)
        virtual_batch_size = getattr(model, "_fit_virtual_batch_size", 128)
        max_epochs = getattr(model, "_fit_max_epochs", 40)
        patience = getattr(model, "_fit_patience", 10)

        weights = sample_weight if sample_weight is not None else 1

        if X_val is not None and y_val is not None:
            X_va = self._prepare_data(X_val)
            y_va = y_val.values.astype(np.int64)
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                eval_set=[(X_va, y_va)],
                eval_name=["val"],
                eval_metric=["auc"],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                weights=weights,
                drop_last=False,
            )
            return model

        model.fit(
            X_train=X_tr,
            y_train=y_tr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights,
            drop_last=False,
        )
        return model

    def predict_proba(self, model: TabNetClassifier, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._prepare_data(X)
        probs = model.predict_proba(X_arr)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        try:
            importances = model.feature_importances_
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
        if filepath.endswith(".zip"):
            model.save_model(filepath.replace(".zip", ""))
        else:
            joblib.dump(model, filepath)
