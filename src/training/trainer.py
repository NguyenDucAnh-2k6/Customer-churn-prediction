"""
Unified Optuna Trainer for hyperparameter optimization, threshold tuning, and artifact logging.
"""

import json
import os
import time
from typing import Any, Dict, Optional
import optuna
import pandas as pd

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper
from src.models.evaluate import (
    calculate_metrics,
    find_best_threshold,
    format_metrics_table,
    plot_feature_importances,
)


from sklearn.model_selection import StratifiedKFold
import numpy as np


class OptunaTrainer:
    """Orchestrates HPO with Optuna, training the best model, threshold tuning, and logging."""

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        split_result: SplitResult,
        db_url: str = "sqlite:///tracking/optuna_study.db",
        study_name: str = "optuna_study",
        metric_name: str = "roc_auc",
        artifacts_dir: str = "src/models/artifacts",
        model_name: str = "model",
        cv_folds: Optional[int] = None,
        seed: int = 42,
    ):
        self.model_wrapper = model_wrapper
        self.split_result = split_result
        self.db_url = db_url
        self.study_name = study_name
        self.metric_name = metric_name
        self.artifacts_dir = artifacts_dir
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.seed = seed

        metric_map = {
            "roc_auc": "roc_auc",
            "auc": "roc_auc",
            "pr_auc": "pr_auc",
            "ap": "pr_auc",
            "precision_at_5": "precision_top_5_pct",
            "precision@5": "precision_top_5_pct",
            "precision_top_5_pct": "precision_top_5_pct",
            "recall_at_10": "recall_top_10_pct",
            "recall@10": "recall_top_10_pct",
            "recall_top_10_pct": "recall_top_10_pct",
            "lift_top_5": "lift_top_5_pct",
            "lift_at_5": "lift_top_5_pct",
            "f1": "f1",
            "accuracy": "accuracy",
        }
        self.metric_key = metric_map.get(self.metric_name.lower(), self.metric_name.lower())

    def _get_or_create_study(self) -> optuna.Study:
        """Connect to SQLite storage and create or load Optuna study."""
        if self.db_url.startswith("sqlite:///"):
            db_file = self.db_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)

        print(f"\n[OPTUNA] Initializing Study in DB: '{self.db_url}' (Study: '{self.study_name}')")
        optuna.logging.set_verbosity(optuna.logging.INFO)
        storage = optuna.storages.RDBStorage(url=self.db_url)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"[OPTUNA] Completed trials in study: {completed}")
        return study

    def _get_cv_splits(self):
        """Generate CV splits: StratifiedGroupKFold if group-stratified, Walk-Forward TimeSeriesSplit if temporal timeseries, else StratifiedKFold."""
        dataset_name = self.split_result.metadata.get("dataset_name", "")
        split_strategy = self.split_result.metadata.get("split_strategy", "")
        split_type = self.split_result.metadata.get("split_type", "")
        train_groups = self.split_result.metadata.get("train_groups")
        train_time_series = self.split_result.metadata.get("train_time_series")

        # 1. Customer-Level Stratified Group CV (Zero Customer Leakage)
        is_group_stratified = (
            split_strategy in ["group_stratified", "stratified_group", "customer_stratified"]
            or split_type in ["group_stratified", "group_stratified_cv", "group_stratified_holdout"]
            or (train_groups is not None and len(train_groups) == len(self.split_result.X_train))
        )

        if is_group_stratified:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            splits = list(sgkf.split(self.split_result.X_train, self.split_result.y_train, groups=train_groups))
            unique_custs = np.unique(train_groups)
            print(f"\n[CROSS-VALIDATION] Setting up {self.cv_folds}-Fold Customer-Stratified Group Validation (Zero Customer Leakage):")
            print(f"  Total Historical Pool: {len(self.split_result.X_train):,} samples across {len(unique_custs):,} unique customers")
            print("  " + "-" * 75)
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                tr_pos_rate = self.split_result.y_train.iloc[tr_idx].mean() * 100
                va_pos_rate = self.split_result.y_train.iloc[va_idx].mean() * 100
                tr_c = len(np.unique(train_groups[tr_idx]))
                va_c = len(np.unique(train_groups[va_idx]))
                print(f"  Fold {fold_idx+1}: Train {len(tr_idx):6,d} rows ({tr_c:5,d} custs, Churn: {tr_pos_rate:5.2f}%) ──► Val {len(va_idx):6,d} rows ({va_c:5,d} custs, Churn: {va_pos_rate:5.2f}%)")
            print("  " + "-" * 75)
            return splits, f"{self.cv_folds}-Fold Customer-Stratified Group CV"

        # 2. Time-Series: Walk-Forward Validation (Expanding Window on unique sorted timestamps)
        is_timeseries_snapshots = (
            dataset_name in ["timeseries", "latest", "presplit", "churn_team"]
            and train_time_series is not None
            and len(train_time_series) == len(self.split_result.X_train)
        )

        if is_timeseries_snapshots:
            from sklearn.model_selection import TimeSeriesSplit
            unique_times = np.sort(np.unique(train_time_series))
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            splits = []
            print(f"\n[CROSS-VALIDATION] Setting up {self.cv_folds}-Fold Walk-Forward TimeSeries Validation:")
            print(f"  Total Historical Pool: {len(self.split_result.X_train):,} samples across {len(unique_times)} snapshot months ({unique_times[0]} to {unique_times[-1]})")
            print("  " + "-" * 75)
            for fold_idx, (train_t_idx, val_t_idx) in enumerate(tscv.split(unique_times)):
                tr_times = unique_times[train_t_idx]
                va_times = unique_times[val_t_idx]
                tr_idx = np.where(np.isin(train_time_series, tr_times))[0]
                va_idx = np.where(np.isin(train_time_series, va_times))[0]
                splits.append((tr_idx, va_idx))

                tr_pos_rate = self.split_result.y_train.iloc[tr_idx].mean() * 100
                va_pos_rate = self.split_result.y_train.iloc[va_idx].mean() * 100
                print(f"  Fold {fold_idx+1}: Train [{tr_times[0]}..{tr_times[-1]}] {len(tr_idx):6,d} rows (Churn: {tr_pos_rate:5.2f}%) ──► Val [{va_times[0]}..{va_times[-1]}] {len(va_idx):6,d} rows (Churn: {va_pos_rate:5.2f}%)")
            print("  " + "-" * 75)
            return splits, f"{self.cv_folds}-Fold Walk-Forward TimeSeries CV"
        else:
            # Static & Point-in-Time: Stratified K-Fold
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            splits = list(skf.split(self.split_result.X_train, self.split_result.y_train))
            print(f"\n[CROSS-VALIDATION] Setting up {self.cv_folds}-Fold Stratified Cross-Validation:")
            print(f"  Total Train Pool: {len(self.split_result.X_train):,} samples")
            print("  " + "-" * 65)
            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                tr_pos_rate = self.split_result.y_train.iloc[tr_idx].mean() * 100
                va_pos_rate = self.split_result.y_train.iloc[va_idx].mean() * 100
                print(f"  Fold {fold_idx+1}: Train {len(tr_idx):5,d} rows (Churn: {tr_pos_rate:5.2f}%) ──► Val {len(va_idx):5,d} rows (Churn: {va_pos_rate:5.2f}%)")
            print("  " + "-" * 65)
            return splits, f"{self.cv_folds}-Fold Stratified CV"

    def run_hpo(self, n_trials: int = 30, timeout: Optional[int] = None) -> optuna.Study:
        """Run Optuna hyperparameter optimization trials."""
        study = self._get_or_create_study()
        splits, split_label = (self._get_cv_splits() if (self.cv_folds and self.cv_folds > 1) else (None, ""))

        def objective(trial: optuna.Trial) -> float:
            params = self.model_wrapper.suggest_hyperparameters(
                trial=trial,
                split_result=self.split_result,
                seed=self.seed,
            )

            # Option A: Cross-Validation on (X_train, y_train) (Walk-Forward or Stratified)
            if splits is not None:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    X_tr = self.split_result.X_train.iloc[train_idx]
                    y_tr = self.split_result.y_train.iloc[train_idx]
                    X_va = self.split_result.X_train.iloc[val_idx]
                    y_va = self.split_result.y_train.iloc[val_idx]
                    w_tr = (
                        self.split_result.train_weights[train_idx]
                        if self.split_result.train_weights is not None
                        else None
                    )

                    fold_model = self.model_wrapper.build_model(params, seed=self.seed + fold_idx)
                    self.model_wrapper.fit(
                        model=fold_model,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_val=X_va,
                        y_val=y_va,
                        sample_weight=w_tr,
                    )
                    y_va_probs = self.model_wrapper.predict_proba(fold_model, X_va)
                    fold_metrics = calculate_metrics(y_va.values, y_va_probs, threshold=0.5)
                    fold_scores.append(fold_metrics.get(self.metric_key, 0.0))

                return float(np.mean(fold_scores))

            # Option B: Single Validation Set (e.g. Out-of-Time Val for Time-Series)
            model = self.model_wrapper.build_model(params, seed=self.seed)
            self.model_wrapper.fit(
                model=model,
                X_train=self.split_result.X_train,
                y_train=self.split_result.y_train,
                X_val=self.split_result.X_val,
                y_val=self.split_result.y_val,
                sample_weight=self.split_result.train_weights,
            )

            y_val_probs = self.model_wrapper.predict_proba(model, self.split_result.X_val)
            metrics = calculate_metrics(self.split_result.y_val.values, y_val_probs, threshold=0.5)
            target_score = metrics.get(self.metric_key, 0.0)
            return float(target_score)

        cv_info = f" with {split_label}" if (self.cv_folds and self.cv_folds > 1) else ""
        print(f"\n[OPTUNA] Running {n_trials} Optimization Trials (Metric: {self.metric_name.upper()}{cv_info})...")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        print("\n[OPTUNA] Optimization Finished!")
        print(f"[OPTUNA] Best Trial #{study.best_trial.number}:")
        print(f"  - Score ({self.metric_name.upper()}): {study.best_value:.5f}")
        print("  - Parameters:")
        for k, v in study.best_params.items():
            print(f"    * {k}: {v}")

        return study

    def train_and_evaluate_best(self, study: optuna.Study) -> Dict[str, Any]:
        """Train final model with optimal hyperparameters, evaluate and save artifacts."""
        os.makedirs(self.artifacts_dir, exist_ok=True)
        best_params = study.best_params.copy()

        # Step 1: Validation / OOF evaluation & threshold tuning
        if self.cv_folds and self.cv_folds > 1:
            splits, split_label = self._get_cv_splits()
            print(f"\n[EVAL] Generating Out-Of-Fold (OOF) Predictions ({split_label})...")
            oof_probs = np.full(len(self.split_result.X_train), np.nan)

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_tr = self.split_result.X_train.iloc[train_idx]
                y_tr = self.split_result.y_train.iloc[train_idx]
                X_va = self.split_result.X_train.iloc[val_idx]
                y_va = self.split_result.y_train.iloc[val_idx]
                w_tr = (
                    self.split_result.train_weights[train_idx]
                    if self.split_result.train_weights is not None
                    else None
                )

                fold_model = self.model_wrapper.build_model(best_params, seed=self.seed + fold_idx)
                self.model_wrapper.fit(
                    model=fold_model,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_val=X_va,
                    y_val=y_va,
                    sample_weight=w_tr,
                )
                oof_probs[val_idx] = self.model_wrapper.predict_proba(fold_model, X_va)

            valid_oof_mask = ~np.isnan(oof_probs)
            val_true = self.split_result.y_train.values[valid_oof_mask]
            val_pred = oof_probs[valid_oof_mask]

            val_metrics_default = calculate_metrics(val_true, val_pred, threshold=0.5)
            best_th, best_val_f1 = find_best_threshold(val_true, val_pred, metric="f1")
            val_metrics_tuned = calculate_metrics(val_true, val_pred, threshold=best_th)
            val_title_suffix = f"{split_label} (Out-Of-Fold)"
        else:
            # Single Validation Set evaluation
            temp_model = self.model_wrapper.build_model(best_params, seed=self.seed)
            self.model_wrapper.fit(
                model=temp_model,
                X_train=self.split_result.X_train,
                y_train=self.split_result.y_train,
                X_val=self.split_result.X_val,
                y_val=self.split_result.y_val,
                sample_weight=self.split_result.train_weights,
            )
            val_probs = self.model_wrapper.predict_proba(temp_model, self.split_result.X_val)
            val_metrics_default = calculate_metrics(self.split_result.y_val.values, val_probs, threshold=0.5)
            best_th, best_val_f1 = find_best_threshold(self.split_result.y_val.values, val_probs, metric="f1")
            val_metrics_tuned = calculate_metrics(self.split_result.y_val.values, val_probs, threshold=best_th)
            val_title_suffix = "Validation Set"

        # Step 2: Fit Final Model on 100% of (X_train, y_train)
        print(f"\n[TRAIN] Training Final {self.model_name.upper()} Model on full Train Set ({len(self.split_result.X_train):,} samples)...")
        final_model = self.model_wrapper.build_model(best_params, seed=self.seed)
        self.model_wrapper.fit(
            model=final_model,
            X_train=self.split_result.X_train,
            y_train=self.split_result.y_train,
            X_val=self.split_result.X_val,
            y_val=self.split_result.y_val,
            sample_weight=self.split_result.train_weights,
        )

        # Step 3: Test evaluation
        test_probs = self.model_wrapper.predict_proba(final_model, self.split_result.X_test)
        test_metrics_default = calculate_metrics(self.split_result.y_test.values, test_probs, threshold=0.5)
        test_metrics_tuned = calculate_metrics(self.split_result.y_test.values, test_probs, threshold=best_th)

        # Print report tables
        print("\n" + "=" * 60)
        print(format_metrics_table(val_metrics_default, f"{self.model_name.upper()} - {val_title_suffix} (Default Threshold = 0.5000)"))
        print(format_metrics_table(val_metrics_tuned, f"{self.model_name.upper()} - {val_title_suffix} (Optimized Threshold = {best_th:.4f})"))
        print("=" * 60)
        print(format_metrics_table(test_metrics_default, f"{self.model_name.upper()} - Test Set (Default Threshold = 0.5000)"))
        print(format_metrics_table(test_metrics_tuned, f"{self.model_name.upper()} - Test Set (Optimized Threshold = {best_th:.4f})"))
        print("=" * 60)

        # Feature Importance
        df_importance = self.model_wrapper.extract_feature_importances(
            final_model, self.split_result.feature_names
        )
        print("\n[IMPORTANCE] Top 15 Most Important Features:")
        for idx, row in df_importance.head(15).iterrows():
            gain_val = row.get("gain", 0.0)
            gain_ratio = row.get("gain_ratio", 0.0)
            print(f"  {idx+1:2d}. {row['feature']:<35} Score: {gain_val:10.2f} ({gain_ratio*100:5.2f}%)")

        # Save artifacts
        ext = ".json" if "xgb" in self.model_name.lower() else ".joblib"
        model_filename = f"{self.model_name}_best_model{ext}"
        model_path = os.path.join(self.artifacts_dir, model_filename)
        self.model_wrapper.save_model(final_model, model_path)
        print(f"\n[SAVE] Saved Best Model to: '{model_path}'")

        params_path = os.path.join(self.artifacts_dir, "best_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "best_trial_score": study.best_value,
                "optimal_threshold": float(best_th),
            }, f, indent=2)
        print(f"[SAVE] Saved Best Parameters to: '{params_path}'")

        # Save feature importance chart (PNG)
        importance_png_path = os.path.join(self.artifacts_dir, "feature_importance.png")
        dataset_label = self.split_result.metadata.get("dataset_name", "dataset").upper()
        plot_feature_importances(
            df_importance=df_importance,
            top_n=20,
            output_path=importance_png_path,
            title=f"Feature Importance ({self.model_name.upper()} on {dataset_label})",
        )

        importance_path = os.path.join(self.artifacts_dir, "feature_importance.csv")
        df_importance.to_csv(importance_path, index=False)
        print(f"[SAVE] Saved Feature Importance to: '{importance_path}'")

        summary_path = os.path.join(self.artifacts_dir, "evaluation_summary.json")
        cleaned_metadata = {
            k: v for k, v in self.split_result.metadata.items()
            if not isinstance(v, (np.ndarray, pd.Series))
        }
        evaluation_summary = {
            "model_name": self.model_name,
            "dataset_metadata": cleaned_metadata,
            "val_metrics_default_th": val_metrics_default,
            "val_metrics_tuned_th": val_metrics_tuned,
            "test_metrics_default_th": test_metrics_default,
            "test_metrics_tuned_th": test_metrics_tuned,
            "optimal_threshold": float(best_th),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_summary, f, indent=2)
        print(f"[SAVE] Saved Evaluation Summary to: '{summary_path}'")

        return evaluation_summary
