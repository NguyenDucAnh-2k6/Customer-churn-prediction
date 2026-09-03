"""
Automated Ablation Study: Impact of Stock/Financial Technical Indicators
Compares performance WITH and WITHOUT stock indicators on TimeSeries and Latest datasets.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from src.data.datasets import DatasetRegistry
from src.models.registry import ModelRegistry
from src.training.trainer import OptunaTrainer


def run_single_experiment(dataset_name: str, stock_features: bool, n_trials: int = 5, cv_folds: int = 5):
    print(f"\n{'='*70}")
    print(f"Running Experiment: Dataset='{dataset_name}' | Stock Features = {stock_features}")
    print(f"{'='*70}")

    dataset = DatasetRegistry.get(dataset_name)
    split_result = dataset.load_and_split(
        split_strategy="group_stratified",
        stock_features=stock_features,
        use_cv=True,
    )

    model_wrapper = ModelRegistry.get("xgboost")

    stock_tag = "" if stock_features else "_nostock"
    study_name = f"xgb_ablation_{dataset_name}{stock_tag}_test"
    artifacts_dir = f"src/models/artifacts/ablation_{dataset_name}{stock_tag}"

    trainer = OptunaTrainer(
        model_wrapper=model_wrapper,
        split_result=split_result,
        db_url="sqlite:///tracking/optuna_study.db",
        study_name=study_name,
        metric_name="roc_auc",
        artifacts_dir=artifacts_dir,
        model_name="xgboost",
        cv_folds=cv_folds,
        seed=42,
    )

    study = trainer.run_hpo(n_trials=n_trials)
    eval_summary = trainer.train_and_evaluate_best(study=study)

    test_metrics = eval_summary.get("test_metrics_default_th", {})
    test_tuned = eval_summary.get("test_metrics_tuned_th", {})

    return {
        "dataset": dataset_name,
        "stock_features": "WITH Stock Indicators (Full)" if stock_features else "WITHOUT Stock Indicators (Ablated)",
        "n_features": len(split_result.feature_names),
        "test_roc_auc": test_metrics.get("roc_auc", 0.0),
        "test_pr_auc": test_metrics.get("pr_auc", 0.0),
        "test_precision_top5": test_metrics.get("precision_top_5_pct", 0.0),
        "test_lift_top5": test_metrics.get("lift_top_5_pct", 1.0),
        "test_recall_top10": test_metrics.get("recall_top_10_pct", 0.0),
        "test_f1": test_tuned.get("f1", test_metrics.get("f1", 0.0)),
        "test_macro_f1": test_tuned.get("f1_macro", test_metrics.get("f1_macro", 0.0)),
        "test_brier": test_metrics.get("brier_score", 0.0),
    }


def main():
    experiments = [
        ("timeseries", True),
        ("timeseries", False),
        ("latest", True),
        ("latest", False),
    ]

    results = []
    for ds, sf in experiments:
        res = run_single_experiment(dataset_name=ds, stock_features=sf, n_trials=3, cv_folds=5)
        results.append(res)

    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 ABLATION STUDY RESULTS: IMPACT OF STOCK/FINANCIAL INDICATORS")
    print("="*80)
    print(df_res.to_string(index=False))

    # Save to Markdown
    md_lines = [
        "# 📊 Ablation Study: Impact of Stock / Financial Behavioral Indicators",
        "",
        "| Dataset | Configuration | Feature Count | Test ROC-AUC | Test PR-AUC | Precision@Top 5% | Top 5% Lift | Test F1-Score | Brier Score |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md_lines.append(
            f"| **{r['dataset'].upper()}** | `{r['stock_features']}` | `{r['n_features']}` | `{r['test_roc_auc']:.4f}` | `{r['test_pr_auc']:.4f}` | `{r['test_precision_top5']*100:.2f}%` | `{r['test_lift_top5']:.2f}x` | `{r['test_f1']:.4f}` | `{r['test_brier']:.4f}` |"
        )
    md_lines.append("")

    os.makedirs("tracking", exist_ok=True)
    with open("tracking/ablation_stock_features.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("\n[SAVE] Saved Ablation Report to 'tracking/ablation_stock_features.md'")


if __name__ == "__main__":
    main()
