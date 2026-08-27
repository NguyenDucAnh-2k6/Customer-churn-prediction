"""
Unified CLI Entrypoint for Customer Churn Modeling and Experiments.

Usage:
    # Run XGBoost on Time-Series dataset:
    python -m src.train --dataset timeseries --model xgboost --n_trials 30

    # Run XGBoost on Static dataset:
    python -m src.train --dataset static --model xgboost --n_trials 30

    # Run Random Forest baseline:
    python -m src.train --dataset static --model random_forest --n_trials 20

    # List all registered datasets and models:
    python -m src.train --list
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.datasets import DatasetRegistry
from src.models.registry import ModelRegistry
from src.training.trainer import OptunaTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ML Training Pipeline with Optuna Registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="timeseries",
        help=f"Dataset to train on. Choices: {DatasetRegistry.list_available()}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help=f"Model architecture to use. Choices: {ModelRegistry.list_available()}",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Max time in seconds for Optuna optimization",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "pr_auc", "precision_at_5", "recall_at_10", "lift_top_5", "f1_macro", "f1"],
        help="Metric to optimize in Optuna objective",
    )
    parser.add_argument(
        "--decay_half_life",
        type=float,
        default=None,
        help="Half-life in months for Exponential Recency Decay Sample Weights (e.g. 12.0)",
    )
    parser.add_argument(
        "--customer_weight_power",
        "--cust_power",
        type=float,
        default=0.0,
        help="Power alpha for Inverse Customer Frequency balancing w = 1 / (N_cust)^alpha (e.g. 0.5 or 1.0)",
    )
    parser.add_argument(
        "--use_usage_weight",
        action="store_true",
        help="Enable engagement activity log-scaling sample weights based on active days",
    )
    parser.add_argument(
        "--drop_low_mi",
        action="store_true",
        help="Ablation: Drop features with near-zero Mutual Information (demographics, CSAT missing, marketing open)",
    )
    parser.add_argument(
        "--drop_collinear",
        action="store_true",
        help="Ablation: Drop highly collinear/duplicate features (|r| >= 0.90)",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="time_based",
        choices=["time_based", "group_stratified", "customer_stratified", "pit"],
        help="Splitting strategy for dataset: 'time_based' (OOT temporal), 'group_stratified' (Customer StratifiedGroupKFold, 0 leakage), 'pit' (Point-in-Time)",
    )
    parser.add_argument(
        "--group_stratified",
        "--customer_stratified",
        action="store_true",
        help="Shorthand for --split_strategy group_stratified (Customer-level StratifiedGroupKFold, Zero Customer Leakage)",
    )
    parser.add_argument(
        "--behavioral_only",
        "--drop_static_tiers",
        action="store_true",
        help="Strategy 4: Drop all static account & tier categorical features (is_paid_tier, subscription_tier, demographics) to prevent shortcut learning",
    )
    parser.add_argument(
        "--stock_features",
        action="store_true",
        default=True,
        help="Include financial & stock-market technical indicators (peer_usage_zscore, engagement_macd, usage_drawdown_from_peak, active_days_volatility_3m) (Default: True)",
    )
    parser.add_argument(
        "--no_stock_features",
        "--drop_stock_features",
        "--no-stock-features",
        action="store_false",
        dest="stock_features",
        help="Ablation: Exclude financial & stock-market technical indicators",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        default="sqlite:///tracking/optuna_study.db",
        help="Optuna SQLite database URL",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name. Defaults to '{model}_{dataset}'",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Directory to save artifacts. Defaults to 'src/models/artifacts/{dataset}_{model}'",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Custom dataset CSV path (optional override)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of Cross-Validation folds for training (e.g., 5). If set, runs K-Fold CV on Train set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered datasets and models and exit",
    )
    return parser.parse_args()


def print_registry_info() -> None:
    """Print available datasets and models."""
    print("=" * 60)
    print("📋 Machine Learning Registry Catalog")
    print("=" * 60)
    print("📊 Registered Datasets:")
    for d in DatasetRegistry.list_available():
        print(f"  • {d}")
    print("\n🤖 Registered Models:")
    for m in ModelRegistry.list_available():
        print(f"  • {m}")
    print("=" * 60)


def main():
    args = parse_args()

    if args.list:
        print_registry_info()
        return

    start_time = time.time()
    dataset_name = args.dataset.lower()
    model_name = args.model.lower()

    # Determine split_strategy and tags
    split_strategy = "group_stratified" if args.group_stratified else args.split_strategy
    strat_tag = f"_{split_strategy}" if split_strategy != "time_based" else ""
    decay_tag = f"_decay{int(args.decay_half_life)}m" if args.decay_half_life else ""
    cust_tag = f"_cust{args.customer_weight_power:.1f}" if args.customer_weight_power > 0 else ""
    usage_tag = "_usage" if args.use_usage_weight else ""
    mi_tag = "_noMI" if args.drop_low_mi else ""
    coll_tag = "_noColl" if args.drop_collinear else ""
    behav_tag = "_behavioral" if args.behavioral_only else ""
    stock_tag = "" if args.stock_features else "_nostock"
    metric_tag = f"_{args.metric}" if args.metric != "roc_auc" else ""
    study_name = args.study_name or f"{model_name}_churn_{dataset_name}{strat_tag}{decay_tag}{cust_tag}{usage_tag}{mi_tag}{coll_tag}{behav_tag}{stock_tag}{metric_tag}"
    artifacts_dir = args.artifacts_dir or f"src/models/artifacts/{dataset_name}_{model_name}{strat_tag}{decay_tag}{cust_tag}{usage_tag}{mi_tag}{coll_tag}{behav_tag}{stock_tag}{metric_tag}"

    print(f"\n{'='*60}")
    print(f"🚀 Starting Experiment: Model='{model_name}' on Dataset='{dataset_name}'")
    print(f"   Split Strategy: '{split_strategy}' | Metric: '{args.metric.upper()}' | Trials: {args.n_trials}")
    print(f"   Optuna Study: '{study_name}'")
    weights_desc = []
    if args.decay_half_life:
        weights_desc.append(f"RecencyDecay({args.decay_half_life}m)")
    if args.customer_weight_power > 0:
        weights_desc.append(f"CustBalancing(power={args.customer_weight_power})")
    if args.use_usage_weight:
        weights_desc.append("UsageEngagement")
    if weights_desc:
        print(f"   Sample Weights: {' + '.join(weights_desc)}")
    
    ablation_desc = []
    if args.behavioral_only:
        ablation_desc.append("Strategy 4: Pure Behavioral (Drop Static Tiers)")
    if not args.stock_features:
        ablation_desc.append("Ablation: Exclude Stock/Financial Features (No Beta/MACD/Drawdown/Volatility)")
    if args.drop_low_mi:
        ablation_desc.append("Drop Low-MI (Noisy)")
    if args.drop_collinear:
        ablation_desc.append("Drop Collinear (|r|>=0.90)")
    if ablation_desc:
        print(f"   Feature Selection Ablation: {' + '.join(ablation_desc)}")
    print(f"   Artifacts Dir: '{artifacts_dir}'")
    print(f"{'='*60}\n")

    # If dataset is static, pit, or group_stratified and --cv is not specified, default to 5-fold CV
    cv_folds = args.cv
    if cv_folds is None and (
        dataset_name in ["static", "pit", "point_in_time", "timeseries_pit", "timeseries_group", "latest_group", "group_stratified"]
        or split_strategy in ["group_stratified", "stratified_group"]
    ):
        cv_folds = 5

    # 1. Instantiate Dataset & Perform Split
    dataset_kwargs = {}
    if args.data_path:
        dataset_kwargs["data_path"] = args.data_path
    if split_strategy != "time_based":
        dataset_kwargs["split_strategy"] = split_strategy
    if cv_folds:
        dataset_kwargs["use_cv"] = True
    if cv_folds and dataset_name in ["static", "pit", "point_in_time", "timeseries_pit"]:
        dataset_kwargs["val_size"] = 0.0
    if args.decay_half_life:
        dataset_kwargs["decay_half_life"] = args.decay_half_life
    if args.customer_weight_power > 0:
        dataset_kwargs["customer_weight_power"] = args.customer_weight_power
    if args.use_usage_weight:
        dataset_kwargs["use_usage_weight"] = args.use_usage_weight
    if args.behavioral_only:
        dataset_kwargs["behavioral_only"] = True
    if not args.stock_features:
        dataset_kwargs["stock_features"] = False
    if args.drop_low_mi:
        dataset_kwargs["drop_low_mi"] = True
    if args.drop_collinear:
        dataset_kwargs["drop_collinear"] = True

    dataset = DatasetRegistry.get(dataset_name)
    split_result = dataset.load_and_split(**dataset_kwargs)

    # 2. Instantiate Model Wrapper
    model_wrapper = ModelRegistry.get(model_name)

    # 3. Instantiate Trainer & Run HPO
    trainer = OptunaTrainer(
        model_wrapper=model_wrapper,
        split_result=split_result,
        db_url=args.db_url,
        study_name=study_name,
        metric_name=args.metric,
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        cv_folds=cv_folds,
        seed=args.seed,
    )

    study = trainer.run_hpo(n_trials=args.n_trials, timeout=args.timeout)

    # 4. Train Final Best Model & Save Artifacts
    trainer.train_and_evaluate_best(study=study)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Experiment Completed in {elapsed:.2f}s ({elapsed/60:.2f} mins).")


if __name__ == "__main__":
    main()
