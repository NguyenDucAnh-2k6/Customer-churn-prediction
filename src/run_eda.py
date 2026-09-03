"""
Unified CLI Entrypoint for Exploratory Data Analysis (EDA).

Usage:
    # Run EDA on all datasets:
    python -m src.eda --all
    python -m src.run_eda --all

    # Run EDA on Round 3 Point-in-Time dataset:
    python -m src.eda --dataset round3

    # Run EDA on Round 3 Time-Series dataset:
    python -m src.eda --dataset round3_timeseries

    # Run EDA on original Time-Series dataset:
    python -m src.eda --dataset timeseries

    # Run EDA on Static dataset:
    python -m src.eda --dataset static
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.eda.report import generate_eda_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exploratory Data Analysis on Churn Datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["round3", "round_3", "r3", "round3_timeseries", "r3_timeseries", "timeseries", "static", "latest", "all"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all available datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/eda",
        help="Base output directory for generated plots and reports",
    )
    return parser.parse_args()


def run_round3_eda(base_output_dir: str):
    data_path = "data/processed/round3/churn_master.csv"
    if not Path(data_path).exists():
        data_path = "data/processed/round3/churn_train.csv"
    print(f"\n[INFO] Loading Round 3 Point-in-Time dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    target_col = "churn" if "churn" in df.columns else "label_churn"
    output_dir = f"{base_output_dir}/round3_point_in_time"
    return generate_eda_suite(
        df=df,
        dataset_name="round3_point_in_time",
        target_col=target_col,
        output_dir=output_dir,
        time_col=None,
    )


def run_round3_timeseries_eda(base_output_dir: str):
    data_path = "data/processed/round3_timeseries/churn_timeseries_master.csv"
    if not Path(data_path).exists():
        data_path = "data/processed/round3_timeseries/churn_timeseries_train.csv"
    print(f"\n[INFO] Loading Round 3 Time-Series Panel dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    target_col = "label_churn" if "label_churn" in df.columns else "churn"
    output_dir = f"{base_output_dir}/round3_timeseries"
    return generate_eda_suite(
        df=df,
        dataset_name="round3_timeseries",
        target_col=target_col,
        output_dir=output_dir,
        time_col="snapshot_month",
    )


def run_timeseries_eda(base_output_dir: str):
    data_path = "data/processed/churn_feature_dataset_processed.csv"
    print(f"\n[INFO] Loading Time-Series dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    output_dir = f"{base_output_dir}/timeseries"
    return generate_eda_suite(
        df=df,
        dataset_name="timeseries",
        target_col="label_churn",
        output_dir=output_dir,
        time_col="snapshot_month",
    )


def run_static_eda(base_output_dir: str):
    data_path = "data/processed/dataset02_fixed.csv"
    print(f"\n[INFO] Loading Static dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    output_dir = f"{base_output_dir}/static"
    return generate_eda_suite(
        df=df,
        dataset_name="static",
        target_col="churn",
        output_dir=output_dir,
        time_col=None,
    )


def run_latest_eda(base_output_dir: str):
    train_path = "data/processed/latest/churn_train.csv"
    val_path = "data/processed/latest/churn_val.csv"
    test_path = "data/processed/latest/churn_test.csv"
    print(f"\n[INFO] Loading Latest Pre-Split dataset from '{train_path}'...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    output_dir = f"{base_output_dir}/latest"
    return generate_eda_suite(
        df=df_all,
        dataset_name="latest",
        target_col="churn_30d",
        output_dir=output_dir,
        time_col="snapshot_month",
    )


def main():
    args = parse_args()
    start_time = time.time()
    run_all = args.all or args.dataset == "all"
    dataset_choice = args.dataset.lower()

    if run_all or dataset_choice in ["round3", "round_3", "r3"]:
        run_round3_eda(args.output_dir)

    if run_all or dataset_choice in ["round3_timeseries", "r3_timeseries"]:
        run_round3_timeseries_eda(args.output_dir)

    if run_all or dataset_choice == "timeseries":
        run_timeseries_eda(args.output_dir)

    if run_all or dataset_choice == "static":
        run_static_eda(args.output_dir)

    if run_all or dataset_choice == "latest":
        run_latest_eda(args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 Complete EDA Pipeline Finished in {elapsed:.2f}s ({elapsed/60:.2f} mins)!")
    print(f"📁 Reports and Charts saved to: '{args.output_dir}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
