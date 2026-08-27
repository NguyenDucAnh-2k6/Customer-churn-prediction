"""
Unified CLI Entrypoint for Exploratory Data Analysis (EDA).

Usage:
    # Run EDA on both datasets:
    python -m src.run_eda --all

    # Run EDA only on time-series dataset:
    python -m src.run_eda --dataset timeseries

    # Run EDA only on static dataset:
    python -m src.run_eda --dataset static
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
        choices=["timeseries", "static", "latest", "all"],
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

    if run_all or args.dataset == "timeseries":
        run_timeseries_eda(args.output_dir)

    if run_all or args.dataset == "static":
        run_static_eda(args.output_dir)

    if run_all or args.dataset == "latest":
        run_latest_eda(args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 Complete EDA Pipeline Finished in {elapsed:.2f}s ({elapsed/60:.2f} mins)!")
    print(f"📁 Reports and Charts saved to: '{args.output_dir}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
