"""
Feature preprocessing, cleaning, and hybrid enrichment orchestrator.
Composes modular transformers for data hygiene, velocity dynamics, and financial indicators.
"""

import os
from typing import List, Optional
import numpy as np
import pandas as pd

from src.features.cleaning import DataCleaningTransformer
from src.features.velocity import VelocityFeatureGenerator
from src.features.financial_indicators import FinancialFeatureGenerator


class ChurnFeaturePreprocessor:
    """Master orchestrator pipeline that applies data hygiene, static master merging,
    velocity features, and financial technical indicators.
    """

    def __init__(
        self,
        static_ml_path: Optional[str] = "data/churn_ml_dataset.csv",
        merge_static_master: bool = False,
        customer_id_col: str = "customer_id",
        snapshot_month_col: str = "snapshot_month",
    ):
        self.static_ml_path = static_ml_path
        self.merge_static_master = merge_static_master
        self.customer_id_col = customer_id_col
        self.snapshot_month_col = snapshot_month_col
        self.df_static_master = None

        # Instantiate modular sub-transformers
        self.cleaner = DataCleaningTransformer()
        self.velocity_gen = VelocityFeatureGenerator()
        self.financial_gen = FinancialFeatureGenerator(
            customer_id_col=customer_id_col,
            snapshot_month_col=snapshot_month_col,
        )

        if self.merge_static_master and self.static_ml_path and os.path.exists(self.static_ml_path):
            try:
                df_ml = pd.read_csv(self.static_ml_path)
                cols_to_keep = [
                    'customer_id',
                    'total_orders', 'completed_orders', 'returned_orders', 'cancelled_orders',
                    'total_spent', 'avg_order_value', 'max_order_value',
                    'total_items_purchased', 'distinct_products_bought', 'distinct_categories_bought',
                    'total_payments', 'successful_payments', 'failed_payments', 'total_payment_amount',
                    'total_support_tickets', 'urgent_tickets', 'account_tickets',
                    'total_usage_sessions', 'total_usage_seconds', 'avg_session_seconds',
                    'mkt_total_interactions', 'mkt_opened_count', 'mkt_clicked_count', 'mkt_converted_count',
                    'mkt_open_rate', 'mkt_click_rate', 'mkt_conversion_rate'
                ]
                avail_cols = [c for c in cols_to_keep if c in df_ml.columns]
                self.df_static_master = df_ml[avail_cols].drop_duplicates(subset=['customer_id']).copy()
                print(f"[PREPROCESSOR] Loaded {len(self.df_static_master):,} customer records from static master '{self.static_ml_path}'")
            except Exception as e:
                print(f"[PREPROCESSOR] Warning: Could not load static master '{self.static_ml_path}': {e}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply modular feature transformation pipeline."""
        df = df.copy()

        # Step 1: Merge Static Customer Master Features (if requested)
        if self.df_static_master is not None and 'total_spent' not in df.columns and 'customer_id' in df.columns:
            df = df.merge(self.df_static_master, on='customer_id', how='left')

        # Step 2: Data Cleaning, Typing & Missing Indicators
        df = self.cleaner.transform(df)

        # Step 3: Velocity, Share, and Acceleration Features
        df = self.velocity_gen.transform(df)

        # Step 4: Financial & Quantitative Technical Indicators (Volatility, MACD, Drawdown, Beta)
        df = self.financial_gen.transform(df)

        return df
