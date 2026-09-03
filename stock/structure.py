import pandas as pd
import numpy as np
from src.data.data_preprocess.stock.utils import safe_rolling, safe_zscore


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].shift(1)

    timeframes = [20, 50, 200]

    for w in timeframes:
        rolling_high = safe_rolling(close, w, 'max', min_pct=0.8)
        rolling_low = safe_rolling(close, w, 'min', min_pct=0.8)

        df[f"range_position_{w}"] = (
            (close - rolling_low) / (rolling_high - rolling_low + 1e-9)
        )

        df[f"breakout_high_strength_{w}"] = np.maximum(
            (close - rolling_high.shift(1)) / (rolling_high.shift(1) + 1e-9),
            0
        )

        df[f"breakout_low_strength_{w}"] = np.maximum(
            (rolling_low.shift(1) - close) / (rolling_low.shift(1) + 1e-9),
            0
        )

        df[f"dist_to_high_{w}"] = (close - rolling_high) / (rolling_high + 1e-9)
        df[f"dist_to_low_{w}"] = (close - rolling_low) / (rolling_low + 1e-9)

        df[f"structure_tightness_{w}"] = (
            (rolling_high - rolling_low) / (close + 1e-9)
        )

    df["range_position_alignment"] = (
        df["range_position_20"] - df["range_position_50"]
    ).abs()  

    df["breakout_confirmation"] = (
        df["breakout_high_strength_20"] * 0.5 +
        df["breakout_high_strength_50"] * 0.3 +
        df["breakout_high_strength_200"] * 0.2
    )

    # ======================================================
    # Z-SCORE NORMALIZATION (IC SURVIVAL)
    # ======================================================
    structure_cols = [
        col for col in df.columns
        if col.startswith((
            'range_position_', 'breakout_', 'dist_to_',
            'structure_tightness', 'range_position_alignment',
            'breakout_confirmation'
        ))
    ]

    for col in structure_cols:
        df[f"{col}_z"] = safe_zscore(df[col], window=252, min_pct=0.8)

    return df
