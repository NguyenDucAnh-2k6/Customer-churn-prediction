import pandas as pd
import numpy as np
from src.data.data_preprocess.stock.utils import safe_rolling, safe_ewm, safe_zscore


def add_trend_features(
    df: pd.DataFrame,
    ema_windows=(20, 50, 200),
    slope_windows=(5, 20, 60)
) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].shift(1)
    ret = close.pct_change()
    for w in ema_windows:
        df[f"ema_{w}"] = safe_ewm(close, span=w, min_pct=0.8)
    for w in ema_windows:
        df[f"price_ema_dist_{w}"] = (
            close - df[f"ema_{w}"]
        ) / (df[f"ema_{w}"] + 1e-9)
    for ema_w in ema_windows:
        ema = df[f"ema_{ema_w}"]
        for s in slope_windows:
            df[f"ema_slope_{ema_w}_{s}"] = (
                (ema - ema.shift(s)) / s
            ) / close
    if 20 in ema_windows and 50 in ema_windows:
        df["ema_20_50_spread"] = (
            df["ema_20"] - df["ema_50"]
        ) / (close + 1e-9)
    if 50 in ema_windows and 200 in ema_windows:
        df["ema_50_200_spread"] = (
            df["ema_50"] - df["ema_200"]
        ) / (close + 1e-9)
    for w in (20, 60, 120):
        df[f"trend_consistency_{w}"] = safe_rolling(
            (ret > 0).astype(float), w, 'mean', min_pct=0.8
        )
    vol_20 = safe_rolling(ret, 20, 'std', min_pct=0.8)
    for w in ema_windows:
        df[f"trend_strength_adj_vol_{w}"] = (
            df[f"price_ema_dist_{w}"] / vol_20
        )
    for col in df.columns:
        if col.startswith((
            "price_ema_dist",
            "ema_slope",
            "ema_",
            "trend_strength_adj_vol"
        )):
            df[f"{col}_z"] = safe_zscore(df[col], window=252, min_pct=0.8)
    return df
