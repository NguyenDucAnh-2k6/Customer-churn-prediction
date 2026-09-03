import pandas as pd
from src.data.data_preprocess.stock.utils import safe_rolling, safe_zscore


def add_volatility_features(
    df: pd.DataFrame,
    vol_windows=(10, 20, 60),
    atr_window=14
) -> pd.DataFrame:
    df = df.copy()
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    close = df["close"].shift(1)
    ret = close.pct_change()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = safe_rolling(tr, atr_window, 'mean', min_pct=0.8)
    df["atr_pct"] = atr / close
    atr_mean_252 = safe_rolling(atr, 252, 'mean', min_pct=0.8)
    df["atr_rel"] = atr / atr_mean_252
    for w in vol_windows:
        df[f"vol_{w}"] = safe_rolling(ret, w, 'std', min_pct=0.8)
    df["vol_term_10_20"] = df["vol_10"] / df["vol_20"]
    df["vol_term_20_60"] = df["vol_20"] / df["vol_60"]
    for w in (20, 60):
        df[f"downside_vol_{w}"] = safe_rolling(
            ret.where(ret < 0, 0), w, 'std', min_pct=0.8
        )
    df["downside_vol_ratio"] = (
        df["downside_vol_20"] / (df["vol_20"] + 1e-9)
    )
    df["vol_of_vol_20"] = safe_rolling(df["vol_20"], 20, 'std', min_pct=0.8)
    df["vol_regime_score"] = safe_zscore(df["vol_20"], window=252, min_pct=0.8)
    for col in df.columns:
        if col.startswith((
            "atr_",
            "vol_",
            "downside_vol",
            "vol_term",
            "vol_regime"
        )):
            df[f"{col}_z"] = safe_zscore(df[col], window=252, min_pct=0.8)
    return df
