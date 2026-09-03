import pandas as pd
import numpy as np


def safe_rolling(series, window, func='mean', min_pct=0.8):   
    min_periods = max(int(window * min_pct), 1)
    
    if func == 'mean':
        return series.rolling(window, min_periods=min_periods).mean()
    elif func == 'std':
        return series.rolling(window, min_periods=min_periods).std()
    elif func == 'min':
        return series.rolling(window, min_periods=min_periods).min()
    elif func == 'max':
        return series.rolling(window, min_periods=min_periods).max()
    elif func == 'sum':
        return series.rolling(window, min_periods=min_periods).sum()
    else:
        raise ValueError(f"Unsupported function: {func}")

def safe_zscore(series, window=252, min_pct=0.8):
    min_periods = max(int(window * min_pct), 1)
    
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score


def add_momentum_features(df: pd.DataFrame, rsi_windows=(14, 28), roc_windows=(5, 10, 20), 
    stoch_window=14, norm_window=252) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    for rsi_w in rsi_windows:
        delta = close.shift(1).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/rsi_w, min_periods=rsi_w, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_w, min_periods=rsi_w, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f"RSI_{rsi_w}"] = 100 - (100 / (1 + rs))
    rsi = df["RSI_14"]
    df["RSI_overbought_smooth"] = 1 / (1 + np.exp(-0.2 * (rsi - 70)))
    df["RSI_oversold_smooth"] = 1 / (1 + np.exp(0.2 * (rsi - 30)))
    df["RSI_dist_neutral"] = (rsi - 50) / 50
    df["RSI_momentum_3"] = rsi.diff(3) / 3
    df["RSI_momentum_5"] = rsi.diff(5) / 5
    df["RSI_accel"] = df["RSI_momentum_3"].diff(3)
    price_mom = close.shift(1).pct_change(14)
    rsi_mom = rsi.diff(14)
    df["RSI_price_divergence"] = (
        (price_mom / (price_mom.abs() + 1e-9)) -
        (rsi_mom / (rsi_mom.abs() + 1e-9))
    ) / 2
    lowest_low = low.shift(1).rolling(stoch_window, min_periods=int(stoch_window*0.8)).min()
    highest_high = high.shift(1).rolling(stoch_window, min_periods=int(stoch_window*0.8)).max()
    stoch_k = 100 * (close.shift(1) - lowest_low) / (highest_high - lowest_low + 1e-9)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_k.rolling(3, min_periods=2).mean()
    df["Stoch_momentum"] = df["Stoch_K"].diff(5)
    df["Stoch_overbought_smooth"] = 1 / (1 + np.exp(-0.15 * (df["Stoch_K"] - 80)))
    df["Stoch_oversold_smooth"] = 1 / (1 + np.exp(0.15 * (df["Stoch_K"] - 20)))
    for w in roc_windows:
        df[f"ROC_{w}"] = close.shift(1).pct_change(w)
        df[f"ROC_{w}_momentum"] = df[f"ROC_{w}"].diff(w)
    df["ROC_consistency"] = (
        np.sign(df["ROC_5"]) == np.sign(df["ROC_20"])
    ).astype(float)
    for w in [5, 10, 20]:
        df[f"momentum_{w}"] = close.shift(1) - close.shift(1 + w)
        tr = pd.concat([
            high.shift(1) - low.shift(1),
            (high.shift(1) - close.shift(2)).abs(),
            (low.shift(1) - close.shift(2)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(w, min_periods=int(w*0.8)).mean()
        df[f"momentum_{w}_atr"] = df[f"momentum_{w}"] / (atr + 1e-9)
    roc_5_std = safe_rolling(df["ROC_5"], 60, 'std', min_pct=0.8)
    roc_10_std = safe_rolling(df["ROC_10"], 60, 'std', min_pct=0.8)
    roc_20_std = safe_rolling(df["ROC_20"], 60, 'std', min_pct=0.8)
    df["momentum_composite"] = (
        0.3 * (df["ROC_5"] / (roc_5_std + 1e-9)) +
        0.4 * (df["ROC_10"] / (roc_10_std + 1e-9)) +
        0.3 * (df["ROC_20"] / (roc_20_std + 1e-9))
    )
    df["RSI_price_corr_20"] = (
        rsi.rolling(20, min_periods=15)
        .corr(close)
    )
    features_to_winsorize = [
        col for col in df.columns 
        if col.startswith(('ROC_', 'momentum_', 'RSI_momentum', 'RSI_accel'))
    ]
    for col in features_to_winsorize:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[f"{col}_winsor"] = df[col].clip(lower=lower, upper=upper)
    features_to_normalize = [
        col for col in df.columns
        if col.startswith((
            'RSI_', 'Stoch_', 'ROC_', 'momentum_',
            'RSI_price_divergence', 'momentum_composite'
        )) and not col.endswith(('_smooth', '_winsor'))
    ]
    for col in features_to_normalize:
        df[f"{col}_z"] = safe_zscore(df[col], window=norm_window, min_pct=0.8)
    df["uptrend_strength"] = (
        df["RSI_overbought_smooth"] * 0.3 +
        (df["momentum_composite"] > 0).astype(float) * 0.4 +
        df["Stoch_overbought_smooth"] * 0.3
    ).clip(0, 1)
    df["downtrend_strength"] = (
        df["RSI_oversold_smooth"] * 0.3 +
        (df["momentum_composite"] < 0).astype(float) * 0.4 +
        df["Stoch_oversold_smooth"] * 0.3
    ).clip(0, 1)
    df["ranging_strength"] = 1 - (
        df["uptrend_strength"] + df["downtrend_strength"]
    ).clip(0, 1)
    momentum_composite_std = safe_rolling(df["momentum_composite"], 60, 'std', min_pct=0.8)
    if "volume" in df.columns:
        vol_ratio = df["volume"] / safe_rolling(df["volume"], 20, 'mean', min_pct=0.8)
        df["momentum_quality"] = (
            df["ROC_consistency"] * 0.3 +
            (df["momentum_composite"].abs() > momentum_composite_std).astype(float) * 0.4 +
            (vol_ratio > 1).astype(float) * 0.3
        ).clip(0, 1)
    else:
        df["momentum_quality"] = (
            df["ROC_consistency"] * 0.5 +
            (df["momentum_composite"].abs() > momentum_composite_std).astype(float) * 0.5
        ).clip(0, 1)
    return df