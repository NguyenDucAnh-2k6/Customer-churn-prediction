import numpy as np
import pandas as pd
from src.data.data_preprocess.market.utils import safe_rolling, safe_ewm, safe_zscore
from src.data.data_collect.getter import get_raw_data

def add_market_trend_features(df, market_name="market"):
    market_df = get_raw_data(stock_name=market_name)
    close = market_df['close'].shift(1)
    new_features = {}
    new_features[f'{market_name}_mom_60'] = close.pct_change(60)
    new_features[f'{market_name}_mom_120'] = close.pct_change(120)
    new_features[f'{market_name}_mom_252'] = close.pct_change(252)
    new_features[f'{market_name}_mom_60_z'] = safe_zscore(new_features[f'{market_name}_mom_60'], 252, 0.8)
    new_features[f'{market_name}_mom_120_z'] = safe_zscore(new_features[f'{market_name}_mom_120'], 252, 0.8)
    ema_50 = safe_ewm(close, 50, 0.8)
    ema_200 = safe_ewm(close, 200, 0.8)
    new_features[f'{market_name}_bull'] = (ema_50 > ema_200).astype(float)
    new_features[f'{market_name}_trend_strength'] = (ema_50 - ema_200) / ema_200
    new_features[f'{market_name}_trend_strength_z'] = safe_zscore(
        new_features[f'{market_name}_trend_strength'], 252, 0.8
    )
    market_peak = close.rolling(252, min_periods=200).max()
    new_features[f'{market_name}_drawdown'] = (close - market_peak) / market_peak
    new_features[f'{market_name}_deep_drawdown'] = (new_features[f'{market_name}_drawdown'] < -0.15).astype(float)
    market_ret = close.pct_change()
    new_features[f'{market_name}_trend_consistency_60'] = safe_rolling(
        (market_ret > 0).astype(float), 60, 'mean', 0.8
    )
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    return df