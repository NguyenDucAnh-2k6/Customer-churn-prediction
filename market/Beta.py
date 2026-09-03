import numpy as np
import pandas as pd
from src.data.data_preprocess.market.utils import safe_rolling, safe_zscore
from src.data.data_collect.getter import get_raw_data

def add_stock_market_beta_features(df, market_name="market"):
    market_df = get_raw_data(stock_name=market_name)
    stock_ret = df['close'].shift(1).pct_change()
    market_ret = market_df['close'].shift(1).pct_change()
    new_features = {}
    for window in [60, 120, 252]:
        cov = stock_ret.rolling(window, min_periods=int(window*0.8)).cov(market_ret)
        var = market_ret.rolling(window, min_periods=int(window*0.8)).var()
        new_features[f'{market_name}_beta_{window}'] = cov / (var + 1e-9)
    new_features[f'{market_name}_beta_stability'] = new_features[f'{market_name}_beta_60'].rolling(252).std()
    new_features[f'{market_name}_beta_trend'] = new_features[f'{market_name}_beta_120'].diff(60)
    new_features[f'{market_name}_rel_strength_20'] = (
        stock_ret.rolling(20).sum() - market_ret.rolling(20).sum()
    )
    new_features[f'{market_name}_rel_strength_60'] = (
        stock_ret.rolling(60).sum() - market_ret.rolling(60).sum()
    )
    new_features[f'{market_name}_rel_strength_60_z'] = safe_zscore(new_features[f'{market_name}_rel_strength_60'], 252, 0.8)
    is_bull = df[f'{market_name}_bull'] == 1.0
    new_features[f'{market_name}_beta_in_bull'] = (
        new_features[f'{market_name}_beta_120']
        .where(is_bull)
        .ffill()
    )
    new_features[f'{market_name}_beta_in_bear'] = (
        new_features[f'{market_name}_beta_120']
        .where(~is_bull)
        .ffill()
    )
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    return df