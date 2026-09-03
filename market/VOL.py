import numpy as np
import pandas as pd
from src.data.data_preprocess.market.utils import safe_rolling, safe_zscore
from src.data.data_collect.getter import get_raw_data

def add_market_volatility_features(df, market_name="market"):
    market_df = get_raw_data(stock_name=market_name)
    market_ret = market_df['close'].shift(1).pct_change()
    features = {}
    features[f'{market_name}_vol_20'] = safe_rolling(market_ret, 20, 'std', 0.8) * np.sqrt(252)
    features[f'{market_name}_vol_60'] = safe_rolling(market_ret, 60, 'std', 0.8) * np.sqrt(252)
    features[f'{market_name}_vol_252'] = safe_rolling(market_ret, 252, 'std', 0.8) * np.sqrt(252)
    features[f'{market_name}_vol_regime'] = safe_zscore(features[f'{market_name}_vol_20'], window=252, min_pct=0.8)
    features[f'{market_name}_is_high_vol_regime'] = (
        features[f'{market_name}_vol_20'] > features[f'{market_name}_vol_252'].rolling(252).quantile(0.75)
    ).astype(float)
    features[f'{market_name}_vol_term_structure'] = features[f'{market_name}_vol_20'] / (features[f'{market_name}_vol_60'] + 1e-9)
    features[f'{market_name}_vol_trend'] = features[f'{market_name}_vol_20'].pct_change(20)
    features[f'{market_name}_downside_vol_20'] = safe_rolling(
        market_ret.where(market_ret < 0, 0), 20, 'std', 0.8
    ) * np.sqrt(252)
    features[f'{market_name}_vol_asymmetry'] = (
        features[f'{market_name}_downside_vol_20'] / (features[f'{market_name}_vol_20'] + 1e-9)
    )
    if features:
        new_df = pd.DataFrame(features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    return df