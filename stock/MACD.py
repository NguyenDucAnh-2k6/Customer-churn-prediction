from src.data.data_collect.getter import get_raw_data
import pandas as pd
from src.data.data_preprocess.stock.utils import safe_rolling, safe_zscore, safe_ewm


def add_momentum_features(
    stock_name: str,
    mom_windows=(5, 20, 60, 120),
    vol_windows=(20, 60),
    ema_windows=(20, 60)
) -> pd.DataFrame:
    data = get_raw_data(stock_name=stock_name).copy()
    data = data[~data.index.duplicated(keep="last")]
    close = data["close"]
    ret = close.pct_change()
    features = {}
    for w in mom_windows:
        features[f"mom_{w}"] = close / close.shift(w) - 1
    features["mom_20_skip5"] = close.shift(5) / close.shift(25) - 1
    for w in ema_windows:
        ema = safe_ewm(close, span=w, min_pct=0.8)
        features[f"price_ema_dist_{w}"] = (close - ema) / ema
        features[f"ema_slope_{w}"] = (ema - ema.shift(w)) / w / close
    for w in vol_windows:
        vol = safe_rolling(ret, w, 'std', min_pct=0.8)
        features[f"vol_{w}"] = vol
        features[f"downside_vol_{w}"] = safe_rolling(
            ret.where(ret < 0, 0), w, 'std', min_pct=0.8
        )
        features[f"vol_of_vol_{w}"] = safe_rolling(vol, w, 'std', min_pct=0.8)
    features["positive_return_ratio_20"] = safe_rolling(
        (ret > 0).astype(float), 20, 'mean', min_pct=0.8
    )
    for w in mom_windows:
        col = f"mom_{w}"
        features[f"{col}_z"] = safe_zscore(features[col], window=252, min_pct=0.8)
    feature_df = pd.DataFrame(features, index=data.index)
    data = pd.concat([data, feature_df], axis=1)
    return data
