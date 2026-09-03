import numpy as np

def add_return_lag_features(df, target_col='target', forecast_horizon=5):
    df = df.copy()
    if forecast_horizon <= 5:
        lags = [5, 10, 15, 20, 25, 30]
    elif forecast_horizon <= 20:
        lags = [20, 30, 40, 60]
    else:
        lags = [60, 90, 120, 180, 252]
    for lag in lags:
        if lag < len(df):
            df[f'return_lag_{lag}'] = df[target_col].shift(lag)
    if forecast_horizon <= 5:
        windows = [3, 5, 10, 20]
    elif forecast_horizon <= 20:
        windows = [5, 10, 20, 40, 60]
    else:
        windows = [10, 20, 60, 120, 252]
    for window in windows:
        df[f'return_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'return_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        df[f'return_max_{window}'] = df[target_col].shift(1).rolling(window).max()
        df[f'return_min_{window}'] = df[target_col].shift(1).rolling(window).min()
        df[f'return_cumsum_{window}'] = df[target_col].shift(1).rolling(window).sum()
    if forecast_horizon <= 5:
        vol_windows = [5, 10, 20]
    elif forecast_horizon <= 20:
        vol_windows = [10, 20, 60]
    else:
        vol_windows = [20, 60, 120]
    for window in vol_windows:
        df[f'return_vol_{window}'] = df[target_col].shift(1).rolling(window).std()
    if len(vol_windows) >= 2:
        df['return_vol_ratio'] = (df[f'return_vol_{vol_windows[0]}'] / 
                                   (df[f'return_vol_{vol_windows[1]}'] + 1e-8))
    if forecast_horizon <= 5:
        mom_periods = [(3, 5), (5, 10), (10, 20)]
    elif forecast_horizon <= 20:
        mom_periods = [(5, 10), (10, 20), (20, 60)]
    else:
        mom_periods = [(20, 60), (60, 120), (120, 252)]
    for short, long in mom_periods:
        df[f'return_mom_{short}_{long}'] = (
            df[target_col].shift(1).rolling(short).mean() - 
            df[target_col].shift(1).rolling(long).mean()
        )
    lookback = forecast_horizon * 4
    rolling_mean = df[target_col].shift(1).rolling(lookback).mean()
    rolling_std = df[target_col].shift(1).rolling(lookback).std()
    df[f'return_zscore_{lookback}'] = (
        df[target_col].shift(1) - rolling_mean
    ) / (rolling_std + 1e-8)
    df[f'return_from_peak_{lookback}'] = (
        df[target_col].shift(1) - 
        df[target_col].shift(1).rolling(lookback).max()
    )
    df[f'return_from_trough_{lookback}'] = (
        df[target_col].shift(1) - 
        df[target_col].shift(1).rolling(lookback).min()
    )
    for window in [forecast_horizon, forecast_horizon * 2]:
        recent_returns = df[target_col].shift(1).rolling(window)
        lagged_returns = df[target_col].shift(window + 1).rolling(window)
        correlation_list = []
        for i in range(len(df)):
            if i < window * 2:  
                correlation_list.append(np.nan)
            else:
                recent = df[target_col].iloc[i-window:i].values
                lagged = df[target_col].iloc[i-2*window:i-window].values
                if len(recent) == window and len(lagged) == window:
                    corr = np.corrcoef(recent, lagged)[0, 1]
                    correlation_list.append(corr)
                else:
                    correlation_list.append(np.nan)
        df[f'return_autocorr_{window}'] = correlation_list
    if forecast_horizon >= 20:
        for window in [60, 120]:
            df[f'return_skew_{window}'] = df[target_col].shift(1).rolling(window).skew()
            df[f'return_kurt_{window}'] = df[target_col].shift(1).rolling(window).kurt()
            df[f'return_q25_{window}'] = df[target_col].shift(1).rolling(window).quantile(0.25)
            df[f'return_q75_{window}'] = df[target_col].shift(1).rolling(window).quantile(0.75)
    return df