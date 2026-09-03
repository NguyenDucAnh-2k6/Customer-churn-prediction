import numpy as np

def safe_rolling(series, window, func='mean', min_pct=0.8):
    '''Safe rolling operation.'''
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


def safe_ewm(series, span, min_pct=0.8):
    '''Safe exponential weighted mean.'''
    min_periods = max(int(span * min_pct), 1)
    return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()

def safe_zscore(series, window=252, min_pct=0.8):
    '''Rolling z-score.'''
    min_periods = max(int(window * min_pct), 1)

    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    return (series - rolling_mean) / rolling_std
