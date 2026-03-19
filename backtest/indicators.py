"""
Technical indicator calculations for the Gold Scalper MTF backtest.
All functions are stateless and operate on pandas Series / DataFrames.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (same as MT5 MODE_EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing (EWM with alpha=1/period).
    Requires columns: high, low, close.
    """
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)

    tr = pd.concat(
        [h - l, (h - c).abs(), (l - c).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def swing_highs(high: pd.Series) -> pd.Series:
    """
    Fractal swing highs: high[i] > high[i-1] AND high[i] > high[i+1].
    Returns a Series with the swing high price where detected, NaN elsewhere.
    """
    h = high.values
    result = np.full(len(h), np.nan)
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            result[i] = h[i]
    return pd.Series(result, index=high.index)


def swing_lows(low: pd.Series) -> pd.Series:
    """
    Fractal swing lows: low[i] < low[i-1] AND low[i] < low[i+1].
    Returns a Series with the swing low price where detected, NaN elsewhere.
    """
    l = low.values
    result = np.full(len(l), np.nan)
    for i in range(1, len(l) - 1):
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            result[i] = l[i]
    return pd.Series(result, index=low.index)
