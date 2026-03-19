"""
Strategy logic functions for Gold Scalper MTF.

All functions are pure (no side effects) and mirror the MQL5 EA logic exactly
so that backtest results are consistent with live execution.

Required function contract (as per strategy spec):
  detect_trend_h1()
  detect_support_resistance_h1()
  detect_pullback_m5()
  detect_candle_pattern()
  calculate_distance_to_sr()
  generate_signal()
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from .indicators import swing_highs, swing_lows


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Trend   = Literal[1, -1, 0]   # 1=UP, -1=DOWN, 0=SIDEWAYS
Signal  = Literal["BUY", "SELL", "NONE"]
Pattern = Literal[0, 1, 2]    # 0=none, 1=pin bar, 2=engulfing


class SRLevel(TypedDict):
    price: float
    type: Literal["resistance", "support"]


# ---------------------------------------------------------------------------
# 1. detect_trend_h1
# ---------------------------------------------------------------------------

def detect_trend_h1(
    ema20: float,
    ema50: float,
    atr_m5: float,
    sideway_threshold: float = 0.3,
) -> Trend:
    """
    Determine H1 trend direction.

    Returns:
        1  – Uptrend   (EMA20 > EMA50, gap significant)
        -1 – Downtrend (EMA20 < EMA50, gap significant)
        0  – Sideways  (gap < sideway_threshold * ATR_M5)
    """
    if atr_m5 <= 0:
        return 0
    if abs(ema20 - ema50) < sideway_threshold * atr_m5:
        return 0
    return 1 if ema20 > ema50 else -1


# ---------------------------------------------------------------------------
# 2. detect_support_resistance_h1
# ---------------------------------------------------------------------------

def detect_support_resistance_h1(
    df_h1: pd.DataFrame,
    atr_m5: float,
    cluster_threshold_mult: float = 0.5,
) -> list[SRLevel]:
    """
    Identify swing-based support/resistance zones on the H1 chart.

    Uses fractal logic: swing high = high[i] > high[i-1] & high[i] > high[i+1].
    Nearby levels (within cluster_threshold_mult * ATR_M5) are merged into one.

    Args:
        df_h1: H1 OHLCV DataFrame with columns [open, high, low, close].
        atr_m5: Current M5 ATR value (used as proximity threshold).
        cluster_threshold_mult: ATR multiplier for clustering nearby levels.

    Returns:
        List of SRLevel dicts sorted by price ascending.
    """
    levels: list[SRLevel] = []
    threshold = cluster_threshold_mult * atr_m5

    sh = swing_highs(df_h1["high"])
    sl = swing_lows(df_h1["low"])

    def _clustered(price: float) -> bool:
        return any(abs(lv["price"] - price) < threshold for lv in levels)

    for price in sh.dropna().values:
        if not _clustered(price):
            levels.append({"price": float(price), "type": "resistance"})

    for price in sl.dropna().values:
        if not _clustered(price):
            levels.append({"price": float(price), "type": "support"})

    levels.sort(key=lambda lv: lv["price"])
    return levels


# ---------------------------------------------------------------------------
# 3. detect_pullback_m5
# ---------------------------------------------------------------------------

def detect_pullback_m5(
    row: pd.Series,
    ema20: float,
    ema50: float,
    is_buy: bool,
) -> bool:
    """
    Check whether price has pulled back into the EMA20–EMA50 zone on M5.

    For BUY : EMA20 > EMA50 and the bar's low touched the zone while
              close remains at or above the bottom of the zone.
    For SELL: EMA20 < EMA50 and the bar's high touched the zone while
              close remains at or below the top of the zone.
    """
    zone_top = max(ema20, ema50)
    zone_bot = min(ema20, ema50)

    if is_buy:
        return (
            ema20 > ema50
            and row["low"]   <= zone_top
            and row["close"] >= zone_bot * 0.998
        )
    else:
        return (
            ema20 < ema50
            and row["high"]  >= zone_bot
            and row["close"] <= zone_top * 1.002
        )


# ---------------------------------------------------------------------------
# 4. detect_candle_pattern
# ---------------------------------------------------------------------------

def detect_candle_pattern(
    df: pd.DataFrame,
    idx: int,
    is_bullish: bool,
    min_body_ratio: float = 0.15,
) -> Pattern:
    """
    Detect pin bar or engulfing pattern on the candle at df.iloc[idx].

    Args:
        df:            M5 OHLCV DataFrame.
        idx:           Index of the CLOSED confirmation candle.
        is_bullish:    True → look for bullish patterns.
        min_body_ratio: Minimum body/range for engulfing (not required for pin).

    Returns:
        0 = no pattern
        1 = pin bar
        2 = engulfing
    """
    if idx < 1 or idx >= len(df):
        return 0

    o1 = df["open"].iloc[idx]
    h1 = df["high"].iloc[idx]
    l1 = df["low"].iloc[idx]
    c1 = df["close"].iloc[idx]

    o2 = df["open"].iloc[idx - 1]
    c2 = df["close"].iloc[idx - 1]

    rng = h1 - l1
    if rng < 1e-9:
        return 0

    body        = abs(c1 - o1)
    upper_wick  = h1 - max(o1, c1)
    lower_wick  = min(o1, c1) - l1

    if is_bullish:
        # Bullish pin bar: long lower wick, close above midpoint
        pin = (
            lower_wick >= 2.0 * body
            and lower_wick >= 0.55 * rng
            and c1 > (l1 + rng * 0.5)
        )
        # Bullish engulfing: prior bearish bar fully engulfed
        engulf = (
            c2 < o2          # prior bearish
            and c1 > o1      # current bullish
            and c1 >= o2     # close above prior open
            and o1 <= c2     # open below prior close
            and body >= min_body_ratio * rng
        )
        if pin:    return 1
        if engulf: return 2
    else:
        # Bearish pin bar: long upper wick, close below midpoint
        pin = (
            upper_wick >= 2.0 * body
            and upper_wick >= 0.55 * rng
            and c1 < (l1 + rng * 0.5)
        )
        # Bearish engulfing
        engulf = (
            c2 > o2
            and c1 < o1
            and c1 <= o2
            and o1 >= c2
            and body >= min_body_ratio * rng
        )
        if pin:    return 1
        if engulf: return 2

    return 0


# ---------------------------------------------------------------------------
# 5. calculate_distance_to_sr
# ---------------------------------------------------------------------------

def calculate_distance_to_sr(
    levels: list[SRLevel],
    price: float,
    is_buy: bool,
) -> float:
    """
    Return the distance from price to the nearest relevant S/R level.

    For BUY : nearest resistance ABOVE price.
    For SELL: nearest support BELOW price.

    Returns float('inf') when no relevant level exists.
    """
    min_dist = float("inf")
    for lv in levels:
        if is_buy and lv["type"] == "resistance" and lv["price"] > price:
            min_dist = min(min_dist, lv["price"] - price)
        elif not is_buy and lv["type"] == "support" and lv["price"] < price:
            min_dist = min(min_dist, price - lv["price"])
    return min_dist


# ---------------------------------------------------------------------------
# 6. generate_signal
# ---------------------------------------------------------------------------

def generate_signal(
    h1_trend: Trend,
    pullback_buy: bool,
    pullback_sell: bool,
    confirm_buy: Pattern,
    confirm_sell: Pattern,
    dist_to_resistance: float,
    dist_to_support: float,
    atr_m5: float,
    sr_dist_mult: float = 0.5,
) -> Signal:
    """
    Combine all filters and return a trading signal.

    BUY  when: H1 UP + M5 pullback + bullish pattern + not near resistance.
    SELL when: H1 DOWN + M5 pullback + bearish pattern + not near support.
    NONE when: none of the above conditions are met.
    """
    min_sr_dist = sr_dist_mult * atr_m5

    if (
        h1_trend == 1
        and pullback_buy
        and confirm_buy > 0
        and dist_to_resistance > min_sr_dist
    ):
        return "BUY"

    if (
        h1_trend == -1
        and pullback_sell
        and confirm_sell > 0
        and dist_to_support > min_sr_dist
    ):
        return "SELL"

    return "NONE"


# ---------------------------------------------------------------------------
# Scoring helper (mirrors EA ComputeScore)
# ---------------------------------------------------------------------------

def compute_score(
    h1_trend: Trend,
    direction: Literal[1, -1],
    ema20_m5: float,
    ema50_m5: float,
    atr_m5: float,
    pullback: bool,
    pattern: Pattern,
) -> int:
    """
    Return a 0–100 confidence score for a proposed trade direction.

    Breakdown:
      30 pts – H1 trend aligned with direction
      25 pts – M5 EMA gap strength
      25 pts – Price pulled back into zone
      20 pts – Candle pattern quality (engulfing > pin bar)
    """
    if h1_trend != direction:
        return 0

    score = 30  # H1 aligned

    gap_ratio = abs(ema20_m5 - ema50_m5) / (atr_m5 if atr_m5 > 0 else 1.0)
    if gap_ratio > 0.5:
        score += 25
    elif gap_ratio > 0.2:
        score += 15
    else:
        score += 5

    if pullback:
        score += 25

    if pattern == 2:    # engulfing
        score += 20
    elif pattern == 1:  # pin bar
        score += 12

    return score
