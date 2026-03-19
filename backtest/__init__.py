"""
backtest – Gold Scalper MTF backtest package.
"""

from .backtest  import run_backtest
from .strategy  import (
    detect_trend_h1,
    detect_support_resistance_h1,
    detect_pullback_m5,
    detect_candle_pattern,
    calculate_distance_to_sr,
    generate_signal,
    compute_score,
)
from .indicators import ema, atr, swing_highs, swing_lows

__all__ = [
    "run_backtest",
    "detect_trend_h1",
    "detect_support_resistance_h1",
    "detect_pullback_m5",
    "detect_candle_pattern",
    "calculate_distance_to_sr",
    "generate_signal",
    "compute_score",
    "ema",
    "atr",
    "swing_highs",
    "swing_lows",
]
