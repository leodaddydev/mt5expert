"""
services/ohlc_service.py – Validate and transform OHLC data.
Provides a human-readable text summary for the LLM prompt.
"""
from __future__ import annotations

from typing import List

from server.models.schemas import OHLCBar
from server.utils.logging import get_logger

logger = get_logger(__name__)


def validate(ohlc: List[OHLCBar]) -> List[OHLCBar]:
    """
    Basic sanity checks on the OHLC list.
    Returns the validated list or raises ValueError.
    """
    if not ohlc:
        raise ValueError("OHLC data is empty.")

    for i, bar in enumerate(ohlc):
        if bar.high < bar.low:
            raise ValueError(
                f"Bar {i} ({bar.time}): high={bar.high} < low={bar.low}"
            )
        if not (bar.low <= bar.open <= bar.high):
            raise ValueError(
                f"Bar {i} ({bar.time}): open={bar.open} not in [low={bar.low}, high={bar.high}]"
            )
        if not (bar.low <= bar.close <= bar.high):
            raise ValueError(
                f"Bar {i} ({bar.time}): close={bar.close} not in [low={bar.low}, high={bar.high}]"
            )

    logger.debug("OHLC validation passed for %d bars.", len(ohlc))
    return ohlc


def summarize(ohlc: List[OHLCBar], symbol: str = "XAUUSD") -> str:
    """
    Build a concise human-readable text summary of the OHLC data
    to embed in the LLM prompt.
    """
    if not ohlc:
        return "No OHLC data provided."

    closes = [bar.close for bar in ohlc]
    highs  = [bar.high  for bar in ohlc]
    lows   = [bar.low   for bar in ohlc]

    period_high = max(highs)
    period_low  = min(lows)
    first_close = closes[0]
    last_close  = closes[-1]
    price_change = last_close - first_close
    pct_change   = (price_change / first_close * 100) if first_close else 0

    # Simple recent vs older half direction
    mid = len(closes) // 2
    avg_first_half  = sum(closes[:mid]) / mid if mid else last_close
    avg_second_half = sum(closes[mid:]) / (len(closes) - mid) if mid else last_close

    direction = "RISING" if avg_second_half > avg_first_half else "FALLING"
    if abs(avg_second_half - avg_first_half) < 0.05 * period_high * 0.001:
        direction = "SIDEWAYS"

    last_bar = ohlc[-1]

    lines = [
        f"Symbol: {symbol}  |  Timeframe: M5  |  Bars: {len(ohlc)}",
        f"Period: {ohlc[0].time} → {ohlc[-1].time}",
        f"Period High: {period_high:.5f}  |  Period Low: {period_low:.5f}",
        f"First Close: {first_close:.5f}  |  Last Close: {last_close:.5f}",
        f"Net Change: {price_change:+.5f} ({pct_change:+.2f}%)",
        f"Recent Direction: {direction}",
        f"",
        f"Last 5 bars (oldest → newest):",
    ]

    recent = ohlc[-5:] if len(ohlc) >= 5 else ohlc
    for bar in recent:
        body = bar.close - bar.open
        candle_type = "BULL" if body >= 0 else "BEAR"
        lines.append(
            f"  [{bar.time}] O={bar.open:.5f} H={bar.high:.5f} "
            f"L={bar.low:.5f} C={bar.close:.5f} Vol={bar.volume} [{candle_type}]"
        )

    lines += [
        f"",
        f"Full OHLC JSON (all {len(ohlc)} bars):",
        _ohlc_to_compact_json(ohlc),
    ]

    return "\n".join(lines)


def _ohlc_to_compact_json(ohlc: List[OHLCBar]) -> str:
    """Convert OHLC list to a compact JSON string for the prompt."""
    import json

    rows = [
        {
            "t": bar.time,
            "o": round(bar.open, 5),
            "h": round(bar.high, 5),
            "l": round(bar.low, 5),
            "c": round(bar.close, 5),
            "v": bar.volume,
        }
        for bar in ohlc
    ]
    return json.dumps(rows, separators=(",", ":"))
