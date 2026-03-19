"""
services/scoring.py – Rule-based scoring engine.
Computes SMA crossover + RSI signals from OHLC data and cross-validates
them against the LLM output to produce a combined confidence score.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from server.models.schemas import OHLCBar
from server.utils.logging import get_logger

logger = get_logger(__name__)

SignalType = Literal["BUY", "SELL", "NONE"]


# ── Indicator helpers ────────────────────────────────────────────────────────

def _sma(values: List[float], period: int) -> Optional[float]:
    """Simple moving average of the last `period` values."""
    if len(values) < period:
        return None
    subset = values[-period:]
    return sum(subset) / period


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Wilder's RSI (simplified using EMA of gains/losses).
    Returns None if not enough data.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    # Seed with simple average of first `period` values
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth remaining
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Main scoring function ────────────────────────────────────────────────────

def compute_rule_signal(
    ohlc: List[OHLCBar],
    sma_fast: int = 9,
    sma_slow: int = 21,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> Tuple[SignalType, float]:
    """
    Derive a trading signal from technical indicators.

    Rules:
        BUY  → fast SMA > slow SMA AND RSI < 70 AND RSI > 30 (momentum not exhausted)
        SELL → fast SMA < slow SMA AND RSI > 30 AND RSI < 70
        Override to NONE when RSI is overbought/oversold (exhaustion)

    Returns:
        (signal, confidence) where confidence ∈ [0, 1]
    """
    if len(ohlc) < max(sma_slow, rsi_period) + 2:
        logger.warning(
            "Not enough bars (%d) for rule-based scoring. Need %d.",
            len(ohlc), max(sma_slow, rsi_period) + 2,
        )
        return "NONE", 0.0

    closes = [bar.close for bar in ohlc]

    fast = _sma(closes, sma_fast)
    slow = _sma(closes, sma_slow)
    rsi  = _rsi(closes, rsi_period)

    if fast is None or slow is None or rsi is None:
        return "NONE", 0.0

    logger.debug(
        "Rule engine → SMA%d=%.5f  SMA%d=%.5f  RSI=%.2f",
        sma_fast, fast, sma_slow, slow, rsi,
    )

    # Confidence proportional to SMA separation relative to price
    last_price = closes[-1]
    sma_separation = abs(fast - slow) / last_price if last_price else 0.0
    base_conf = min(sma_separation * 500, 0.8)  # cap at 0.8

    if fast > slow:
        # Uptrend
        if rsi >= rsi_overbought:
            return "NONE", base_conf * 0.3   # overbought – don't buy
        rsi_factor = 1.0 - (rsi - 50) / 50   # higher RSI → lower score
        return "BUY", round(min(base_conf * (0.5 + rsi_factor * 0.5), 1.0), 3)

    elif fast < slow:
        # Downtrend
        if rsi <= rsi_oversold:
            return "NONE", base_conf * 0.3   # oversold – don't sell
        rsi_factor = (rsi - 50) / 50          # higher RSI → higher score for sell
        return "SELL", round(min(base_conf * (0.5 + rsi_factor * 0.5), 1.0), 3)

    return "NONE", 0.0


def combine_scores(
    llm_signal: SignalType,
    llm_confidence: float,
    rule_signal: SignalType,
    rule_score: float,
    llm_weight: float = 0.6,
) -> Tuple[SignalType, float]:
    """
    Combine LLM and rule-based signals into a final decision.

    Agreement  → boost confidence
    Disagreement → reduce confidence, fall back to NONE if severe
    """
    rule_weight = 1.0 - llm_weight

    if llm_signal == rule_signal:
        # Full agreement
        combined = llm_weight * llm_confidence + rule_weight * rule_score
        return llm_signal, round(min(combined * 1.1, 1.0), 3)

    if rule_signal == "NONE":
        # Rule engine abstains – trust LLM at reduced weight
        combined = llm_confidence * llm_weight
        return llm_signal, round(combined, 3)

    if llm_signal == "NONE":
        # LLM abstains – trust rule engine at reduced weight
        combined = rule_score * rule_weight
        return rule_signal, round(combined, 3)

    # Conflict (BUY vs SELL)
    logger.warning(
        "Signal conflict: LLM=%s (%.2f) vs Rule=%s (%.2f). Returning NONE.",
        llm_signal, llm_confidence, rule_signal, rule_score,
    )
    return "NONE", round((llm_confidence * llm_weight + rule_score * rule_weight) * 0.4, 3)
