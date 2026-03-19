"""
services/scoring.py – Multi-feature scoring engine for XAUUSD M5/H1 strategy.

Feature groups (matching strategy spec):
  A. Trend        – H1 + M5 EMA direction & strength
  B. Support/Resistance – H1 fractal levels, distance normalised by ATR
  C. Pullback     – price position relative to EMA zone
  D. Candle/PA    – body ratio, engulfing, pin bar
  E. Volatility   – ATR relative to recent average
  F. Session      – London / NY / Asia
  G. Liquidity sweep – wick-based stop-hunt detection

Scoring formula (weighted):
  rule_score  = 0.30 * trend  + 0.30 * SR  + 0.20 * pullback  + 0.20 * candle
  final_score = 0.70 * rule_score  + 0.30 * (llm_confidence * 10)

Decision thresholds (raw additive score):
  ≥ +7  → BUY
  ≤ -7  → SELL
  else  → NONE
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from server.models.schemas import IndicatorContext, OHLCBar
from server.utils.logging import get_logger

logger = get_logger(__name__)

SignalType = Literal["BUY", "SELL", "NONE"]

# ── Score thresholds ────────────────────────────────────────────────────────
SCORE_BUY_THRESHOLD  =  7
SCORE_SELL_THRESHOLD = -7

# ── Normalisation constants ─────────────────────────────────────────────────
TREND_STRENGTH_MIN   = 0.5   # |EMA20-EMA50| / ATR must exceed this
SR_MIN_DIST          = 0.5   # min ATR multiples to nearest S/R
PULLBACK_MAX_DIST    = 0.5   # dist_to_ema / ATR < this = good pullback
PULLBACK_EXTEND_MAX  = 1.5   # dist_to_ema / ATR > this = overextended
CANDLE_BODY_MIN      = 0.6   # body / range > this = strong candle
ATR_AVG_PERIOD       = 14    # bars to compute average ATR
SR_LOOKBACK          = 40    # H1 bars for S/R detection


# ── Feature dataclass ───────────────────────────────────────────────────────

@dataclass
class FeatureSet:
    """All computed features for one bar evaluation."""

    # A. Trend
    trend_h1:           float = 0.0   # EMA20_H1 - EMA50_H1
    trend_m5:           float = 0.0   # EMA20_M5 - EMA50_M5
    trend_strength_h1:  float = 0.0   # |EMA20_H1-EMA50_H1| / ATR

    # B. S/R
    dist_res_norm:     float = 999.0   # (nearest_res - price) / ATR
    dist_sup_norm:     float = 999.0   # (price - nearest_sup) / ATR
    near_resistance:   bool  = False
    near_support:      bool  = False

    # C. Pullback
    dist_to_ema_norm:  float = 0.0   # |price - EMA20_M5| / ATR
    in_ema_zone:       bool  = False
    overextended:      bool  = False

    # D. Candle
    candle_body_ratio: float = 0.0
    bullish_engulf:    bool  = False
    bearish_engulf:    bool  = False
    bullish_pinbar:    bool  = False
    bearish_pinbar:    bool  = False

    # E. Volatility
    atr_norm:          float = 1.0   # ATR / avg_ATR
    low_volatility:    bool  = False

    # F. Session (injected from IndicatorContext or detected by server)
    in_london:         bool  = False
    in_ny:             bool  = False
    in_asia:           bool  = False

    # G. Liquidity sweep
    sweep_high:        bool  = False   # wick above recent high then close lower
    sweep_low:         bool  = False   # wick below recent low then close higher

    # Proposed direction from EA
    ea_direction:      str   = ""
    ea_score:          int   = 0


@dataclass
class ScoringResult:
    """Scoring output – additive score and component breakdown."""

    # Component raw scores (additive)
    trend_score:    float = 0.0
    sr_score:       float = 0.0
    pullback_score: float = 0.0
    candle_score:   float = 0.0
    vol_score:      float = 0.0
    session_score:  float = 0.0
    sweep_score:    float = 0.0

    # Totals
    additive_score: float = 0.0     # sum of all above
    rule_score_norm: float = 0.0    # weighted, normalised 0–1
    final_score:    float = 0.0     # combined with LLM confidence 0–1

    # Decision before LLM
    rule_signal:    SignalType = "NONE"

    # Feature snapshot (for DB / logging)
    features:       Optional[FeatureSet] = None


# ── EMA helper ──────────────────────────────────────────────────────────────

def _ema(values: List[float], period: int) -> List[float]:
    """Vectorised EMA (same as MetaTrader MODE_EMA / ewm adjust=False)."""
    if not values:
        return []
    result = [values[0]]
    k = 2.0 / (period + 1)
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _last_ema(closes: List[float], period: int) -> Optional[float]:
    e = _ema(closes, period)
    return e[-1] if e else None


# ── ATR helper ──────────────────────────────────────────────────────────────

def _compute_atr_series(ohlc: List[OHLCBar], period: int = 14) -> List[float]:
    """Return ATR series using Wilder smoothing."""
    trs = []
    for i in range(len(ohlc)):
        h = ohlc[i].high
        l = ohlc[i].low
        prev_c = ohlc[i - 1].close if i > 0 else ohlc[i].close
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    # Wilder smooth (EWM alpha = 1/period)
    atr_vals = [trs[0]]
    for tr in trs[1:]:
        atr_vals.append((atr_vals[-1] * (period - 1) + tr) / period)
    return atr_vals


# ── S/R fractal helpers ─────────────────────────────────────────────────────

def _fractal_highs(ohlc: List[OHLCBar]) -> List[float]:
    result = []
    for i in range(1, len(ohlc) - 1):
        if ohlc[i].high > ohlc[i - 1].high and ohlc[i].high > ohlc[i + 1].high:
            result.append(ohlc[i].high)
    return result


def _fractal_lows(ohlc: List[OHLCBar]) -> List[float]:
    result = []
    for i in range(1, len(ohlc) - 1):
        if ohlc[i].low < ohlc[i - 1].low and ohlc[i].low < ohlc[i + 1].low:
            result.append(ohlc[i].low)
    return result


def _nearest_resistance(levels: List[float], price: float) -> Optional[float]:
    above = [l for l in levels if l > price]
    return min(above) if above else None


def _nearest_support(levels: List[float], price: float) -> Optional[float]:
    below = [l for l in levels if l < price]
    return max(below) if below else None


# ── Session detection ────────────────────────────────────────────────────────

def _detect_session(ohlc: List[OHLCBar]) -> Tuple[bool, bool, bool]:
    """
    Detect trading session from last bar timestamp string.
    MT5 format: '2026.03.18 10:05' (broker time, assume UTC offset ~0..+3).
    Returns (london, ny, asia).
    """
    try:
        ts = ohlc[-1].time  # e.g. '2026.03.18 10:05'
        hour = int(ts[11:13])
        london = 7 <= hour < 16
        ny     = 13 <= hour < 22
        asia   = not (london or ny)
        return london, ny, asia
    except Exception:
        return False, False, False


# ── Main feature computation ─────────────────────────────────────────────────

def compute_features(
    ohlc_m5: List[OHLCBar],
    indicators: Optional[IndicatorContext] = None,
    ema_fast: int = 20,
    ema_slow: int = 50,
    atr_period: int = 14,
) -> FeatureSet:
    """
    Compute all features from M5 OHLC + optional pre-computed IndicatorContext.

    When `indicators` is provided (from MQL5 EA), H1 EMA values are taken
    directly from it.  Otherwise they are approximated from M5 data.
    """
    f = FeatureSet()

    if len(ohlc_m5) < ema_slow + atr_period + 2:
        logger.warning("Not enough bars (%d) for full feature computation.", len(ohlc_m5))
        return f

    closes = [b.close for b in ohlc_m5]
    price  = closes[-1]

    # ── ATR ──────────────────────────────────────────────────────────
    atr_series = _compute_atr_series(ohlc_m5, atr_period)
    atr = atr_series[-1]
    avg_atr = statistics.mean(atr_series[-ATR_AVG_PERIOD:]) if len(atr_series) >= ATR_AVG_PERIOD else atr
    if atr <= 0:
        logger.warning("ATR is zero – skipping feature computation.")
        return f

    f.atr_norm      = atr / avg_atr
    f.low_volatility = atr < 0.5   # absolute threshold for XAUUSD

    # ── A. Trend ─────────────────────────────────────────────────────
    if indicators:
        ema20_h1 = indicators.h1_ema20
        ema50_h1 = indicators.h1_ema50
        ema20_m5 = indicators.m5_ema20
        ema50_m5 = indicators.m5_ema50
        f.ea_direction = indicators.direction
        f.ea_score     = indicators.score
    else:
        # Approximate H1 from M5 by sampling every 12 bars
        sampled = closes[::12]
        ema20_h1 = _last_ema(sampled, ema_fast) or price
        ema50_h1 = _last_ema(sampled, ema_slow) or price
        ema20_m5 = _last_ema(closes,  ema_fast) or price
        ema50_m5 = _last_ema(closes,  ema_slow) or price

    f.trend_h1 = ema20_h1 - ema50_h1
    f.trend_m5 = ema20_m5 - ema50_m5
    f.trend_strength_h1 = abs(f.trend_h1) / atr

    # ── B. Support / Resistance (from M5 fractal as H1 proxy) ────────
    sr_window = ohlc_m5[-SR_LOOKBACK:] if len(ohlc_m5) >= SR_LOOKBACK else ohlc_m5
    res_levels = _fractal_highs(sr_window)
    sup_levels = _fractal_lows(sr_window)

    nearest_res = _nearest_resistance(res_levels, price)
    nearest_sup = _nearest_support(sup_levels, price)

    f.dist_res_norm = (nearest_res - price) / atr if nearest_res else 999.0
    f.dist_sup_norm = (price - nearest_sup) / atr if nearest_sup else 999.0

    f.near_resistance = f.dist_res_norm < SR_MIN_DIST
    f.near_support    = f.dist_sup_norm < SR_MIN_DIST

    # ── C. Pullback & EMA zone ────────────────────────────────────────
    zone_top = max(ema20_m5, ema50_m5)
    zone_bot = min(ema20_m5, ema50_m5)

    f.dist_to_ema_norm = abs(price - ema20_m5) / atr
    f.in_ema_zone      = zone_bot <= price <= zone_top
    f.overextended     = f.dist_to_ema_norm > PULLBACK_EXTEND_MAX

    # ── D. Candle patterns (last two bars) ────────────────────────────
    if len(ohlc_m5) >= 2:
        cur  = ohlc_m5[-1]
        prev = ohlc_m5[-2]

        rng  = cur.high - cur.low
        if rng > 0:
            body        = abs(cur.close - cur.open)
            upper_wick  = cur.high - max(cur.open, cur.close)
            lower_wick  = min(cur.open, cur.close) - cur.low

            f.candle_body_ratio = body / rng

            # Engulfing
            f.bullish_engulf = (
                cur.close > cur.open
                and prev.close < prev.open
                and cur.close >= prev.open
                and cur.open  <= prev.close
            )
            f.bearish_engulf = (
                cur.close < cur.open
                and prev.close > prev.open
                and cur.close <= prev.open
                and cur.open  >= prev.close
            )

            # Pin bar
            f.bullish_pinbar = (
                lower_wick >= 2.0 * body
                and lower_wick >= 0.55 * rng
                and cur.close > (cur.low + rng * 0.5)
            )
            f.bearish_pinbar = (
                upper_wick >= 2.0 * body
                and upper_wick >= 0.55 * rng
                and cur.close < (cur.low + rng * 0.5)
            )

    # ── E. Volatility already computed above (atr_norm, low_volatility) ──

    # ── F. Session ────────────────────────────────────────────────────
    f.in_london, f.in_ny, f.in_asia = _detect_session(ohlc_m5)

    # ── G. Liquidity sweep (last bar) ─────────────────────────────────
    if len(ohlc_m5) >= 3:
        recent_high = max(b.high for b in ohlc_m5[-10:-1])
        recent_low  = min(b.low  for b in ohlc_m5[-10:-1])
        cur = ohlc_m5[-1]
        # Sweep high: wick above recent high but close back below it
        f.sweep_high = cur.high > recent_high and cur.close < recent_high
        # Sweep low: wick below recent low but close back above it
        f.sweep_low  = cur.low  < recent_low  and cur.close > recent_low

    return f


# ── Additive scoring ─────────────────────────────────────────────────────────

def _score_direction(f: FeatureSet, direction: str) -> ScoringResult:
    """
    Compute additive score for a given direction (BUY or SELL).
    Uses the point system from the strategy spec.
    """
    r = ScoringResult(features=f)
    is_buy = (direction == "BUY")

    # ── A. Trend ─────────────────────────────────────────────────────
    # H1  +3 / -3
    if f.trend_h1 > 0:
        r.trend_score += 3 if is_buy else -3
    else:
        r.trend_score += -3 if is_buy else 3

    # M5  +2 / -2
    if f.trend_m5 > 0:
        r.trend_score += 2 if is_buy else -2
    else:
        r.trend_score += -2 if is_buy else 2

    # Trend strength  +1
    if f.trend_strength_h1 > TREND_STRENGTH_MIN:
        r.trend_score += 1

    # ── B. S/R ───────────────────────────────────────────────────────
    if is_buy:
        if f.near_resistance:
            r.sr_score -= 5           # too close to resistance
        elif f.dist_res_norm > SR_MIN_DIST:
            r.sr_score += 3           # good distance from resistance
        if f.dist_sup_norm > SR_MIN_DIST:
            r.sr_score += 1           # well above support
    else:
        if f.near_support:
            r.sr_score -= 5           # too close to support
        elif f.dist_sup_norm > SR_MIN_DIST:
            r.sr_score += 3           # good distance from support
        if f.dist_res_norm > SR_MIN_DIST:
            r.sr_score += 1           # well below resistance

    # ── C. Pullback ──────────────────────────────────────────────────
    if f.dist_to_ema_norm < PULLBACK_MAX_DIST:
        r.pullback_score += 2
    if f.in_ema_zone:
        r.pullback_score += 2
    if f.overextended:
        r.pullback_score -= 2

    # ── D. Candle ────────────────────────────────────────────────────
    if f.candle_body_ratio > CANDLE_BODY_MIN:
        r.candle_score += 2
    if is_buy:
        if f.bullish_engulf:  r.candle_score += 2
        if f.bullish_pinbar:  r.candle_score += 2
    else:
        if f.bearish_engulf:  r.candle_score += 2
        if f.bearish_pinbar:  r.candle_score += 2

    # ── E. Volatility ────────────────────────────────────────────────
    if f.atr_norm > 1.0:
        r.vol_score += 1
    if f.low_volatility:
        r.vol_score -= 2

    # ── F. Session ───────────────────────────────────────────────────
    if f.in_london:  r.session_score += 1
    if f.in_ny:      r.session_score += 1
    if f.in_asia:    r.session_score -= 1

    # ── G. Liquidity sweep ────────────────────────────────────────────
    if is_buy  and f.sweep_low:   r.sweep_score += 2
    if not is_buy and f.sweep_high: r.sweep_score += 2

    r.additive_score = (
        r.trend_score + r.sr_score + r.pullback_score
        + r.candle_score + r.vol_score + r.session_score + r.sweep_score
    )
    return r


def _weighted_norm(r: ScoringResult) -> float:
    """
    Convert component scores to a weighted normalised score [0, 1].
    Weights: SR(0.30) + Trend(0.30) + Pullback(0.20) + Candle(0.20)
    Each component normalised to its maximum possible value.
    """
    SR_MAX       = 9.0   # 3 + 5 penalty avoided + 1 = ~9 possible pts
    TREND_MAX    = 6.0   # 3 + 2 + 1
    PULLBACK_MAX = 4.0   # 2 + 2
    CANDLE_MAX   = 6.0   # 2 + 2 + 2

    def clamp01(v: float, mx: float) -> float:
        return max(0.0, min(1.0, v / mx)) if mx else 0.0

    score = (
        0.30 * clamp01(r.sr_score,       SR_MAX)
      + 0.30 * clamp01(r.trend_score,    TREND_MAX)
      + 0.20 * clamp01(r.pullback_score, PULLBACK_MAX)
      + 0.20 * clamp01(r.candle_score,   CANDLE_MAX)
    )
    return round(score, 4)


# ── Public entry points ──────────────────────────────────────────────────────

def compute_rule_signal(
    ohlc: List[OHLCBar],
    indicators: Optional[IndicatorContext] = None,
) -> Tuple[SignalType, float, ScoringResult]:
    """
    Compute rule-based signal and scoring for both BUY and SELL.

    Returns:
        (signal, normalised_score 0-1, ScoringResult for the winning direction)
    """
    features = compute_features(ohlc, indicators)

    # Evaluate both directions; pick the higher additive score
    buy_result  = _score_direction(features, "BUY")
    sell_result = _score_direction(features, "SELL")

    buy_result.rule_score_norm  = _weighted_norm(buy_result)
    sell_result.rule_score_norm = _weighted_norm(sell_result)

    best = buy_result if buy_result.additive_score > sell_result.additive_score else sell_result
    direction = "BUY" if best is buy_result else "SELL"

    if best.additive_score >= SCORE_BUY_THRESHOLD:
        signal = "BUY"
    elif best.additive_score <= SCORE_SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "NONE"

    best.rule_signal = signal

    logger.info(
        "Scoring  BUY_add=%.1f  SELL_add=%.1f  → signal=%s  norm=%.3f",
        buy_result.additive_score, sell_result.additive_score,
        signal, best.rule_score_norm,
    )

    # Return the result for whichever direction was signalled
    # (or best if NONE so LLM still receives useful context)
    return signal, best.rule_score_norm, best


def combine_scores(
    llm_signal: SignalType,
    llm_confidence: float,
    rule_signal: SignalType,
    rule_score: float,
    scoring_result: Optional[ScoringResult] = None,
    llm_weight: float = 0.7,
) -> Tuple[SignalType, float]:
    """
    Combine rule-based and LLM scores.
    Formula: final = 0.7 * rule_norm + 0.3 * (llm_confidence)

    Agreement  → boost confidence
    Conflict   → NONE
    """
    rule_weight = 1.0 - llm_weight

    # Combined score regardless of agreement
    combined = llm_weight * rule_score + rule_weight * llm_confidence

    if llm_signal == rule_signal and llm_signal != "NONE":
        return llm_signal, round(min(combined * 1.1, 1.0), 3)

    if rule_signal == "NONE":
        return llm_signal, round(llm_confidence * llm_weight, 3)

    if llm_signal == "NONE":
        # Only rule engine has a view
        return rule_signal, round(rule_score * rule_weight, 3)

    # Conflict
    logger.warning(
        "Signal conflict: LLM=%s(%.2f) vs Rule=%s(%.2f). → NONE",
        llm_signal, llm_confidence, rule_signal, rule_score,
    )
    return "NONE", round(combined * 0.4, 3)


# ── Feature serialisation (for logging / DB storage) ─────────────────────────

def scoring_result_to_dict(r: ScoringResult) -> dict:
    """Flatten ScoringResult + FeatureSet to a dict for storage or LLM context."""
    d: dict = {
        "rule_signal":      r.rule_signal,
        "additive_score":   round(r.additive_score, 2),
        "rule_score_norm":  round(r.rule_score_norm, 4),
        "final_score":      round(r.final_score, 4),
        "component_trend":    round(r.trend_score, 2),
        "component_sr":       round(r.sr_score, 2),
        "component_pullback": round(r.pullback_score, 2),
        "component_candle":   round(r.candle_score, 2),
        "component_vol":      round(r.vol_score, 2),
        "component_session":  round(r.session_score, 2),
        "component_sweep":    round(r.sweep_score, 2),
    }
    if r.features:
        f = r.features
        d.update({
            "feat_trend_h1":         round(f.trend_h1, 5),
            "feat_trend_m5":         round(f.trend_m5, 5),
            "feat_trend_strength":   round(f.trend_strength_h1, 4),
            "feat_dist_res_norm":    round(f.dist_res_norm, 4),
            "feat_dist_sup_norm":    round(f.dist_sup_norm, 4),
            "feat_near_resistance":  f.near_resistance,
            "feat_near_support":     f.near_support,
            "feat_dist_to_ema_norm": round(f.dist_to_ema_norm, 4),
            "feat_in_ema_zone":      f.in_ema_zone,
            "feat_overextended":     f.overextended,
            "feat_candle_body":      round(f.candle_body_ratio, 4),
            "feat_bullish_engulf":   f.bullish_engulf,
            "feat_bearish_engulf":   f.bearish_engulf,
            "feat_bullish_pinbar":   f.bullish_pinbar,
            "feat_bearish_pinbar":   f.bearish_pinbar,
            "feat_atr_norm":         round(f.atr_norm, 4),
            "feat_low_vol":          f.low_volatility,
            "feat_london":           f.in_london,
            "feat_ny":               f.in_ny,
            "feat_asia":             f.in_asia,
            "feat_sweep_high":       f.sweep_high,
            "feat_sweep_low":        f.sweep_low,
            "feat_ea_direction":     f.ea_direction,
            "feat_ea_score":         f.ea_score,
        })
    return d

