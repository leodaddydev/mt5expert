"""
models/schemas.py – Pydantic models for request/response validation.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── OHLC Bar ────────────────────────────────────────────────────────────────

class OHLCBar(BaseModel):
    time: str = Field(..., description="ISO timestamp, e.g. '2026.03.18 10:05'")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info) -> float:  # noqa: N805
        data = info.data
        if "open" in data and v < data["open"]:
            raise ValueError("high must be >= open")
        if "close" in data and v < data["close"]:
            raise ValueError("high must be >= close")
        if "low" in data and v < data["low"]:
            raise ValueError("high must be >= low")
        return v


# ── Indicator Context (pre-computed by MQL5 EA) ─────────────────────────────

class IndicatorContext(BaseModel):
    """
    Technical indicators pre-computed by the MQL5 EA.

    The EA runs all rule-based filters (H1 trend, M5 pullback, candle pattern,
    S/R distance) BEFORE calling the server.  These values are included so the
    LLM can use them as context and either confirm or reject the trade idea.
    """
    h1_trend: int = Field(
        ..., description="H1 trend direction: 1=UP, -1=DOWN, 0=SIDEWAYS"
    )
    h1_ema20: float = Field(..., description="H1 EMA20 value")
    h1_ema50: float = Field(..., description="H1 EMA50 value")
    m5_ema20: float = Field(..., description="M5 EMA20 value")
    m5_ema50: float = Field(..., description="M5 EMA50 value")
    atr_m5: float = Field(..., description="M5 ATR(14) value")
    direction: str = Field(
        ..., description="EA's proposed trade direction: BUY or SELL"
    )
    candle_pattern: int = Field(
        ..., description="Confirmation pattern: 0=none, 1=pin bar, 2=engulfing"
    )
    score: int = Field(
        ..., ge=0, le=100,
        description="EA pre-filter confidence score 0–100"
    )
    swing_sl: float = Field(
        ..., description="Swing-based stop-loss reference price computed by EA"
    )


# ── Request ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., examples=["XAUUSD"])
    timeframe: str = Field(..., examples=["M5"])
    ohlc: List[OHLCBar] = Field(..., min_length=1, max_length=500)
    image: Optional[str] = Field(
        None,
        description="Base64-encoded PNG screenshot from MetaTrader 5",
    )
    indicators: Optional[IndicatorContext] = Field(
        None,
        description="Pre-computed indicator values from the MQL5 EA (optional).",
    )

    @field_validator("ohlc")
    @classmethod
    def ohlc_not_empty(cls, v: List[OHLCBar]) -> List[OHLCBar]:
        if not v:
            raise ValueError("ohlc list must not be empty")
        return v


# ── Response ────────────────────────────────────────────────────────────────

class ScoringDetail(BaseModel):
    """
    Detailed scoring breakdown computed by the Python server.
    Included in the response so the MQL5 EA can log it and for debugging.
    """
    rule_signal:       str   = "NONE"
    additive_score:    float = 0.0
    rule_score_norm:   float = 0.0
    final_score:       float = 0.0
    # Component scores
    component_trend:    float = 0.0
    component_sr:       float = 0.0
    component_pullback: float = 0.0
    component_candle:   float = 0.0
    component_vol:      float = 0.0
    component_session:  float = 0.0
    component_sweep:    float = 0.0
    # Key normalised features
    feat_trend_h1:        Optional[float] = None
    feat_trend_m5:        Optional[float] = None
    feat_trend_strength:  Optional[float] = None
    feat_dist_res_norm:   Optional[float] = None
    feat_dist_sup_norm:   Optional[float] = None
    feat_near_resistance: Optional[bool]  = None
    feat_near_support:    Optional[bool]  = None
    feat_dist_to_ema_norm:Optional[float] = None
    feat_in_ema_zone:     Optional[bool]  = None
    feat_overextended:    Optional[bool]  = None
    feat_candle_body:     Optional[float] = None
    feat_bullish_engulf:  Optional[bool]  = None
    feat_bearish_engulf:  Optional[bool]  = None
    feat_bullish_pinbar:  Optional[bool]  = None
    feat_bearish_pinbar:  Optional[bool]  = None
    feat_atr_norm:        Optional[float] = None
    feat_low_vol:         Optional[bool]  = None
    feat_london:          Optional[bool]  = None
    feat_ny:              Optional[bool]  = None
    feat_asia:            Optional[bool]  = None
    feat_sweep_high:      Optional[bool]  = None
    feat_sweep_low:       Optional[bool]  = None


class AnalyzeResponse(BaseModel):
    trend: Literal["UP", "DOWN", "SIDEWAYS"] = Field(
        ..., description="Overall market trend"
    )
    support: List[float] = Field(
        default_factory=list,
        description="Key support price levels",
    )
    resistance: List[float] = Field(
        default_factory=list,
        description="Key resistance price levels",
    )
    signal: Literal["BUY", "SELL", "NONE"] = Field(
        ..., description="Trading signal"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="LLM confidence score"
    )
    reason: str = Field(..., description="Short explanation of the decision")

    # Optional scoring engine fields
    rule_signal: Optional[Literal["BUY", "SELL", "NONE"]] = Field(
        None, description="Rule-based signal from scoring engine"
    )
    rule_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Normalised rule-based score 0–1"
    )
    combined_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Combined rule + LLM score"
    )
    scoring: Optional[ScoringDetail] = Field(
        None, description="Full feature breakdown from scoring engine"
    )

    # Metadata
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    image_path: Optional[str] = None
    chart_path: Optional[str] = None


# ── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_model: str
    vision_enabled: bool
