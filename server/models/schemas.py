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


# ── Request ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., examples=["XAUUSD"])
    timeframe: str = Field(..., examples=["M5"])
    ohlc: List[OHLCBar] = Field(..., min_length=1, max_length=500)
    image: Optional[str] = Field(
        None,
        description="Base64-encoded PNG screenshot from MetaTrader 5",
    )

    @field_validator("ohlc")
    @classmethod
    def ohlc_not_empty(cls, v: List[OHLCBar]) -> List[OHLCBar]:
        if not v:
            raise ValueError("ohlc list must not be empty")
        return v


# ── Response ────────────────────────────────────────────────────────────────

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
        None, description="Rule-based signal from SMA/RSI engine"
    )
    rule_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Rule-based confidence score"
    )
    combined_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Combined LLM + rule-based score"
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
