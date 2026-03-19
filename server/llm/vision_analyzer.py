"""
llm/vision_analyzer.py – High-level chart analysis orchestrator.

Builds the LLM prompt from OHLC summary + optional chart image,
calls Ollama, and parses the structured JSON response into AnalyzeResponse.
"""
from __future__ import annotations

from typing import List, Optional

from server.config import get_settings
from server.llm.ollama_client import OllamaClient, safe_parse_json
from server.models.schemas import AnalyzeResponse, IndicatorContext
from server.utils.logging import get_logger

# Avoid circular import: ScoringResult is only used for type annotation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server.services.scoring import ScoringResult

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert professional gold (XAUUSD) trader with 20+ years of experience.
You analyze M5 candlestick charts and OHLC data to identify high-probability trading opportunities.

When pre-computed indicator data is provided by the trading EA, treat it as a second opinion
from a rule-based system. Your job is to CONFIRM or REJECT that trade idea based on your
own reading of the price action, chart structure, and market context.

Your analysis must always:
1. Identify the current trend (UP, DOWN, or SIDEWAYS)
2. Identify key support and resistance price levels (list up to 5 each)
3. Generate a clear trading signal (BUY, SELL, or NONE)
4. Assign a confidence score between 0.0 (no confidence) and 1.0 (absolute certainty)
5. Give a short, precise reason for your decision

Important: Be conservative. Only output BUY or SELL when you have strong conviction.
If the pre-filter direction conflicts with what you see in price action, output NONE.

You MUST respond with valid JSON only. No markdown, no extra text. Exactly this structure:
{
  "trend": "UP" | "DOWN" | "SIDEWAYS",
  "support": [price1, price2, ...],
  "resistance": [price1, price2, ...],
  "signal": "BUY" | "SELL" | "NONE",
  "confidence": 0.0 to 1.0,
  "reason": "Brief explanation"
}"""


# ── User prompt builder ───────────────────────────────────────────────────────

def _build_prompt(
    ohlc_summary: str,
    has_image: bool,
    indicators: Optional[IndicatorContext] = None,
    scoring_result: "Optional[ScoringResult]" = None,
) -> str:
    lines = [
        "Analyze the following XAUUSD M5 chart data and give me a trading decision.",
        "",
        "=== OHLC DATA ===",
        ohlc_summary,
        "",
    ]

    if has_image:
        lines += [
            "=== CHART IMAGE ===",
            "A screenshot of the M5 chart is attached. Use both the image and OHLC data.",
            "",
        ]

    if indicators is not None:
        trend_label = {1: "UP", -1: "DOWN", 0: "SIDEWAYS"}.get(indicators.h1_trend, "UNKNOWN")
        pattern_label = {0: "None", 1: "Pin Bar", 2: "Engulfing"}.get(indicators.candle_pattern, "Unknown")
        lines += [
            "=== EA PRE-FILTER RESULTS ===",
            "The MQL5 EA has already run rule-based filters and proposes the following trade:",
            f"  Proposed direction : {indicators.direction}",
            f"  EA confidence score: {indicators.score}/100",
            "",
            f"  H1 Trend  : {trend_label} (EMA20={indicators.h1_ema20:.5f}, EMA50={indicators.h1_ema50:.5f})",
            f"  M5 EMA    : EMA20={indicators.m5_ema20:.5f}, EMA50={indicators.m5_ema50:.5f}",
            f"  M5 ATR(14): {indicators.atr_m5:.5f}",
            f"  Pattern   : {pattern_label}",
            f"  Swing SL  : {indicators.swing_sl:.5f}",
            "",
        ]

    if scoring_result is not None:
        f = scoring_result.features
        lines += [
            "=== SERVER SCORING ENGINE RESULTS ===",
            f"Additive score   : {scoring_result.additive_score:+.1f}  (BUY threshold ≥ +7, SELL ≤ -7)",
            f"Rule signal      : {scoring_result.rule_signal}",
            f"Normalised score : {scoring_result.rule_score_norm:.3f} (0–1)",
            "",
            "Score breakdown:",
            f"  Trend    : {scoring_result.trend_score:+.1f}  (H1+M5 EMA direction, weight 30%)",
            f"  S/R      : {scoring_result.sr_score:+.1f}  (distance to support/resistance, weight 30%)",
            f"  Pullback : {scoring_result.pullback_score:+.1f}  (price in EMA zone, weight 20%)",
            f"  Candle   : {scoring_result.candle_score:+.1f}  (body ratio / engulf / pin, weight 20%)",
            f"  Vol      : {scoring_result.vol_score:+.1f}  (ATR relative to average)",
            f"  Session  : {scoring_result.session_score:+.1f}  (London/NY/Asia)",
            f"  Sweep    : {scoring_result.sweep_score:+.1f}  (liquidity sweep detected)",
        ]
        if f is not None:
            lines += [
                "",
                "Key normalised features (all distances divided by ATR):",
                f"  dist_resistance : {f.dist_res_norm:.2f}x ATR  {'⚠ near resistance' if f.near_resistance else ''}",
                f"  dist_support    : {f.dist_sup_norm:.2f}x ATR  {'⚠ near support' if f.near_support else ''}",
                f"  dist_to_ema20   : {f.dist_to_ema_norm:.2f}x ATR  {'✓ in zone' if f.in_ema_zone else ''}{'  ⚠ overextended' if f.overextended else ''}",
                f"  candle_body     : {f.candle_body_ratio:.2f}  (≥0.6 = strong)",
                f"  atr_norm        : {f.atr_norm:.2f}  (vs recent average)",
                f"  sweep_low/high  : {f.sweep_low} / {f.sweep_high}",
            ]
        lines += [
            "",
            "Your task: Review ALL of the above. CONFIRM or REJECT the proposed trade.",
            "Give more weight to S/R distance and trend alignment than to candle patterns alone.",
            "Output NONE if any critical condition is violated (e.g. near_resistance for BUY).",
            "",
        ]
    elif indicators is None:
        lines += [
            "=== TASK ===",
            "Determine if there is a valid BUY, SELL, or NONE signal right now.",
            "",
        ]
    else:
        lines += [
            "Your task: CONFIRM or REJECT this trade idea based on your own chart analysis.",
            "If you agree, output the same direction with your confidence score.",
            "If you disagree or see conflicting signals, output NONE.",
            "",
        ]

    lines += [
        "1. Identify trend, support, and resistance.",
        "2. Output BUY, SELL, or NONE with confidence 0.0–1.0.",
        "3. Provide a brief reason (1-2 sentences).",
        "",
        "Respond with JSON only.",
    ]

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

async def analyze_chart(
    client: OllamaClient,
    ohlc_summary: str,
    image_path: Optional[str] = None,
    indicators: Optional[IndicatorContext] = None,
    scoring_result: "Optional[ScoringResult]" = None,
) -> AnalyzeResponse:
    """
    Call Ollama to analyze chart data and return a structured AnalyzeResponse.

    Args:
        client:       Shared OllamaClient instance (do not create per-request)
        ohlc_summary: Human-readable OHLC text from ohlc_service.summarize()
        image_path:   Path to the saved PNG screenshot (optional)

    Returns:
        AnalyzeResponse populated with LLM output.

    Raises:
        ValueError: If LLM response cannot be parsed.
        httpx.HTTPStatusError: If Ollama returns a non-2xx status.
    """
    settings = get_settings()

    # Choose model: vision model only if image is present AND configured
    use_vision = (
        image_path is not None
        and settings.use_vision_model
    )
    model = settings.ollama_vision_model if use_vision else settings.ollama_model

    prompt = _build_prompt(
        ohlc_summary,
        has_image=use_vision,
        indicators=indicators,
        scoring_result=scoring_result,
    )

    logger.info(
        "Analyzing chart → model=%s  use_vision=%s  image=%s",
        model, use_vision, image_path,
    )

    raw_text = await client.generate(
        model=model,
        prompt=prompt,
        image_path=image_path if use_vision else None,
        system=SYSTEM_PROMPT,
        temperature=0.1,   # low temperature for deterministic output
    )

    # Parse and validate
    data = safe_parse_json(raw_text)
    logger.debug("Parsed LLM JSON: %s", data)

    # Coerce / default missing optional fields
    data.setdefault("support",    [])
    data.setdefault("resistance", [])
    data.setdefault("confidence", 0.5)

    # Clamp confidence
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

    # Normalise to uppercase
    data["trend"]  = str(data.get("trend",  "SIDEWAYS")).upper().strip()
    data["signal"] = str(data.get("signal", "NONE")).upper().strip()

    # Validate enum values
    if data["trend"] not in ("UP", "DOWN", "SIDEWAYS"):
        logger.warning("Unexpected trend value '%s' – defaulting to SIDEWAYS.", data["trend"])
        data["trend"] = "SIDEWAYS"

    if data["signal"] not in ("BUY", "SELL", "NONE"):
        logger.warning("Unexpected signal value '%s' – defaulting to NONE.", data["signal"])
        data["signal"] = "NONE"

    return AnalyzeResponse(**data)
