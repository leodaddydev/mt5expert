"""
llm/vision_analyzer.py – High-level chart analysis orchestrator.

Builds the LLM prompt from OHLC summary + optional chart image,
calls Ollama, and parses the structured JSON response into AnalyzeResponse.
"""
from __future__ import annotations

from typing import List, Optional

from server.config import get_settings
from server.llm.ollama_client import OllamaClient, safe_parse_json
from server.models.schemas import AnalyzeResponse
from server.utils.logging import get_logger

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert professional gold (XAUUSD) trader with 20+ years of experience.
You analyze M5 candlestick charts and OHLC data to identify high-probability trading opportunities.

Your analysis must always:
1. Identify the current trend (UP, DOWN, or SIDEWAYS)
2. Identify key support and resistance price levels (list up to 5 each)
3. Generate a clear trading signal (BUY, SELL, or NONE)
4. Assign a confidence score between 0.0 (no confidence) and 1.0 (absolute certainty)
5. Give a short, precise reason for your decision

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

def _build_prompt(ohlc_summary: str, has_image: bool) -> str:
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
            "A screenshot of the M5 chart is attached. Use both the image and OHLC data for analysis.",
            "",
        ]

    lines += [
        "=== TASK ===",
        "1. Identify trend, support, and resistance from the data above.",
        "2. Determine if there is a valid BUY, SELL, or NONE signal right now.",
        "3. Rate your confidence from 0.0 to 1.0.",
        "4. Provide a brief reason (1-2 sentences).",
        "",
        "Respond with JSON only.",
    ]

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

async def analyze_chart(
    client: OllamaClient,
    ohlc_summary: str,
    image_path: Optional[str] = None,
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

    prompt = _build_prompt(ohlc_summary, has_image=use_vision)

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
