"""
routes/analyze.py – POST /analyze endpoint.

Full pipeline:
  1. Validate request (Pydantic)
  2. Decode + save screenshot
  3. Validate OHLC
  4. Render candlestick chart (optional)
  5. Call Ollama LLM
  6. Cross-validate with rule-based scoring engine
  7. Return AnalyzeResponse
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from server.config import get_settings
from server.llm.vision_analyzer import analyze_chart
from server.models.schemas import AnalyzeRequest, AnalyzeResponse
from server.services import image_service, ohlc_service, scoring
from server.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze chart data and return a trading signal",
    status_code=status.HTTP_200_OK,
)
async def analyze(request_body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """
    Receive OHLC + optional chart screenshot from MetaTrader 5,
    run LLM analysis, and return a structured trading signal.
    """
    t_start = time.perf_counter()
    settings = get_settings()
    ollama_client = request.app.state.ollama_client

    symbol    = request_body.symbol
    timeframe = request_body.timeframe

    logger.info(
        "Received /analyze request  symbol=%s  timeframe=%s  bars=%d  has_image=%s",
        symbol, timeframe, len(request_body.ohlc), request_body.image is not None,
    )

    # ── 1. Validate OHLC ────────────────────────────────────────────
    try:
        validated_ohlc = ohlc_service.validate(request_body.ohlc)
    except ValueError as exc:
        logger.warning("OHLC validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid OHLC data: {exc}",
        )

    # ── 2. Decode + save screenshot ─────────────────────────────────
    image_path: Optional[str] = None
    if request_body.image:
        try:
            image_path = image_service.decode_and_save(
                request_body.image,
                symbol,
                settings.image_storage_path,
            )
        except ValueError as exc:
            logger.warning("Image decode failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image decode error: {exc}",
            )

    # ── 3. Render candlestick chart (optional) ───────────────────────
    chart_path: Optional[str] = image_service.render_candlestick_chart(
        validated_ohlc,
        symbol,
        settings.chart_render_path,
    )

    # ── 4. Build OHLC text summary ───────────────────────────────────
    ohlc_summary = ohlc_service.summarize(validated_ohlc, symbol)

    # ── 5. Call LLM ──────────────────────────────────────────────────
    try:
        llm_response = await analyze_chart(
            client=ollama_client,
            ohlc_summary=ohlc_summary,
            image_path=image_path,
        )
    except ValueError as exc:
        logger.error("LLM parse error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM response parse error: {exc}",
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama service error: {exc}",
        )

    # ── 6. Rule-based scoring ─────────────────────────────────────────
    rule_signal, rule_score = scoring.compute_rule_signal(validated_ohlc)
    final_signal, combined_score = scoring.combine_scores(
        llm_signal=llm_response.signal,
        llm_confidence=llm_response.confidence,
        rule_signal=rule_signal,
        rule_score=rule_score,
    )

    logger.info(
        "Analysis complete  LLM=%s(%.2f)  Rule=%s(%.2f)  Combined=%s(%.2f)  elapsed=%.2fs",
        llm_response.signal, llm_response.confidence,
        rule_signal, rule_score,
        final_signal, combined_score,
        time.perf_counter() - t_start,
    )

    # ── 7. Build final response ───────────────────────────────────────
    return AnalyzeResponse(
        trend=llm_response.trend,
        support=llm_response.support,
        resistance=llm_response.resistance,
        signal=final_signal,
        confidence=llm_response.confidence,
        reason=llm_response.reason,
        rule_signal=rule_signal,
        rule_score=rule_score,
        combined_score=combined_score,
        symbol=symbol,
        timeframe=timeframe,
        image_path=image_path,
        chart_path=chart_path,
    )
