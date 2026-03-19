"""
main.py – FastAPI application entry point.

Lifecycle:
  - startup:  initialise shared OllamaClient, configure logging
  - shutdown: close async HTTP client gracefully
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_settings
from server.llm.ollama_client import OllamaClient
from server.routes.analyze import router as analyze_router
from server.routes.health import router as health_router
from server.utils.logging import configure_logging, get_logger


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown context manager."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("main")

    logger.info("Starting MT5Bot server on %s:%d", settings.server_host, settings.server_port)
    logger.info("Ollama model       : %s", settings.ollama_model)
    logger.info("Ollama vision model: %s (enabled=%s)", settings.ollama_vision_model, settings.use_vision_model)
    logger.info("Screenshot storage : %s", settings.image_storage_path)
    logger.info("Chart render path  : %s", settings.chart_render_path)

    # Create shared async HTTP client
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
    )
    app.state.ollama_client = client

    # Verify Ollama is reachable
    if await client.is_healthy():
        logger.info("Ollama is reachable at %s ✓", settings.ollama_base_url)
    else:
        logger.warning(
            "Ollama is NOT reachable at %s – requests will fail until it starts.",
            settings.ollama_base_url,
        )

    yield  # ← app runs here

    logger.info("Shutting down MT5Bot server …")
    await client.aclose()
    logger.info("Ollama HTTP client closed.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="MT5Bot – AI Trading Signal Server",
        description=(
            "Receives OHLC data and chart screenshots from MetaTrader 5, "
            "analyses them with a local Ollama LLM, and returns structured trading signals."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS – allow MT5 EA or any local client
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(analyze_router, tags=["Analysis"])
    app.include_router(health_router,  tags=["Health"])

    return app


app = create_app()


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "server.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
