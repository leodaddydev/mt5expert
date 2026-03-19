"""
routes/health.py – GET /health endpoint.
"""
from fastapi import APIRouter, Request

from server.config import get_settings
from server.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request) -> HealthResponse:
    settings  = get_settings()
    client    = request.app.state.ollama_client
    is_up     = await client.is_healthy()

    return HealthResponse(
        status="ok" if is_up else "degraded",
        ollama_model=settings.ollama_model,
        vision_enabled=settings.use_vision_model,
    )
