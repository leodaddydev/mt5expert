"""
llm/ollama_client.py – Async HTTP client for the local Ollama API.

Supports:
  - Text-only generation  → POST /api/generate  (no images field)
  - Vision generation     → POST /api/generate  (with base64 images[])
  - Structured JSON output via Ollama's `format` parameter (JSON schema)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from server.utils.logging import get_logger

logger = get_logger(__name__)

# JSON schema Ollama must conform to (matches AnalyzeResponse core fields)
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "trend":      {"type": "string", "enum": ["UP", "DOWN", "SIDEWAYS"]},
        "support":    {"type": "array",  "items": {"type": "number"}},
        "resistance": {"type": "array",  "items": {"type": "number"}},
        "signal":     {"type": "string", "enum": ["BUY", "SELL", "NONE"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason":     {"type": "string"},
    },
    "required": ["trend", "support", "resistance", "signal", "confidence", "reason"],
}


class OllamaClient:
    """
    Async wrapper around the Ollama REST API.
    Re-uses a single httpx.AsyncClient per instance (call .aclose() when done).
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client  = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Low-level generate ──────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        prompt: str,
        image_path: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Call POST /api/generate and return the response text.

        Args:
            model:       Ollama model tag (e.g. 'deepseek-llm:7b', 'llava:7b')
            prompt:      User prompt string
            image_path:  If set, read this PNG and send as base64 in `images[]`
            system:      Optional system prompt (overrides Modelfile default)
            temperature: Sampling temperature (low = deterministic)

        Returns:
            The model's response text string.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
            ValueError: If response JSON is malformed.
        """
        payload: Dict[str, Any] = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {
                "temperature": temperature,
                "seed": 42,          # deterministic outputs
                "num_predict": 512,  # enough for our JSON response
            },
        }

        if system:
            payload["system"] = system

        # Attach image for vision models
        if image_path:
            b64 = _read_image_as_base64(image_path)
            if b64:
                payload["images"] = [b64]
                logger.debug("Attached image to Ollama request (%d chars b64).", len(b64))
            else:
                logger.warning("Could not read image at %s – sending text-only.", image_path)

        logger.info(
            "Calling Ollama  model=%s  vision=%s  prompt_len=%d",
            model, bool(image_path), len(prompt),
        )

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()

        body = response.json()

        if "error" in body:
            raise ValueError(f"Ollama error: {body['error']}")

        text = body.get("response", "")
        logger.info(
            "Ollama response received. eval_count=%s  total_duration_ms=%s",
            body.get("eval_count"),
            _ns_to_ms(body.get("total_duration", 0)),
        )
        return text

    # ── Health check ────────────────────────────────────────────────────────

    async def is_healthy(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_image_as_base64(image_path: str) -> Optional[str]:
    """Read an image file and return its base64-encoded content."""
    p = Path(image_path)
    if not p.exists():
        logger.error("Image file not found: %s", image_path)
        return None
    try:
        raw = p.read_bytes()
        return base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        logger.error("Failed to read image %s: %s", image_path, exc)
        return None


def _ns_to_ms(nanoseconds: int) -> str:
    """Convert Ollama nanosecond durations to a readable ms string."""
    if not nanoseconds:
        return "N/A"
    return f"{nanoseconds / 1_000_000:.0f}ms"


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Safely extract and parse JSON from a model response string.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(inner_lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block
        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Cannot parse JSON from model response:\n{text[:300]}")
