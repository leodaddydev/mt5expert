"""
config.py – Application settings loaded from environment / .env file.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-llm:7b"          # text-only default
    ollama_vision_model: str = "llava:7b"           # vision model (optional)
    ollama_timeout: float = 120.0                   # seconds
    use_vision_model: bool = False                  # set True if llava is pulled

    # ── Storage ──────────────────────────────────────────────────────
    image_storage_path: str = "data/screenshots"
    chart_render_path: str = "data/charts"

    # ── Server ───────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"

    # ── Config meta ──────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        Path(self.image_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.chart_render_path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
