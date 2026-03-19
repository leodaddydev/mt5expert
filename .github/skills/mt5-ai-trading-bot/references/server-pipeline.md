# Server Pipeline & Schemas — Reference

## FastAPI Entry Point

```python
# server/main.py
app = FastAPI()
app.include_router(analyze.router)
app.include_router(health.router)
# Run: uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

## Config (`server/config.py`)

```python
class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    text_model: str = "deepseek-llm:7b"
    vision_model: str = "llava:7b"
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

## Request Validation Flow

```
POST /analyze
  ↓ body → AnalyzeRequest (Pydantic v2 auto-validates)
  ↓ ohlc  → list[OHLCBar]  (min 10 bars enforced)
  ↓ image → Optional[str]  (base64; None for text-only mode)
  ↓ indicators → Optional[IndicatorContext]  (from GoldScalper EA)
```

## Response Structure

```json
{
  "signal": "BUY",
  "confidence": 0.82,
  "reason": "Strong uptrend confirmed by LLM, score 9/20",
  "rule_signal": "BUY",
  "rule_score": 0.75,
  "combined_score": 0.77,
  "scoring": {
    "rule_signal": "BUY",
    "additive_score": 9,
    "rule_score_norm": 0.75,
    "final_score": 0.77,
    "trend_score": 0.8,
    "sr_score": 0.7,
    "pullback_score": 0.65,
    "candle_score": 0.9,
    "volatility_score": 0.6,
    "session_score": 1.0,
    "sweep_score": 0.5,
    "dist_res_norm": 1.2,
    "dist_sup_norm": 0.3,
    "in_ema_zone": true,
    "candle_body": 0.72,
    ...
  }
}
```

## Image Service (`services/image_service.py`)

```python
def save_base64_image(b64_string: str, output_path: Path) -> Path:
    data = base64.b64decode(b64_string)
    output_path.write_bytes(data)
    return output_path
```

## OHLC Service (`services/ohlc_service.py`)

```python
def render_candlestick(ohlc: list[OHLCBar], output_path: Path) -> Path:
    # Uses mplfinance to render chart PNG
    # Returns path to saved image

def build_ohlc_summary(ohlc: list[OHLCBar]) -> str:
    # Returns last N bars as text summary for LLM prompt
```

## Vision Analyzer (`llm/vision_analyzer.py`)

```python
async def analyze_chart(
    image_path: Path | None,
    ohlc_text: str,
    scoring_result: ScoringResult | None = None
) -> tuple[str, float, str]:
    # Returns (signal, confidence, reason)
    # signal: "BUY" | "SELL" | "NEUTRAL"
    # confidence: 0.0 – 1.0
    # reason: human-readable explanation

def _build_prompt(ohlc_text: str, scoring_result: ScoringResult | None) -> str:
    # Injects scoring breakdown table into system prompt
    # LLM sees: additive_score, component breakdown, key features
```

## Ollama Client (`llm/ollama_client.py`)

```python
async def chat(model: str, messages: list[dict]) -> str:
    # POST ollama_url/api/chat → returns content string

async def generate_with_image(model: str, prompt: str, image_b64: str) -> str:
    # POST ollama_url/api/generate with image → returns response string
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mt5expert
uvicorn server.main:app --reload
```

Key dependencies in `environment.yml`:
- `fastapi`, `uvicorn`, `httpx`, `pydantic`, `pydantic-settings`
- `mplfinance`, `matplotlib`, `pandas`, `pillow`
- `python=3.11`

## Adding a New Endpoint

1. Create `server/routes/my_route.py` with `router = APIRouter()`
2. Define `@router.get/post(...)`
3. Import and mount in `server/main.py`: `app.include_router(my_route.router)`

## Adding a New Pydantic Model

- Place in `server/models/schemas.py`
- Use `model_config = ConfigDict(...)` for v2 configuration
- Import `Optional` from `typing` or use `X | None` syntax (Python 3.10+)
