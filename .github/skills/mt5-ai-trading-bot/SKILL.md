---
name: mt5-ai-trading-bot
description: "Use when working on the MT5 AI Trading Bot project — MQL5 Expert Advisors, Python FastAPI server, hybrid rule-based + LLM scoring engine, multi-timeframe strategy (M5+H1), XAUUSD gold scalping, Ollama LLM integration, chart analysis, feature scoring, backtesting. Triggers: MQL5, MetaTrader, Expert Advisor, GoldScalper, ChartAnalyzer, scoring engine, vision analyzer, indicator context, trade signal."
argument-hint: "Which component to work on? (mql5 / scoring / server / backtest / all)"
---

# MT5 AI Trading Bot

Hybrid trading system: MQL5 EA pre-filters with indicators → Python FastAPI server scores features → LLM (Ollama) confirms with vision → execute trade.

## Project Layout

```
mql5/
  ChartAnalyzer.mq5          # M5 data capture + screenshot → POST /analyze
  ChartAnalyzer_M1.mq5       # M1 variant (same flow, PERIOD_M1, 100 bars)
  GoldScalper_MTF.mq5        # Hybrid EA: pre-filter → AI confirm → trade

server/
  main.py                    # FastAPI app entry point
  config.py                  # Pydantic BaseSettings (URLs, model names)
  models/schemas.py          # Pydantic v2 request/response models
  routes/analyze.py          # POST /analyze pipeline (5 ordered steps)
  routes/health.py           # GET /health
  services/scoring.py        # 7-group feature scoring engine
  services/image_service.py  # Base64 decode → save PNG
  services/ohlc_service.py   # Render mplfinance candlestick chart
  llm/vision_analyzer.py     # Ollama prompt builder + async caller
  llm/ollama_client.py       # Raw httpx Ollama API wrapper
  utils/logging.py           # Structured logger

backtest/
  indicators.py              # EMA, ATR, swing helpers
  strategy.py                # 6 pure strategy functions
  backtest.py                # Full loop + metrics + charts
```

## Core Architecture — Three-Step Flow

```
MQL5 OnTick()
  │
  ├─ Step 1: Pre-filter (all indicators locally)
  │    H1 trend (EMA20/50) → M5 pullback → candle pattern → S/R distance → ComputeScore()
  │    If any filter fails → return (no server call)
  │
  ├─ Step 2: Send to Python server (POST /analyze)
  │    Payload: symbol, timeframe, ohlc[], image(base64), indicators{}
  │    indicators: h1_trend, h1_ema20/50, m5_ema20/50, atr_m5, direction,
  │                candle_pattern(0-2), score(0-100), swing_sl
  │
  └─ Step 3: Parse AI response
       aiSignal == direction && aiConfidence >= MinConfidence → ExecuteTrade()
```

## Python Server Pipeline (`routes/analyze.py`)

```
POST /analyze
  1. Validate OHLC (min bars, OHLC consistency)
  2. Decode + save PNG screenshot
  3. Render mplfinance candlestick chart
  4. Build OHLC text summary string
  5. compute_rule_signal(ohlc, indicators) → ScoringResult   ← BEFORE LLM
  6. analyze_chart(image, ohlc_text, scoring_result)         ← LLM informed of scores
  7. combine_scores(llm_signal, llm_conf, rule_signal, rule_score, scoring_result)
  8. Return AnalyzeResponse with signal, confidence, reason, scoring details
```

## Feature Scoring Engine (`services/scoring.py`)

Seven additive groups. Threshold: **≥ +7 → BUY**, **≤ -7 → SELL**, else NEUTRAL.

| Group | Feature | BUY pts | SELL pts |
|-------|---------|---------|----------|
| A. Trend | H1 EMA20>EMA50 | +3 | -3 |
| A. Trend | M5 EMA20>EMA50 | +2 | -2 |
| A. Trend | Trend strength > 0.5×ATR | +1 | — |
| B. S/R | Price far from resistance (dist_res_norm > 0.5) | +3 | — |
| B. S/R | Near resistance (dist_res_norm < 0.2) | -5 | — |
| B. S/R | Price far from support (dist_sup_norm > 0.5) | — | -3 |
| B. S/R | Near support (dist_sup_norm < 0.2) | — | +5 |
| C. Pullback | dist_to_ema < 0.5×ATR | +2 | +2 |
| C. Pullback | in_ema_zone | +2 | +2 |
| C. Pullback | overextended (> 2×ATR from EMA) | -2 | -2 |
| D. Candle | body_ratio > 0.6 | +2 | +2 |
| D. Candle | engulfing pattern | +2 | +2 |
| D. Candle | pin bar pattern | +2 | +2 |
| E. Volatility | atr_norm > 1.0 | +1 | +1 |
| E. Volatility | low volatility (atr_norm < 0.5) | -2 | -2 |
| F. Session | London (07-16 UTC) | +1 | +1 |
| F. Session | New York (13-22 UTC) | +1 | +1 |
| F. Session | Asia (22-07 UTC) | -1 | -1 |
| G. Sweep | sweep_low detected | +2 | — |
| G. Sweep | sweep_high detected | — | +2 |

**Weighted normalised score** (0–1): `0.30×SR + 0.30×Trend + 0.20×Pullback + 0.20×Candle`

**Final combined score**: `final = 0.7 × rule_score_norm + 0.3 × llm_confidence`

Key function signatures:
```python
compute_rule_signal(ohlc: list[dict], indicators: IndicatorContext | None)
    -> tuple[SignalType, float, ScoringResult]

combine_scores(llm_signal, llm_confidence, rule_signal, rule_score,
               scoring_result=None, llm_weight=0.7)
    -> tuple[SignalType, float]

scoring_result_to_dict(r: ScoringResult) -> dict   # for DB / LLM context injection
```

## Pydantic Models (`models/schemas.py`)

```python
class IndicatorContext(BaseModel):
    h1_trend: int           # 1=UP, -1=DOWN, 0=SIDEWAYS
    h1_ema20: float
    h1_ema50: float
    m5_ema20: float
    m5_ema50: float
    atr_m5: float
    direction: str          # "BUY" | "SELL"
    candle_pattern: int     # 0=none, 1=pin bar, 2=engulfing
    score: int              # MQL5 pre-score 0-100
    swing_sl: float

class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: str
    ohlc: list[OHLCBar]
    image: str | None        # base64 PNG
    indicators: IndicatorContext | None

class AnalyzeResponse(BaseModel):
    signal: str              # "BUY" | "SELL" | "NEUTRAL"
    confidence: float        # 0-1
    reason: str
    rule_signal: str
    rule_score: float        # normalised 0-1
    combined_score: float
    scoring: ScoringDetail | None
```

## LLM Integration (`llm/vision_analyzer.py`)

- Text model: `deepseek-llm:7b`, Vision model: `llava:7b`
- `analyze_chart(image_path, ohlc_text, scoring_result)` → `(signal, confidence, reason)`
- `_build_prompt()` injects a **SERVER SCORING ENGINE RESULTS** section into the system prompt with the full score table so LLM can confirm/reject the rule-based signal
- Ollama runs locally; configure URL in `config.py`

## MQL5 Patterns

**Indicator handles** — always release in `OnDeinit()`:
```mql5
g_hEMA20_M5 = iMA(SYMBOL, TF_M5, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
// ...
IndicatorRelease(g_hEMA20_M5);
```

**Reading buffer value**:
```mql5
double GetBuffer(int handle, int shift) {
    double buf[]; ArraySetAsSeries(buf, true);
    if(CopyBuffer(handle, 0, shift, 1, buf) <= 0) return 0.0;
    return buf[0];
}
```

**Base64 encode file → send via WebRequest**:
```mql5
CryptEncode(CRYPT_BASE64, fileBytes, key, encoded);
string b64 = CharArrayToString(encoded, 0, WHOLE_ARRAY, CP_UTF8);
WebRequest("POST", ServerURL, headers, timeout, postData, responseData, response);
```

**JSON field extraction** (no external lib):
```mql5
string ExtractStringField(const string json, const string key) { ... }
double ExtractDoubleField(const string json, const string key) { ... }
```

**S/R detection** — fractal swing highs/lows on H1:
```mql5
if(highs[i] > highs[i-1] && highs[i] > highs[i+1])  // fractal high → resistance
if(lows[i]  < lows[i-1]  && lows[i]  < lows[i+1])   // fractal low  → support
```

## Technology Stack

| Layer | Tech |
|-------|------|
| EA language | MQL5 (MetaTrader 5) |
| Web server | Python 3.11, FastAPI 0.115+, uvicorn |
| Validation | Pydantic v2, BaseSettings |
| HTTP client | httpx (async) |
| LLM runtime | Ollama (local) |
| Chart rendering | mplfinance, matplotlib, pandas |
| Environment | Conda (`environment.yml`) |

## Common Tasks

**Add a new scoring feature:** Edit `FeatureSet` dataclass → add computation in `compute_features()` → add points in `_score_direction()` → add to `_weighted_norm()` weights if needed → rebuild `ScoringDetail`.

**Create a new timeframe EA:** Copy `ChartAnalyzer_M1.mq5`, change `PERIOD_*`, bar count, filename, log prefix.

**Tune thresholds:** `scoring.py` — additive threshold (±7), normalised weights (0.30/0.30/0.20/0.20), ATR multipliers. `GoldScalper_MTF.mq5` — `MinConfidence`, `SR_DistMultiplier`, `SidewayThreshold`.

**Run backtest:** `python -m backtest.backtest data/XAUUSD_M5.csv data/XAUUSD_H1.csv`

## References

- [Scoring system details](./references/scoring.md)
- [MQL5 indicator & trade patterns](./references/mql5-patterns.md)
- [Server pipeline & schemas](./references/server-pipeline.md)
