# Scoring Engine — Full Reference

## `FeatureSet` dataclass fields

| Field | Source | Description |
|-------|--------|-------------|
| `h1_trend` | indicators.h1_trend | 1/0/-1 |
| `m5_trend` | ema20_m5 vs ema50_m5 | 1/0/-1 |
| `trend_strength` | abs(ema20-ema50)/atr | normalised |
| `dist_res_norm` | nearest H1 resistance dist / atr | 0–∞ |
| `dist_sup_norm` | nearest H1 support dist / atr | 0–∞ |
| `near_resistance` | dist_res_norm < 0.2 | bool |
| `near_support` | dist_sup_norm < 0.2 | bool |
| `dist_to_ema` | abs(close - ema20) | price units |
| `in_ema_zone` | close between ema20/ema50 | bool |
| `overextended` | dist_to_ema > 2×atr | bool |
| `body_ratio` | abs(close-open) / (high-low) | 0–1 |
| `is_engulfing` | candle_pattern == 2 | bool |
| `is_pin_bar` | candle_pattern == 1 | bool |
| `atr_norm` | atr_m5 / median_atr_20 | normalised |
| `session` | "london"/"ny"/"asia"/"off" | str |
| `sweep_low` | recent low undercut then closed above | bool |
| `sweep_high` | recent high overshot then closed below | bool |

## `ScoringResult` dataclass

```python
@dataclass
class ScoringResult:
    features: FeatureSet
    additive_score: int          # raw sum (e.g. +9, -5)
    rule_signal: SignalType      # "BUY" | "SELL" | "NEUTRAL"
    # component normalised 0-1 scores
    trend_score: float
    sr_score: float
    pullback_score: float
    candle_score: float
    volatility_score: float
    session_score: float
    sweep_score: float
    rule_score_norm: float       # weighted: 0.30*sr + 0.30*trend + 0.20*pullback + 0.20*candle
    final_score: float           # set by combine_scores()
```

## Decision Thresholds

| Additive score | Signal |
|----------------|--------|
| ≥ +7 | BUY |
| ≤ -7 | SELL |
| -6 to +6 | NEUTRAL |

## Weight Distribution for `rule_score_norm`

```python
rule_score_norm = (
    0.30 * sr_score_01 +
    0.30 * trend_score_01 +
    0.20 * pullback_score_01 +
    0.20 * candle_score_01
)
```
Volatility, Session, Sweep only affect the additive score (not the normalised score), by design — they are confirming factors, not primary drivers.

## `combine_scores` formula

```python
final = llm_weight * rule_score_norm + (1 - llm_weight) * llm_confidence
# default llm_weight = 0.7 (rules dominate, LLM is confirming)
```

Signal priority: if rule_signal != llm_signal → use rule_signal (EA already pre-filtered).

## Tuning Parameters

- S/R distance thresholds: 0.2 (near) and 0.5 (far) — tweak based on broker spread
- ATR multipliers for pullback zone: 0.5 (tight), 2.0 (overextended)
- Additive threshold ±7: lower = more trades, higher = fewer, higher quality
- LLM weight 0.7: increase if trust in rule engine is lower; decrease for more LLM reliance
