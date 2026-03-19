"""
Gold Scalper MTF – Python Backtest Engine
==========================================

Usage:
    python -m backtest.backtest <m5_csv> <h1_csv> [options]

    Options:
        --risk      Risk % per trade        (default: 1.0)
        --rr        Risk:Reward ratio       (default: 2.0)
        --balance   Starting balance USD    (default: 10000)
        --sideway   Sideways ATR threshold  (default: 0.3)
        --sr-mult   S/R distance multiplier (default: 0.5)

CSV format (both files must use these column names):
    datetime, open, high, low, close, volume

Example:
    python -m backtest.backtest data/XAUUSD_M5.csv data/XAUUSD_H1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .indicators import ema, atr
from .strategy import (
    detect_trend_h1,
    detect_support_resistance_h1,
    detect_pullback_m5,
    detect_candle_pattern,
    calculate_distance_to_sr,
    generate_signal,
    compute_score,
)

# ---------------------------------------------------------------------------
# Default hyper-parameters (match MQL5 EA defaults)
# ---------------------------------------------------------------------------
EMA_FAST         = 20
EMA_SLOW         = 50
ATR_PERIOD       = 14
RR_RATIO         = 2.0
RISK_PERCENT     = 1.0
INITIAL_BALANCE  = 10_000.0
SIDEWAY_THRESH   = 0.3
SR_DIST_MULT     = 0.5
MIN_ATR_FILTER   = 0.50      # Skip bars with ATR < this value
SWING_LOOKBACK   = 10        # M5 bars back for SL placement
H1_SR_LOOKBACK   = 50        # H1 bars back for S/R detection

# London/NY session filter (UTC hours)
SESSION_RANGES = [(7, 16), (13, 22)]


# ===========================================================================
# Data loading
# ===========================================================================

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load a OHLCV CSV file and normalise column names to lowercase."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ===========================================================================
# Indicator computation
# ===========================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], EMA_FAST)
    df["ema50"] = ema(df["close"], EMA_SLOW)
    df["atr"]   = atr(df, ATR_PERIOD)
    return df


# ===========================================================================
# Align H1 context to each M5 bar via backward asof merge
# ===========================================================================

def align_h1_to_m5(df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most recent H1 bar's indicator values to each M5 row.
    Uses pd.merge_asof with direction='backward' to avoid look-ahead bias.
    The full H1 DataFrame is also carried so S/R windows can be sliced.
    """
    h1_sub = df_h1[["datetime", "ema20", "ema50", "atr"]].copy()
    h1_sub = h1_sub.rename(columns={
        "ema20": "h1_ema20",
        "ema50": "h1_ema50",
        "atr":   "h1_atr",
    })
    merged = pd.merge_asof(
        df_m5.sort_values("datetime"),
        h1_sub.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    return merged.reset_index(drop=True)


# ===========================================================================
# Session filter
# ===========================================================================

def in_session(dt: pd.Timestamp) -> bool:
    h = dt.hour
    return any(start <= h < end for start, end in SESSION_RANGES)


# ===========================================================================
# Main backtest loop
# ===========================================================================

def run_backtest(
    m5_path: str,
    h1_path: str,
    risk_pct: float = RISK_PERCENT,
    rr: float = RR_RATIO,
    initial_balance: float = INITIAL_BALANCE,
    sideway_thresh: float = SIDEWAY_THRESH,
    sr_dist_mult: float = SR_DIST_MULT,
    use_session_filter: bool = True,
) -> dict[str, Any]:
    """
    Run a full vectorised-scan backtest of GoldScalper_MTF strategy.

    Trade simulation:
        - Scan M5 bars sequentially (no look-ahead).
        - On valid signal, find the first future bar that hits SL or TP.
        - Skip to the bar after trade closes before looking for next entry.

    Returns a dict with performance metrics and the trades DataFrame.
    """
    # --- Load & prepare data ------------------------------------------------
    df_m5_raw = load_ohlcv(m5_path)
    df_h1_raw = load_ohlcv(h1_path)

    df_m5 = add_indicators(df_m5_raw)
    df_h1 = add_indicators(df_h1_raw)

    df = align_h1_to_m5(df_m5, df_h1)

    warm_up = max(EMA_SLOW + ATR_PERIOD + 2, H1_SR_LOOKBACK)
    df = df.iloc[warm_up:].reset_index(drop=True)
    df_m5 = df_m5.iloc[warm_up:].reset_index(drop=True)   # aligned view

    # Pre-build a datetime→position lookup for df_h1 (for S/R windowing)
    h1_dt_index = df_h1.set_index("datetime")

    trades:  list[dict] = []
    equity:  list[float] = [initial_balance]
    balance: float = initial_balance

    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]

        # Warm-up guard
        if pd.isna(row["ema20"]) or pd.isna(row["h1_ema20"]):
            i += 1
            continue

        # Session filter
        if use_session_filter and not in_session(row["datetime"]):
            i += 1
            continue

        atr_m5   = float(row["atr"])
        ema20_m5 = float(row["ema20"])
        ema50_m5 = float(row["ema50"])
        ema20_h1 = float(row["h1_ema20"])
        ema50_h1 = float(row["h1_ema50"])

        # ATR / volatility filter
        if atr_m5 < MIN_ATR_FILTER:
            i += 1
            continue

        # 1. H1 trend
        h1_trend = detect_trend_h1(ema20_h1, ema50_h1, atr_m5, sideway_thresh)
        if h1_trend == 0:
            i += 1
            continue

        # 2. H1 S/R from the most recent H1_SR_LOOKBACK bars up to current time
        h1_window = df_h1[df_h1["datetime"] <= row["datetime"]].tail(H1_SR_LOOKBACK)
        if len(h1_window) < 3:
            i += 1
            continue
        sr_levels = detect_support_resistance_h1(h1_window, atr_m5, sr_dist_mult)

        # 3. M5 pullback (use current bar's OHLC)
        pullback_buy  = detect_pullback_m5(row, ema20_m5, ema50_m5, True)
        pullback_sell = detect_pullback_m5(row, ema20_m5, ema50_m5, False)

        # 4. Candle pattern on the current bar (treated as the closed bar)
        pattern_buy  = detect_candle_pattern(df_m5, i, True)
        pattern_sell = detect_candle_pattern(df_m5, i, False)

        # 5. Distance to nearest S/R
        price        = float(row["close"])
        dist_to_res  = calculate_distance_to_sr(sr_levels, price, True)
        dist_to_sup  = calculate_distance_to_sr(sr_levels, price, False)

        # 6. Signal
        signal = generate_signal(
            h1_trend,
            pullback_buy, pullback_sell,
            pattern_buy,  pattern_sell,
            dist_to_res,  dist_to_sup,
            atr_m5,       sr_dist_mult,
        )

        if signal == "NONE":
            i += 1
            continue

        # 7. SL / TP calculation
        lookback_slice = df_m5.iloc[max(0, i - SWING_LOOKBACK) : i]

        if signal == "BUY":
            swing_sl  = float(lookback_slice["low"].min())  if len(lookback_slice) else price - atr_m5
            sl_price  = swing_sl - atr_m5 * 0.2
            entry     = price
            tp_price  = entry + (entry - sl_price) * rr
        else:
            swing_sl  = float(lookback_slice["high"].max()) if len(lookback_slice) else price + atr_m5
            sl_price  = swing_sl + atr_m5 * 0.2
            entry     = price
            tp_price  = entry - (sl_price - entry) * rr

        sl_dist = abs(entry - sl_price)
        if sl_dist < 1e-9:
            i += 1
            continue

        risk_amount = balance * risk_pct / 100.0
        # lot_size expressed as "units of currency per pip" for simplified P&L
        lot_size    = risk_amount / sl_dist

        # 8. Simulate forward until SL or TP hit
        result     = None
        exit_price = None
        exit_i     = i + 1

        while exit_i < len(df):
            future = df.iloc[exit_i]
            if signal == "BUY":
                if float(future["low"])  <= sl_price:
                    result, exit_price = "LOSS", sl_price
                    break
                if float(future["high"]) >= tp_price:
                    result, exit_price = "WIN",  tp_price
                    break
            else:
                if float(future["high"]) >= sl_price:
                    result, exit_price = "LOSS", sl_price
                    break
                if float(future["low"])  <= tp_price:
                    result, exit_price = "WIN",  tp_price
                    break
            exit_i += 1

        if result is None:
            # Trade still open at end of data – skip
            i += 1
            continue

        pnl = (
            (exit_price - entry) * lot_size
            if signal == "BUY"
            else (entry - exit_price) * lot_size
        )
        balance += pnl
        equity.append(balance)

        score = compute_score(
            h1_trend, 1 if signal == "BUY" else -1,
            ema20_m5, ema50_m5, atr_m5,
            pullback_buy if signal == "BUY" else pullback_sell,
            pattern_buy  if signal == "BUY" else pattern_sell,
        )

        trades.append({
            "entry_time":  row["datetime"],
            "exit_time":   df.iloc[exit_i]["datetime"],
            "signal":      signal,
            "entry":       round(entry, 5),
            "sl":          round(sl_price, 5),
            "tp":          round(tp_price, 5),
            "exit_price":  round(exit_price, 5),
            "result":      result,
            "pnl":         round(pnl, 2),
            "balance":     round(balance, 2),
            "score":       score,
            "h1_trend":    h1_trend,
            "pattern":     pattern_buy if signal == "BUY" else pattern_sell,
            "atr":         round(atr_m5, 5),
        })

        i = exit_i + 1   # jump to bar after trade closes

    return _compute_metrics(trades, equity)


# ===========================================================================
# Performance metrics
# ===========================================================================

def _compute_metrics(
    trades: list[dict],
    equity: list[float],
) -> dict[str, Any]:
    if not trades:
        print("No trades were generated. Check your data range and parameters.")
        return {"error": "no_trades"}

    df_t = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "WIN"]
    loses = df_t[df_t["result"] == "LOSS"]

    win_rate       = len(wins) / len(df_t) * 100
    avg_win        = float(wins["pnl"].mean())  if len(wins)  else 0.0
    avg_loss       = float(loses["pnl"].abs().mean()) if len(loses) else 0.0
    gross_profit   = float(wins["pnl"].sum())
    gross_loss     = float(loses["pnl"].abs().sum())
    profit_factor  = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq = pd.Series(equity)
    running_max    = eq.cummax()
    drawdown_pct   = (eq - running_max) / running_max * 100
    max_drawdown   = float(drawdown_pct.min())

    recovery_factor = (
        (equity[-1] - equity[0]) / abs(max_drawdown / 100 * equity[0])
        if max_drawdown < 0 else float("inf")
    )

    metrics = {
        "total_trades":       len(df_t),
        "winning_trades":     int(len(wins)),
        "losing_trades":      int(len(loses)),
        "win_rate_pct":       round(win_rate, 2),
        "profit_factor":      round(profit_factor, 3),
        "max_drawdown_pct":   round(max_drawdown, 2),
        "recovery_factor":    round(recovery_factor, 3),
        "total_pnl":          round(float(df_t["pnl"].sum()), 2),
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "avg_rr_realised":    round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
        "initial_balance":    equity[0],
        "final_balance":      round(equity[-1], 2),
        "return_pct":         round((equity[-1] / equity[0] - 1) * 100, 2),
        "trades":             df_t,
    }

    _print_summary(metrics)
    _save_trades_csv(df_t)
    _plot_results(equity, df_t)

    return metrics


def _print_summary(m: dict) -> None:
    separator = "=" * 42
    print(f"\n{separator}")
    print("  GOLD SCALPER MTF – BACKTEST RESULTS")
    print(separator)
    skip = {"trades"}
    for k, v in m.items():
        if k in skip:
            continue
        print(f"  {k:<26}: {v}")
    print(f"{separator}\n")


def _save_trades_csv(df_t: pd.DataFrame) -> None:
    out = Path("data/backtest_trades.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_t.to_csv(out, index=False)
    print(f"Trade log saved → {out}")


def _plot_results(equity: list[float], df_t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # --- Equity curve -------------------------------------------------------
    axes[0].plot(equity, color="royalblue", linewidth=1.5, label="Equity")
    axes[0].fill_between(range(len(equity)), equity, equity[0],
                         alpha=0.08, color="royalblue")
    axes[0].axhline(equity[0], color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Balance (USD)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --- Per-trade P&L bars -------------------------------------------------
    colors = ["#27ae60" if r == "WIN" else "#e74c3c" for r in df_t["result"]]
    axes[1].bar(range(len(df_t)), df_t["pnl"], color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Trade P&L")
    axes[1].set_ylabel("P&L (USD)")
    axes[1].grid(True, alpha=0.3)

    # --- Drawdown -----------------------------------------------------------
    eq = pd.Series(equity)
    running_max = eq.cummax()
    drawdown    = (eq - running_max) / running_max * 100
    axes[2].fill_between(range(len(drawdown)), drawdown, 0,
                         color="#e74c3c", alpha=0.6, label="Drawdown %")
    axes[2].set_title("Drawdown (%)")
    axes[2].set_ylabel("Drawdown %")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out = Path("data/backtest_result.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Equity chart saved → {out}")
    plt.close()


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gold Scalper MTF – Backtest Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("m5_csv",  help="Path to M5 OHLCV CSV file")
    p.add_argument("h1_csv",  help="Path to H1 OHLCV CSV file")
    p.add_argument("--risk",     type=float, default=RISK_PERCENT,    help="Risk %% per trade")
    p.add_argument("--rr",       type=float, default=RR_RATIO,        help="Risk:Reward ratio")
    p.add_argument("--balance",  type=float, default=INITIAL_BALANCE, help="Starting balance USD")
    p.add_argument("--sideway",  type=float, default=SIDEWAY_THRESH,  help="Sideways ATR threshold")
    p.add_argument("--sr-mult",  type=float, default=SR_DIST_MULT,    help="S/R distance mult")
    p.add_argument("--no-session", action="store_true",               help="Disable session filter")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        m5_path=args.m5_csv,
        h1_path=args.h1_csv,
        risk_pct=args.risk,
        rr=args.rr,
        initial_balance=args.balance,
        sideway_thresh=args.sideway,
        sr_dist_mult=args.sr_mult,
        use_session_filter=not args.no_session,
    )
