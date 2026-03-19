"""
services/image_service.py – Decode base64 screenshot, save to disk,
and optionally render a candlestick chart from OHLC data.
"""
from __future__ import annotations

import base64
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from server.models.schemas import OHLCBar
from server.utils.logging import get_logger

logger = get_logger(__name__)


def decode_and_save(
    b64_image: str,
    symbol: str,
    storage_path: str,
) -> str:
    """
    Decode a base64 PNG string and save it to disk.

    Returns:
        Absolute path of the saved image file.

    Raises:
        ValueError: If decoding fails.
    """
    try:
        # Strip optional data-URI prefix (data:image/png;base64,...)
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_image)
    except Exception as exc:
        raise ValueError(f"Base64 decode failed: {exc}") from exc

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = f"{symbol}_{timestamp}_{uid}.png"

    dest_dir = Path(storage_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    dest_path.write_bytes(image_bytes)
    logger.info("Screenshot saved → %s (%d bytes)", dest_path, len(image_bytes))
    return str(dest_path)


def render_candlestick_chart(
    ohlc: List[OHLCBar],
    symbol: str,
    chart_path: str,
) -> Optional[str]:
    """
    Render OHLC data as a candlestick chart using mplfinance.

    Returns:
        Absolute path of the rendered chart PNG, or None on failure.
    """
    try:
        import mplfinance as mpf
        import pandas as pd

        records = [
            {
                "Date": pd.to_datetime(bar.time),
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            }
            for bar in ohlc
        ]
        df = pd.DataFrame(records).set_index("Date")
        df.index = pd.DatetimeIndex(df.index)

        dest_dir = Path(chart_path)
        dest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = dest_dir / f"{symbol}_chart_{timestamp}.png"

        mc = mpf.make_marketcolors(
            up="green", down="red", inherit=True
        )
        style = mpf.make_mpf_style(
            marketcolors=mc, gridstyle="--", y_on_right=False
        )

        mpf.plot(
            df,
            type="candle",
            style=style,
            title=f"{symbol} M5",
            volume=True,
            savefig=dict(fname=str(out_path), dpi=120, bbox_inches="tight"),
        )

        logger.info("Candlestick chart rendered → %s", out_path)
        return str(out_path)

    except ImportError:
        logger.warning("mplfinance not installed – skipping chart render.")
        return None
    except Exception as exc:
        logger.error("Chart render failed: %s", exc)
        return None
