"""Chấm điểm Rule-based trên 5 nhóm chỉ báo kỹ thuật."""

from __future__ import annotations

import pandas as pd

START_SCORE = 50.0
CROSS_LOOKBACK = 20


def _num(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(number) else number


def _recent_flag(series: pd.Series, lookback: int = CROSS_LOOKBACK) -> bool:
    if series is None or series.empty:
        return False
    window = series.iloc[-lookback:] if len(series) > lookback else series
    return bool(window.fillna(False).any())


def calculate_rule_based_score(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Điểm khởi đầu 50, cộng/trừ 5 nhóm, clip [0, 100]."""
    if df is None or df.empty:
        return START_SCORE, ["Không có dữ liệu"]

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    score = START_SCORE
    reasons: list[str] = []

    close = _num(latest.get("close"))
    sma_50 = _num(latest.get("sma_50"))
    sma_200 = _num(latest.get("sma_200"))
    macd = _num(latest.get("macd"))
    macd_signal = _num(latest.get("macd_signal"))
    macd_hist = _num(latest.get("macd_hist"))
    prev_hist = _num(prev.get("macd_hist"))

    if close > sma_50 > sma_200 > 0:
        score += 10
        reasons.append("Uptrend mạnh (Giá > SMA50 > SMA200)")
    elif 0 < close < sma_50 < sma_200:
        score -= 8
        reasons.append("Downtrend (Giá < SMA50 < SMA200)")

    if _recent_flag(df.get("golden_cross")):
        score += 8
        reasons.append("Golden Cross gần đây")
    elif _recent_flag(df.get("death_cross")):
        score -= 10
        reasons.append("Death Cross gần đây")

    if macd > macd_signal and macd_hist > prev_hist:
        score += 7
        reasons.append("MACD tích cực & Histogram mở rộng")
    elif macd < macd_signal:
        score -= 10
        reasons.append("MACD dưới Signal")

    if close > sma_50 > 0:
        score += 5
        reasons.append("Giá trên SMA50")

    rsi = _num(latest.get("rsi"), default=50.0)
    prev_rsi = _num(prev.get("rsi"), default=rsi)
    if 30 <= rsi <= 50 and rsi > prev_rsi:
        score += 10
        reasons.append(f"RSI phục hồi ({rsi:.1f})")
    elif 50 < rsi <= 60:
        score += 7
        reasons.append(f"RSI tích cực ({rsi:.1f})")
    elif rsi > 80:
        score -= 12
        reasons.append(f"RSI quá mua mạnh ({rsi:.1f})")
    elif rsi > 70:
        score -= 8
        reasons.append(f"RSI quá mua ({rsi:.1f})")

    stoch_k = _num(latest.get("stoch_k"), default=50.0)
    stoch_d = _num(latest.get("stoch_d"), default=50.0)
    prev_k = _num(prev.get("stoch_k"), default=stoch_k)
    prev_d = _num(prev.get("stoch_d"), default=stoch_d)
    if stoch_k < 20 and stoch_k > stoch_d and prev_k <= prev_d:
        score += 8
        reasons.append("Stochastic cắt mua vùng Oversold")
    elif stoch_k > 80 and stoch_k < stoch_d:
        score -= 8
        reasons.append("Stochastic cắt bán vùng Overbought")

    vol_ratio = _num(latest.get("vol_ratio"), default=1.0)
    prev_close = _num(prev.get("close"), default=close)
    price_change = (close - prev_close) / (abs(prev_close) + 1e-9)
    if vol_ratio >= 1.5 and price_change > 0:
        score += 10
        reasons.append(f"Volume bùng nổ ({vol_ratio:.1f}x) + Giá tăng")
    elif vol_ratio >= 1.5 and price_change < -0.02:
        score -= 10
        reasons.append(f"Áp lực bán tháo ({vol_ratio:.1f}x)")

    if "obv" in df.columns and len(df) >= 5:
        obv_now = _num(df["obv"].iloc[-1])
        obv_prev = _num(df["obv"].iloc[-5])
        if obv_now > obv_prev:
            score += 5
            reasons.append("OBV tăng (dòng tiền vào)")
        elif obv_now < obv_prev:
            score -= 5
            reasons.append("OBV giảm (dòng tiền ra)")

    bb_pos = _num(latest.get("bb_position"), default=0.5)
    if bb_pos < 0.2 and close > prev_close:
        score += 8
        reasons.append("Bật tăng từ dải dưới BB")
    elif 0.2 <= bb_pos <= 0.5:
        score += 4
        reasons.append("Giá vùng tích lũy BB")
    if bb_pos >= 0.95:
        score -= 5
        reasons.append("Chạm dải trên BB")

    atr_pct = _num(latest.get("atr_pct"), default=3.0)
    if atr_pct < 3.0:
        score += 3
        reasons.append("Biến động thấp")
    elif atr_pct > 5.0:
        score -= 5
        reasons.append("Biến động cao")

    vnindex_vs_sma50 = _num(latest.get("vnindex_vs_sma50"))
    vnindex_return_5d = _num(latest.get("vnindex_return_5d"))
    if vnindex_vs_sma50 > 0:
        score += 5
        reasons.append("VNINDEX trên SMA50")
    else:
        score -= 5
        reasons.append("VNINDEX dưới SMA50")
    if vnindex_return_5d > 1.0:
        score += 3
    elif vnindex_return_5d < -3.0:
        score -= 5
        reasons.append("VNINDEX giảm mạnh tuần qua")

    return max(0.0, min(100.0, score)), reasons
