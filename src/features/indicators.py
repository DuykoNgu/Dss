"""Phase 3: Tính chỉ báo kỹ thuật (trend, momentum, volatility, volume)."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import ta

# Ngưỡng tối thiểu để tính toán (fail-soft, không crash data ngắn).
# LƯU Ý: window lớn nhất là 200 (sma_200) nên với df 50-199 dòng,
# các cột sma_200/sma_50/macd... sẽ NaN ở nhiều dòng đầu — downstream
# phải dropna trước khi train. Giữ 50 thay vì 200 để vẫn tận dụng
# được data ngắn cho các chỉ báo window nhỏ (RSI14, SMA10/20).
MIN_ROWS = 50

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm ~25 cột chỉ báo vào DataFrame OHLCV sạch."""
    if df is None or len(df) < MIN_ROWS:
        n = 0 if df is None else len(df)
        print(f"[indicators] Bỏ qua: chỉ có {n} dòng, cần tối thiểu {MIN_ROWS}.")
        return df

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    df = df.copy()
    # Mọi chỉ báo đều phụ thuộc thứ tự thời gian — không giả định đầu vào đã sort
    df = df.sort_values("time").reset_index(drop=True)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"].astype(float)

    # Xu hướng
    for w in (10, 20, 50, 200):
        df[f"sma_{w}"] = ta.trend.sma_indicator(close, window=w)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Động lượng
    df["rsi"] = ta.momentum.rsi(close, window=14)
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["williams_r"] = ta.momentum.williams_r(high=high, low=low, close=close, lbp=14)

    # Biến động
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"].abs() + 1e-9)
    df["atr"] = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)
    df["atr_pct"] = df["atr"] / (close.abs() + 1e-9) * 100

    # Khối lượng
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["vol_sma_20"] = ta.trend.sma_indicator(volume, window=20)
    df["vol_ratio"] = df["volume"] / (df["vol_sma_20"] + 1e-9)

    return df
